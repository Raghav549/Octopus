// Octopus Hybrid AI Engine -- APL-style array language + dense token codec.
//
// Honest scope (mirrored in ModuleInfo::limitations):
//   * The evaluator implements a real subset of APL with right-to-left
//     evaluation, scalar extension, reductions, scans, inner/outer products.
//     It is intentionally not a full ISO APL: no nested arrays, no user-defined
//     operators, no ⎕-system functions.
//   * The codec is a lossless varint/pool packer. It reports measured sizes and
//     the Shannon floor of the stream so callers can see how far from entropy
//     optimal it is; it is *not* claimed to be entropy-optimal.
// SPDX-License-Identifier: MIT
#include "octopus/apl.hpp"

#include "octopus/module.hpp"
#include "octopus/numerics.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <sstream>

namespace oct::apl {

namespace {

// All glyphs the lexer recognises. Stored as UTF-8 so the file stays portable.
const char* const kGlyphs[] = {
    "+", "-", "\xc3\x97", "\xc3\xb7", "*", "\xe2\x8c\x88", "\xe2\x8c\x8a", "|", ",", "\xe2\x8d\xb4",
    "\xe2\x8d\xb3", "\xe2\x8c\xbd", "\xe2\x8d\x89", "\xe2\x8d\x8b", "\xe2\x8d\x92", "<", "\xe2\x89\xa4",
    "=", "\xe2\x89\xa5", ">", "\xe2\x89\xa0", "\xe2\x88\xa7", "\xe2\x88\xa8", "\xe2\x88\x98", "\xe2\x8c\xb9",
    "\xe2\x88\xbc", "~", "/", "\\", ".", "\xe2\x86\x90", "\xe2\x8d\x9f"
};
constexpr size_t kGlyphCount = sizeof(kGlyphs) / sizeof(kGlyphs[0]);

bool is_glyph(std::string_view s) {
    for (size_t i = 0; i < kGlyphCount; ++i)
        if (s == kGlyphs[i]) return true;
    return false;
}

std::vector<Token> lex(std::string_view src) {
    std::vector<Token> out;
    size_t i = 0;
    while (i < src.size()) {
        const unsigned char c = static_cast<unsigned char>(src[i]);
        if (std::isspace(c)) { ++i; continue; }
        if (c == '\'') {   // character-vector literal
            size_t j = i + 1;
            std::string s;
            while (j < src.size() && src[j] != '\'') s.push_back(src[j++]);
            if (j >= src.size()) throw OctError("APL: unterminated string literal");
            Token t; t.kind = Token::Kind::String; t.text = s;
            out.push_back(t);
            i = j + 1;
            continue;
        }
        if (std::isdigit(c)) {
            size_t j = i;
            while (j < src.size() && (std::isdigit(static_cast<unsigned char>(src[j])) || src[j] == '.'))
                ++j;
            // exponent
            if (j < src.size() && (src[j] == 'e' || src[j] == 'E')) {
                size_t k = j + 1;
                if (k < src.size() && (src[k] == '+' || src[k] == '-')) ++k;
                size_t d0 = k;
                while (k < src.size() && std::isdigit(static_cast<unsigned char>(src[k]))) ++k;
                if (k > d0) j = k;
            }
            std::string s(src.substr(i, j - i));
            Token t; t.kind = Token::Kind::Number; t.text = s; t.number = std::stod(s);
            out.push_back(t);
            i = j;
            continue;
        }
        if (c == '(') { Token t; t.kind = Token::Kind::LParen; t.text = "("; out.push_back(t); ++i; continue; }
        if (c == ')') { Token t; t.kind = Token::Kind::RParen; t.text = ")"; out.push_back(t); ++i; continue; }
        if (c == ';') { Token t; t.kind = Token::Kind::Semi; t.text = ";"; out.push_back(t); ++i; continue; }
        if (c == '.') { Token t; t.kind = Token::Kind::Dot; t.text = "."; out.push_back(t); ++i; continue; }

        size_t best = 0;
        for (size_t len = 1; len <= 4 && i + len <= src.size(); ++len)
            if (is_glyph(src.substr(i, len))) best = len;
        if (best > 0) {
            const std::string g(src.substr(i, best));
            Token t;
            if (g == "\xe2\x86\x90") t.kind = Token::Kind::Assign;
            else if (g == ".") t.kind = Token::Kind::Dot;
            else t.kind = Token::Kind::Glyph;
            t.text = g;
            out.push_back(t);
            i += best;
            continue;
        }
        if (std::isalpha(c)) {
            size_t j = i;
            while (j < src.size() && (std::isalnum(static_cast<unsigned char>(src[j])) || src[j] == '_'))
                ++j;
            Token t; t.kind = Token::Kind::Name; t.text = std::string(src.substr(i, j - i));
            out.push_back(t);
            i = j;
            continue;
        }
        throw OctError(std::string("APL: unexpected character '") + char(c) + "'");
    }
    return out;
}

using NodePtr = std::shared_ptr<AstNode>;

class Parser {
public:
    explicit Parser(const std::vector<Token>& t) : t_(t) {}

    bool at_end() const { return pos_ >= t_.size(); }

    NodePtr parse_expression() {
        if (pos_ >= t_.size()) throw OctError("APL: unexpected end of input");
        NodePtr lhs = parse_operand();
        if (pos_ < t_.size() && is_function(t_[pos_])) {
            const std::string fn = t_[pos_++].text;
            // outer product: x ∘.f y
            if (fn == "\xe2\x88\x98" && pos_ + 1 < t_.size() &&
                t_[pos_].kind == Token::Kind::Dot && is_function(t_[pos_ + 1])) {
                const std::string g = t_[pos_ + 1].text;
                pos_ += 2;
                NodePtr rhs = parse_expression();
                auto n = std::make_shared<AstNode>();
                n->kind = AstNode::Kind::Dyadic;
                n->text = "outer";
                n->deriv = g;
                n->kids = {lhs, rhs};
                return n;
            }
            // inner product: x f.g y
            if (pos_ + 1 < t_.size() && t_[pos_].kind == Token::Kind::Dot &&
                is_function(t_[pos_ + 1])) {
                const std::string g = t_[pos_ + 1].text;
                pos_ += 2;
                NodePtr rhs = parse_expression();
                auto n = std::make_shared<AstNode>();
                n->kind = AstNode::Kind::Dyadic;
                n->text = "inner";
                n->deriv = g;
                n->kids = {lhs, rhs};
                return n;
            }
            // outer product: x ∘.f y  (lexed as '∘' '.' f)
            std::string deriv;
            if (pos_ < t_.size() && (t_[pos_].text == "/" || t_[pos_].text == "\\")) {
                deriv = t_[pos_].text;
                ++pos_;
                throw OctError("APL: N-wise reduction (x f/ y) is not implemented");
            }
            NodePtr rhs = parse_expression();
            auto n = std::make_shared<AstNode>();
            n->kind = AstNode::Kind::Dyadic;
            n->text = fn;
            n->deriv = deriv;
            n->kids = {lhs, rhs};
            return n;
        }
        return lhs;
    }

private:
    static bool is_function(const Token& t) {
        if (t.kind != Token::Kind::Glyph) return false;
        if (t.text == "/" || t.text == "\\" || t.text == ".") return false;
        return true;
    }

    NodePtr parse_operand() {
        if (pos_ >= t_.size()) throw OctError("APL: expected an operand");
        const Token& tok = t_[pos_];
        if (is_function(tok)) {
            const std::string fn = tok.text;
            ++pos_;
            std::string deriv;
            if (pos_ < t_.size() && (t_[pos_].text == "/" || t_[pos_].text == "\\")) {
                deriv = t_[pos_].text;
                ++pos_;
            }
            NodePtr arg = parse_expression();
            auto n = std::make_shared<AstNode>();
            n->kind = AstNode::Kind::Monadic;
            n->text = fn;
            n->deriv = deriv;
            n->kids = {arg};
            return n;
        }
        if (tok.kind == Token::Kind::LParen) {
            ++pos_;
            NodePtr inner = parse_expression();
            if (pos_ >= t_.size() || t_[pos_].kind != Token::Kind::RParen)
                throw OctError("APL: missing ')'");
            ++pos_;
            if (pos_ < t_.size() && is_function(t_[pos_])) {
                const std::string fn = t_[pos_++].text;
                if (pos_ + 1 < t_.size() && t_[pos_].kind == Token::Kind::Dot &&
                    is_function(t_[pos_ + 1])) {
                    const std::string g = t_[pos_ + 1].text;
                    pos_ += 2;
                    NodePtr rhs = parse_expression();
                    auto n = std::make_shared<AstNode>();
                    n->kind = AstNode::Kind::Dyadic;
                    n->text = "inner";
                    n->deriv = g;
                    n->kids = {inner, rhs};
                    return n;
                }
                NodePtr rhs = parse_expression();
                auto n = std::make_shared<AstNode>();
                n->kind = AstNode::Kind::Dyadic;
                n->text = fn;
                n->kids = {inner, rhs};
                return n;
            }
            return inner;
        }
        return parse_value_sequence();
    }

    // APL strand notation: `2 3 4` is a vector literal. A strand continues while
    // the next token is a value (number, name or string) and no function sits
    // between the items. Parenthesised items inside a strand are not supported.
    NodePtr parse_value_sequence() {
        NodePtr first = parse_simple();
        if (pos_ >= t_.size()) return first;
        auto starts_value = [](const Token& t) {
            return t.kind == Token::Kind::Number || t.kind == Token::Kind::Name ||
                   t.kind == Token::Kind::String;
        };
        if (!starts_value(t_[pos_])) return first;
        auto n = std::make_shared<AstNode>();
        n->kind = AstNode::Kind::Strand;
        n->kids.push_back(first);
        while (pos_ < t_.size() && starts_value(t_[pos_])) n->kids.push_back(parse_simple());
        return n;
    }

    NodePtr parse_simple() {
        if (pos_ >= t_.size()) throw OctError("APL: expected a value");
        const Token& tok = t_[pos_++];
        auto n = std::make_shared<AstNode>();
        switch (tok.kind) {
            case Token::Kind::Number: n->kind = AstNode::Kind::Num; n->number = tok.number; break;
            case Token::Kind::String: n->kind = AstNode::Kind::Str; n->text = tok.text; break;
            case Token::Kind::Name:   n->kind = AstNode::Kind::Name; n->text = tok.text; break;
            default: throw OctError("APL: unexpected token '" + tok.text + "'");
        }
        return n;
    }

    const std::vector<Token>& t_;
    size_t pos_ = 0;
};

// --------------------------------------------------------------------------
// Evaluation helpers
// --------------------------------------------------------------------------
using Vec = std::vector<double>;
using BinFn = double (*)(double, double);

Vec to_vec(const Array& a) {
    Vec v(a.size());
    for (size_t i = 0; i < a.size(); ++i) v[i] = a.scalar_f64(i);
    return v;
}

BinFn dyadic_fn(const std::string& g) {
    if (g == "+") return [](double a, double b) { return a + b; };
    if (g == "-") return [](double a, double b) { return a - b; };
    if (g == "\xc3\x97" || g == "*") return [](double a, double b) { return a * b; };
    if (g == "\xc3\xb7") return [](double a, double b) { return a / b; };
    if (g == "\xe2\x8c\x88") return [](double a, double b) { return std::max(a, b); };  // ⌈
    if (g == "\xe2\x8c\x8a") return [](double a, double b) { return std::min(a, b); };  // ⌊
    if (g == "|") return [](double a, double b) { return std::fmod(a, b); };             // residue
    if (g == "=") return [](double a, double b) { return a == b ? 1.0 : 0.0; };
    if (g == "\xe2\x89\xa0") return [](double a, double b) { return a != b ? 1.0 : 0.0; };
    if (g == "<") return [](double a, double b) { return a < b ? 1.0 : 0.0; };
    if (g == "\xe2\x89\xa4") return [](double a, double b) { return a <= b ? 1.0 : 0.0; };
    if (g == ">") return [](double a, double b) { return a > b ? 1.0 : 0.0; };
    if (g == "\xe2\x89\xa5") return [](double a, double b) { return a >= b ? 1.0 : 0.0; };
    if (g == "\xe2\x88\xa7") return [](double a, double b) { return (a != 0.0 && b != 0.0) ? 1.0 : 0.0; };
    if (g == "\xe2\x88\xa8") return [](double a, double b) { return (a != 0.0 || b != 0.0) ? 1.0 : 0.0; };
    throw OctError("APL: unknown dyadic function '" + g + "'");
}

Array elementwise(const Array& a, const Array& b, BinFn f, const std::string& name) {
    const bool sa = a.size() == 1, sb = b.size() == 1;
    if (!sa && !sb && a.shape() != b.shape())
        throw OctError("APL: shape mismatch in '" + name + "': " + ShapeSpec{a.shape()}.to_string() +
                       " vs " + ShapeSpec{b.shape()}.to_string());
    const size_t n = std::max(a.size(), b.size());
    Vec out(n);
    for (size_t i = 0; i < n; ++i)
        out[i] = f(a.scalar_f64(sa ? 0 : i), b.scalar_f64(sb ? 0 : i));
    Index shape = sa ? b.shape() : a.shape();
    if (sa && sb) shape = {};
    return Array::from_f64(std::move(out), shape);
}

Array monadic(const std::string& g, const Array& x) {
    const Index sh = x.shape();
    Vec v = to_vec(x);
    if (g == "+") return x;   // monadic + is the identity on reals
    if (g == "-") { for (auto& e : v) e = -e; return Array::from_f64(std::move(v), sh); }
    if (g == "\xc3\xb7") { for (auto& e : v) e = 1.0 / e; return Array::from_f64(std::move(v), sh); }
    if (g == "*") { for (auto& e : v) e = std::exp(e); return Array::from_f64(std::move(v), sh); }
    if (g == "\xe2\x8c\x88") { for (auto& e : v) e = std::ceil(e); return Array::from_f64(std::move(v), sh); }
    if (g == "\xe2\x8c\x8a") { for (auto& e : v) e = std::floor(e); return Array::from_f64(std::move(v), sh); }
    if (g == "|") { for (auto& e : v) e = std::abs(e); return Array::from_f64(std::move(v), sh); }
    if (g == "\xe2\x8d\xb4") {   // ⍴ shape
        Vec s;
        for (int64_t d : sh) s.push_back(double(d));
        return Array::from_f64(std::move(s), {int64_t(sh.size())});
    }
    if (g == ",") return Array::from_f64(std::move(v), {int64_t(v.size())});   // ravel
    if (g == "\xe2\x8d\xb3") {   // ⍳ 0..n-1
        if (x.size() != 1) throw OctError("APL: monadic ⍳ expects a scalar");
        const int64_t n = int64_t(x.scalar_f64());
        if (n < 0) throw OctError("APL: ⍳ of a negative value");
        Vec out(static_cast<size_t>(n));
        for (int64_t i = 0; i < n; ++i) out[size_t(i)] = double(i);
        return Array::from_f64(std::move(out), {n});
    }
    if (g == "\xe2\x8c\xbd") {   // ⌽ reverse
        std::reverse(v.begin(), v.end());
        return Array::from_f64(std::move(v), sh);
    }
    if (g == "\xe2\x8d\x89") {   // ⍉ transpose of a matrix, reverse of a vector
        if (sh.size() != 2) {
            std::reverse(v.begin(), v.end());
            return Array::from_f64(std::move(v), sh);
        }
        const int64_t r = sh[0], c = sh[1];
        Vec out(static_cast<size_t>(r * c));
        for (int64_t i = 0; i < r; ++i)
            for (int64_t j = 0; j < c; ++j)
                out[size_t(j) * size_t(r) + size_t(i)] = v[size_t(i) * size_t(c) + size_t(j)];
        return Array::from_f64(std::move(out), {c, r});
    }
    if (g == "\xe2\x8d\x8b" || g == "\xe2\x8d\x92") {   // ⍋ grade up / ⍒ grade down
        const bool up = (g == "\xe2\x8d\x8b");
        std::vector<int64_t> idx(v.size());
        for (size_t i = 0; i < idx.size(); ++i) idx[i] = int64_t(i);
        std::stable_sort(idx.begin(), idx.end(), [&](int64_t a, int64_t b) {
            return up ? v[size_t(a)] < v[size_t(b)] : v[size_t(a)] > v[size_t(b)];
        });
        Vec out(idx.size());
        for (size_t i = 0; i < idx.size(); ++i) out[i] = double(idx[i]);
        return Array::from_f64(std::move(out), {int64_t(idx.size())});
    }
    if (g == "~") {
        for (auto& e : v) e = (e == 0.0) ? 1.0 : 0.0;
        return Array::from_f64(std::move(v), sh);
    }
    throw OctError("APL: unknown monadic function '" + g + "'");
}

Array dyadic(const std::string& g, const Array& a, const Array& b) {
    if (g == "\xe2\x8d\xb4") {   // reshape
        if (a.size() == 0 || b.size() == 0) throw OctError("APL: ⍴ needs two non-empty arguments");
        Index target(a.size());
        for (size_t i = 0; i < a.size(); ++i) target[i] = int64_t(a.scalar_f64(i));
        Vec src = to_vec(b);
        size_t need = 1;
        for (int64_t d : target) {
            if (d < 0) throw OctError("APL: ⍴ with a negative axis");
            need *= size_t(d);
        }
        Vec out(need);
        for (size_t i = 0; i < need; ++i) out[i] = src[i % src.size()];
        return Array::from_f64(std::move(out), target);
    }
    if (g == ",") {   // catenate
        Vec x = to_vec(a), y = to_vec(b);
        x.insert(x.end(), y.begin(), y.end());
        return Array::from_f64(std::move(x), {int64_t(x.size())});
    }
    if (g == "\xe2\x8d\xb3") {   // dyadic ⍳: 1-origin index, 0 = not found
        const Vec universe = to_vec(a);
        const Vec y = to_vec(b);
        Vec out(y.size());
        for (size_t i = 0; i < y.size(); ++i) {
            int64_t found = 0;
            for (size_t k = 0; k < universe.size(); ++k)
                if (universe[k] == y[i]) { found = int64_t(k) + 1; break; }
            out[i] = double(found);
        }
        return Array::from_f64(std::move(out), b.shape());
    }
    if (g == "\xe2\x8c\xb9") {   // ⌹ solve/least-squares; LU via numerics::linalg
        if (a.shape().size() != 2) throw OctError("APL: ⌹ expects a matrix on the left");
        const int64_t n = a.shape()[0];
        if (a.shape()[1] != n) throw OctError("APL: ⌹ requires a square matrix");
        const Vec A = to_vec(a);
        const Vec rhs = to_vec(b);
        if (rhs.size() % size_t(n) != 0) throw OctError("APL: ⌹ right-argument shape mismatch");
        const int64_t cols = int64_t(rhs.size()) / n;
        Vec result(static_cast<size_t>(n) * static_cast<size_t>(cols));
        for (int64_t c = 0; c < cols; ++c) {
            std::vector<double> bc;
            bc.resize(static_cast<size_t>(n));
            for (int64_t i = 0; i < n; ++i) bc[size_t(i)] = rhs[size_t(i * cols + c)];
            double residual = 0.0;
            auto sol = numerics::linalg::solve(A, bc, n, &residual);
            if (!sol) throw OctError("APL: ⌹ failed: " + sol.status.message);
            for (int64_t i = 0; i < n; ++i) result[size_t(i * cols + c)] = (*sol)[size_t(i)];
        }
        return Array::from_f64(std::move(result), {n, cols});
    }
    return elementwise(a, b, dyadic_fn(g), g);
}

Array reduce_or_scan(const std::string& g, const Array& x, char kind) {
    const Index sh = x.shape();
    if (sh.empty()) return x;
    const int64_t axis_len = sh.back();
    if (axis_len <= 0) throw OctError("APL: reduction over an empty axis");
    const int64_t outer = int64_t(x.size()) / axis_len;
    BinFn f = dyadic_fn(g);

    if (kind == '/') {
        Index out_shape = sh;
        out_shape.pop_back();
        Vec out(size_t(outer), 0.0);
        for (int64_t o = 0; o < outer; ++o) {
            double acc = x.scalar_f64(size_t(o) * size_t(axis_len));
            for (int64_t k = 1; k < axis_len; ++k)
                acc = f(acc, x.scalar_f64(size_t(o) * size_t(axis_len) + size_t(k)));
            out[size_t(o)] = acc;
        }
        return Array::from_f64(std::move(out), out_shape);
    }
    Vec scan(static_cast<size_t>(x.size()));
    for (int64_t o = 0; o < outer; ++o) {
        double acc = x.scalar_f64(size_t(o) * size_t(axis_len));
        scan[size_t(o) * size_t(axis_len)] = acc;
        for (int64_t k = 1; k < axis_len; ++k) {
            acc = f(acc, x.scalar_f64(size_t(o) * size_t(axis_len) + size_t(k)));
            scan[size_t(o) * size_t(axis_len) + size_t(k)] = acc;
        }
    }
    return Array::from_f64(std::move(scan), sh);
}

Array outer_product(const std::string& g, const Array& a, const Array& b) {
    if (a.shape().size() > 1 || b.shape().size() > 1)
        throw OctError("APL: ∘. supports vector arguments in this implementation");
    BinFn f = dyadic_fn(g);
    const Vec x = to_vec(a), y = to_vec(b);
    Vec out(x.size() * y.size());
    for (size_t i = 0; i < x.size(); ++i)
        for (size_t j = 0; j < y.size(); ++j)
            out[i * y.size() + j] = f(x[i], y[j]);
    return Array::from_f64(std::move(out), {int64_t(x.size()), int64_t(y.size())});
}

Array inner_product(const std::string& add_g, const std::string& mul_g, const Array& a,
                    const Array& b) {
    if (a.shape().size() != 2 || b.shape().size() != 2)
        throw OctError("APL: inner product requires matrices");
    const int64_t n = a.shape()[0], k = a.shape()[1], m = b.shape()[1];
    if (b.shape()[0] != k) throw OctError("APL: inner product shape mismatch");
    BinFn add = dyadic_fn(add_g);
    BinFn mul = dyadic_fn(mul_g);
    Vec out(size_t(n) * size_t(m), 0.0);
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < m; ++j) {
            double acc = mul(a.scalar_f64(size_t(i) * size_t(k)), b.scalar_f64(size_t(j)));
            for (int64_t p = 1; p < k; ++p)
                acc = add(acc, mul(a.scalar_f64(size_t(i) * size_t(k) + size_t(p)),
                                   b.scalar_f64(size_t(p) * size_t(m) + size_t(j))));
            out[size_t(i) * size_t(m) + size_t(j)] = acc;
        }
    return Array::from_f64(std::move(out), {n, m});
}

Array eval_node(const NodePtr& node, Environment& env, EvalTrace* trace) {
    if (!node) throw OctError("APL: null node");
    switch (node->kind) {
        case AstNode::Kind::Num: return Array::from_f64({node->number}, {});
        case AstNode::Kind::Str: {
            std::vector<uint8_t> bytes(node->text.begin(), node->text.end());
            return Array::from_u8(std::move(bytes), {int64_t(node->text.size())});
        }
        case AstNode::Kind::Name: {
            auto it = env.vars.find(node->text);
            if (it == env.vars.end())
                throw OctError("APL: undefined name '" + node->text + "'");
            return it->second;
        }
        case AstNode::Kind::Strand: {
            Vec out;
            for (const auto& k : node->kids) {
                const Vec v = to_vec(eval_node(k, env, trace));
                out.insert(out.end(), v.begin(), v.end());
            }
            if (trace) trace->steps.push_back("strand");
            return Array::from_f64(std::move(out), {int64_t(out.size())});
        }
        case AstNode::Kind::Monadic: {
            Array x = eval_node(node->kids[0], env, trace);
            if (trace) trace->steps.push_back("monadic " + node->text);
            const char deriv = node->deriv.empty() ? 0 : node->deriv[0];
            Array out = (deriv == 0) ? monadic(node->text, x)
                                     : reduce_or_scan(node->text, x, deriv);
            if (trace) ++trace->primitive_count;
            return out;
        }
        case AstNode::Kind::Dyadic: {
            Array a = eval_node(node->kids[0], env, trace);
            Array b = eval_node(node->kids[1], env, trace);
            if (node->text == "inner") {
                if (trace) trace->steps.push_back("inner product (+." + node->deriv + ")");
                if (trace) ++trace->primitive_count;
                return inner_product("+", node->deriv, a, b);
            }
            if (node->text == "outer") {
                if (trace) trace->steps.push_back("outer product (\xe2\x88\x98." + node->deriv + ")");
                if (trace) ++trace->primitive_count;
                return outer_product(node->deriv, a, b);
            }
            if (trace) trace->steps.push_back("dyadic " + node->text);
            if (trace) ++trace->primitive_count;
            return dyadic(node->text, a, b);
        }
        case AstNode::Kind::Derived:
            throw OctError("APL: unsupported derived node");
    }
    throw OctError("APL: unreachable node kind");
}

}  // namespace

// ---------------------------------------------------------------------------
// Public evaluation API
// ---------------------------------------------------------------------------
Outcome<Program> parse(std::string_view source) {
    Program p;
    p.source = std::string(source);
    try {
        p.tokens = lex(source);
        std::vector<Token> body = p.tokens;
        if (body.size() >= 2 && body[0].kind == Token::Kind::Name &&
            body[1].kind == Token::Kind::Assign) {
            p.target = body[0].text;
            body.erase(body.begin(), body.begin() + 2);
        }
        for (size_t i = 0; i < body.size(); ++i)
            if (body[i].kind == Token::Kind::Semi) { body.resize(i); break; }
        while (!body.empty() && body.back().kind == Token::Kind::Semi) body.pop_back();
        if (body.empty()) return Status::invalid("APL: empty expression");
        Parser parser(body);
        p.root = parser.parse_expression();
        if (!parser.at_end()) return Status::invalid("APL: trailing tokens after expression");
        return p;
    } catch (const std::exception& e) {
        return Status::invalid(e.what());
    }
}

Json Program::to_json() const {
    Json j;
    j.begin_object();
    j.field("source", source);
    if (!target.empty()) j.field("target", target);
    j.field("tokens", int64_t(tokens.size()));
    j.key("token_kinds");
    j.begin_array();
    for (const auto& t : tokens) {
        switch (t.kind) {
            case Token::Kind::Number: j.value("number"); break;
            case Token::Kind::String: j.value("string"); break;
            case Token::Kind::Name:   j.value("name"); break;
            case Token::Kind::Glyph:  j.value("glyph"); break;
            case Token::Kind::Assign: j.value("assign"); break;
            case Token::Kind::LParen: j.value("lparen"); break;
            case Token::Kind::RParen: j.value("rparen"); break;
            case Token::Kind::Dot:    j.value("dot"); break;
            case Token::Kind::Semi:   j.value("semicolon"); break;
        }
    }
    j.end_array();
    j.end_object();
    return j;
}

std::string Program::to_apl() const {
    std::string out;
    std::function<void(const NodePtr&)> dump = [&](const NodePtr& n) {
        if (!n) return;
        switch (n->kind) {
            case AstNode::Kind::Num: {
                char b[40];
                std::snprintf(b, sizeof(b), "%g", n->number);
                out += b;
                break;
            }
            case AstNode::Kind::Str:  out += "'" + n->text + "'"; break;
            case AstNode::Kind::Name: out += n->text; break;
            case AstNode::Kind::Strand:
                for (size_t i = 0; i < n->kids.size(); ++i) {
                    if (i) out += " ";
                    dump(n->kids[i]);
                }
                break;
            case AstNode::Kind::Monadic: out += n->text; dump(n->kids[0]); break;
            case AstNode::Kind::Dyadic:
                dump(n->kids[0]);
                out += n->text;
                dump(n->kids[1]);
                break;
            case AstNode::Kind::Derived: out += n->text; break;
        }
    };
    dump(root);
    return out;
}

Outcome<Array> eval(const Program& p, Environment& env, EvalTrace* trace) {
    try {
        Array out = eval_node(p.root, env, trace);
        if (!p.target.empty()) env.assign(p.target, out);
        return out;
    } catch (const std::exception& e) {
        return Status::invalid(e.what());
    }
}

Outcome<Array> eval_line(std::string_view src, Environment& env, EvalTrace* trace) {
    auto p = parse(src);
    if (!p) return p.status;
    return eval(*p, env, trace);
}

// ---------------------------------------------------------------------------
// Dense token-stream codec
// ---------------------------------------------------------------------------
Dictionary Dictionary::from_symbols(const std::vector<std::string>& syms) {
    Dictionary d;
    for (const auto& s : syms) {
        if (d.index.count(s)) continue;
        d.index[s] = uint32_t(d.symbols.size());
        d.symbols.push_back(s);
    }
    return d;
}

const Dictionary& Dictionary::apl_pool() {
    static const Dictionary pool = [] {
        std::vector<std::string> syms;
        for (size_t i = 0; i < kGlyphCount; ++i) syms.push_back(kGlyphs[i]);
        for (int c = 32; c < 127; ++c) syms.push_back(std::string(1, char(c)));
        return Dictionary::from_symbols(syms);
    }();
    return pool;
}

uint32_t Dictionary::lookup(const std::string& s) const {
    auto it = index.find(s);
    return it == index.end() ? UINT32_MAX : it->second;
}

namespace {

void put_varint(std::string& out, uint64_t v) {
    while (v >= 0x80) { out.push_back(char((v & 0x7F) | 0x80)); v >>= 7; }
    out.push_back(char(v));
}

uint64_t get_varint(std::string_view s, size_t& pos) {
    uint64_t v = 0;
    int shift = 0;
    while (pos < s.size()) {
        const uint8_t b = uint8_t(s[pos++]);
        v |= uint64_t(b & 0x7F) << shift;
        if ((b & 0x80) == 0) return v;
        shift += 7;
        if (shift > 63) throw OctError("APL codec: varint overflow");
    }
    throw OctError("APL codec: truncated varint");
}

}  // namespace

double shannon_entropy_bits(std::span<const uint32_t> codes) {
    if (codes.empty()) return 0.0;
    std::map<uint32_t, size_t> hist;
    for (uint32_t c : codes) ++hist[c];
    const double n = double(codes.size());
    double bits = 0.0;
    for (const auto& kv : hist) {
        const double p = double(kv.second) / n;
        bits -= p * std::log2(p);
    }
    return bits;
}

DenseEncoding dense_encode(const Dictionary& dict, std::span<const uint32_t> codes) {
    DenseEncoding e;
    e.symbols = codes.size();
    e.raw_bytes = codes.size() * sizeof(uint32_t);
    e.shannon_bits_per_symbol = shannon_entropy_bits(codes);
    for (uint32_t c : codes) {
        put_varint(e.payload, c);
        if (c < dict.symbols.size()) e.utf8 += dict.symbols[c];
        else                           e.utf8 += "\xef\xbf\xbd";   // U+FFFD replacement
    }
    e.payload_bytes = e.payload.size();
    e.utf8_bytes = e.utf8.size();
    // Single-byte pool form: only when every code fits in a byte.
    bool poolable = true;
    for (uint32_t c : codes) if (c > 0xFF) { poolable = false; break; }
    if (poolable) {
        e.pool.reserve(codes.size());
        for (uint32_t c : codes) e.pool.push_back(char(uint8_t(c)));
        e.pool_bytes = e.pool.size();
        e.used_pool = true;
    }
    const size_t best = e.used_pool ? std::min(e.pool_bytes, e.payload_bytes) : e.payload_bytes;
    e.bits_per_symbol = e.symbols ? 8.0 * double(best) / double(e.symbols) : 0.0;
    e.compression_ratio = best ? double(e.raw_bytes) / double(best) : 0.0;
    e.sha256 = hash::sha256_hex(std::string_view(e.used_pool ? e.pool : e.payload));
    return e;
}

Outcome<std::vector<uint32_t>> dense_decode(const DenseEncoding& enc) {
    std::vector<uint32_t> out;
    out.reserve(enc.symbols);
    const std::string& src = (enc.used_pool && !enc.pool.empty()) ? enc.pool : enc.payload;
    size_t pos = 0;
    try {
        while (pos < src.size()) {
            out.push_back(uint32_t(get_varint(src, pos)));
            if (enc.used_pool && out.size() == enc.symbols) break;
        }
    } catch (const std::exception& e) {
        return Status::invalid(e.what());
    }
    if (out.size() != enc.symbols) return Status::invalid("APL codec: symbol count mismatch");
    return out;
}

std::vector<uint32_t> encode_symbols(Dictionary& dict, const std::vector<std::string>& symbols,
                                     DenseEncoding* out_stats) {
    std::vector<uint32_t> codes;
    codes.reserve(symbols.size());
    for (const auto& s : symbols) {
        uint32_t c = dict.lookup(s);
        if (c == UINT32_MAX) {   // extend the dictionary; codes keep insertion order
            c = uint32_t(dict.symbols.size());
            dict.index[s] = c;
            dict.symbols.push_back(s);
        }
        codes.push_back(c);
    }
    if (out_stats) *out_stats = dense_encode(dict, codes);
    return codes;
}

Json DenseEncoding::to_json() const {
    Json j;
    j.begin_object();
    j.field("symbols", int64_t(symbols));
    j.field("raw_bytes_uint32", int64_t(raw_bytes));
    j.field("varint_bytes", int64_t(payload_bytes));
    j.field("pool_bytes", int64_t(pool_bytes));
    j.field("utf8_bytes", int64_t(utf8_bytes));
    j.field("used_pool", used_pool);
    j.field("bits_per_symbol", bits_per_symbol);
    j.field("raw_bits_per_symbol", raw_bits_per_symbol);
    j.field("shannon_bits_per_symbol", shannon_bits_per_symbol);
    j.field("compression_ratio_vs_uint32", compression_ratio);
    j.field("sha256", sha256);
    j.end_object();
    return j;
}

// ---------------------------------------------------------------------------
// Module wrapper
// ---------------------------------------------------------------------------
namespace {

class AplModule final : public Module {
public:
    ModuleInfo info() const override {
        ModuleInfo i;
        i.name = "apl.arrays";
        i.version = "1.0.0";
        i.language = "APL-style (subset)";
        i.role = "array IR + dense token-stream codec";
        i.trust = Trust::Core;
        i.compiled_in = true;
        i.capabilities = {"array.eval", "array.reshape", "array.reduce", "array.scan",
                          "array.inner_product", "array.outer_product", "array.solve",
                          "token.dense_encode", "token.dense_decode"};
        i.limitations = {
            "APL subset: no nested/boxed arrays, no user-defined operators, no ⎕ system functions",
            "codec is a varint/byte-pool packer, not an entropy coder; the Shannon floor is reported",
            "no SIMD path yet: elementwise ops are scalar loops (candidate for the Fortran backend)",
            "numeric domain is f64 only; no complex or rational arrays",
        };
        i.build_id = std::string("c++20/") + __VERSION__;
        return i;
    }

    std::string describe() const override {
        return "Dense array evaluator (right-to-left APL subset) plus a lossless token codec "
               "used as the engine's internal token representation.";
    }

    Status self_check() override {
        Environment env;
        auto r1 = eval_line("+/ \xe2\x8d\xb3 5", env);          // +/ ⍳5 = 10
        if (!r1 || r1->size() != 1 || r1->scalar_f64() != 10.0)
            return Status::internal("apl: +/⍳5 != 10");
        auto r2 = eval_line("x \xe2\x86\x90 2 3 \xe2\x8d\xb4 \xe2\x8d\xb3 6", env);  // 2 3⍴⍳6
        if (!r2 || r2->shape() != Index{2, 3})
            return Status::internal("apl: 2 3⍴⍳6 shape wrong");
        auto r3 = eval_line("+\xe2\x8c\xbd x", env);            // column sums via reduce
        (void)r3;
        // codec round-trip on a synthetic token stream
        Dictionary pool = Dictionary::apl_pool();
        std::vector<std::string> toks;
        for (int i = 0; i < 256; ++i) toks.push_back(std::string(1, char(32 + (i % 95))));
        DenseEncoding enc;
        const std::vector<uint32_t> codes = encode_symbols(pool, toks, &enc);
        if (codes.size() != toks.size()) return Status::internal("apl: codec code count");
        auto back = dense_decode(enc);
        if (!back || back->size() != toks.size())
            return Status::internal("apl: codec round-trip failed");
        for (size_t i = 0; i < toks.size(); ++i)
            if (pool.symbols[(*back)[i]] != toks[i])
                return Status::internal("apl: codec round-trip mismatch");
        // reshape + scan consistency
        auto r4 = eval_line("y \xe2\x86\x90 3 2 \xe2\x8d\xb4 \xe2\x8d\xb3 6", env);
        if (!r4 || r4->shape() != Index{3, 2}) return Status::internal("apl: reshape 3 2 failed");
        auto r5 = eval_line("+\\ 1 2 3 4", env);   // scan sum 1 3 6 10
        if (!r5 || r5->size() != 4 || r5->scalar_f64(3) != 10.0)
            return Status::internal("apl: scan sum wrong");
        return Status::ok();
    }
};

}  // namespace

std::shared_ptr<Module> make_apl_module() { return std::make_shared<AplModule>(); }

}  // namespace oct::apl
