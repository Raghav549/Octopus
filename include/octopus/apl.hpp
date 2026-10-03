// Octopus Hybrid AI Engine -- APL-style array language layer.
//
// Role in the engine (docs/DESIGN.md): a dense, expression-oriented IR for
// array transformations, plus a token-packing codec used as the internal
// representation of token-id streams.
//
// Honest scope:
//   * The evaluator implements a real (if compact) APL subset with genuine
//     right-to-left evaluation, scalar extension, reductions, scans, inner and
//     outer products.
//   * The codec compresses symbol streams by varint/pool packing. It reports the
//     *measured* size against uint32 storage, the single-byte pool size and the
//     Shannon entropy of the stream. It is lossless. It is not entropy-optimal:
//     that would need an arithmetic/range coder, which is not implemented here
//     and is stated as such in the module's limitation list.
// SPDX-License-Identifier: MIT
#pragma once

#include "octopus/core.hpp"
#include "octopus/tensor.hpp"

#include <map>
#include <span>

namespace oct { class Module; }

namespace oct::apl {

// ---------------------------------------------------------------------------
// Lexer / parser
// ---------------------------------------------------------------------------
struct Token {
    enum class Kind : uint8_t { Number, String, Name, Glyph, Assign, LParen, RParen, Dot, Semi };
    Kind        kind = Kind::Number;
    std::string text;
    double      number = 0.0;
};

struct AstNode {
    enum class Kind : uint8_t { Num, Str, Name, Strand, Monadic, Dyadic, Derived };
    Kind                            kind = Kind::Num;
    double                          number = 0.0;
    std::string                     text;      // glyph, name or string
    std::string                     deriv;     // '/', '\\', or the second function of f.g
    std::vector<std::shared_ptr<AstNode>> kids;
};

struct Program {
    std::string             source;
    std::vector<Token>      tokens;
    std::string             target;       // assignment target, empty for expressions
    std::shared_ptr<AstNode> root;
    Json   to_json() const;
    std::string to_apl() const;           // canonical normalised form
};

Outcome<Program> parse(std::string_view source);

class Environment {
public:
    std::map<std::string, Array> vars;
    void assign(const std::string& name, Array value) { vars[name] = std::move(value); }
    bool has(const std::string& name) const { return vars.count(name) != 0; }
};

struct EvalTrace {
    std::vector<std::string> steps;   // human-readable, one per primitive applied
    int64_t primitive_count = 0;
};

Outcome<Array> eval(const Program& p, Environment& env, EvalTrace* trace = nullptr);
// Convenience: parse + evaluate a single line.
Outcome<Array> eval_line(std::string_view src, Environment& env, EvalTrace* trace = nullptr);

// ---------------------------------------------------------------------------
// Dense token-stream codec
// ---------------------------------------------------------------------------
struct Dictionary {
    std::vector<std::string>        symbols;
    std::map<std::string, uint32_t> index;
    static Dictionary from_symbols(const std::vector<std::string>& syms);
    // A 256-entry pool (APL primitives + byte alphabet) that lets a stream be
    // written one byte per symbol where possible.
    static const Dictionary& apl_pool();
    uint32_t lookup(const std::string& s) const;      // UINT32_MAX when absent
};

struct DenseEncoding {
    size_t symbols = 0;
    size_t raw_bytes = 0;          // 4 bytes/symbol baseline (uint32 ids)
    size_t payload_bytes = 0;      // varint stream
    size_t pool_bytes = 0;         // one byte per symbol when codes fit in 8 bits
    size_t utf8_bytes = 0;         // UTF-8 text form
    double bits_per_symbol = 0.0;
    double raw_bits_per_symbol = 32.0;
    double shannon_bits_per_symbol = 0.0;
    double compression_ratio = 1.0;
    bool   used_pool = false;
    std::string payload;           // varint-packed codes
    std::string pool;              // single-byte codes (empty when not applicable)
    std::string utf8;
    std::string sha256;
    Json to_json() const;
};

DenseEncoding   dense_encode(const Dictionary& dict, std::span<const uint32_t> codes);
Outcome<std::vector<uint32_t>> dense_decode(const DenseEncoding& enc);
// Entropy of the code histogram (bits/symbol) -- the information-theoretic floor.
double shannon_entropy_bits(std::span<const uint32_t> codes);
// Maps symbols to codes (extending the dictionary when a symbol is new) and
// returns the code sequence. Round-trips through dense_decode.
std::vector<uint32_t> encode_symbols(Dictionary& dict,
                                     const std::vector<std::string>& symbols,
                                     DenseEncoding* out_stats = nullptr);

// ---------------------------------------------------------------------------
// Module wrapper (registered with the engine)
// ---------------------------------------------------------------------------
std::shared_ptr<oct::Module> make_apl_module();

}  // namespace oct::apl
