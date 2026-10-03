// Octopus Hybrid AI Engine -- GGUF tokenizer (byte-level BPE + fallback).
// SPDX-License-Identifier: MIT
#include "octopus/tokenizer.hpp"

#include "octopus/module.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <fstream>
#include <sstream>

namespace oct::tokenizer {

namespace {

// GPT-2 bytes_to_unicode(): maps every byte to a printable unicode codepoint.
const std::array<int, 256>& byte_to_codepoint() {
    static const std::array<int, 256> table = [] {
        std::array<int, 256> t{};
        std::vector<int> bs, cs;
        for (int b = int('!'); b <= int('~'); ++b) bs.push_back(b);
        for (int b = 161; b <= 172; ++b) bs.push_back(b);
        for (int b = 174; b <= 255; ++b) bs.push_back(b);
        cs = bs;
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
                bs.push_back(b);
                cs.push_back(256 + n);
                ++n;
            }
        }
        for (size_t i = 0; i < bs.size(); ++i) t[size_t(bs[i])] = cs[i];
        return t;
    }();
    return table;
}

std::string utf8_of(int codepoint) {
    std::string out;
    if (codepoint < 0x80) {
        out += char(codepoint);
    } else if (codepoint < 0x800) {
        out += char(0xC0 | (codepoint >> 6));
        out += char(0x80 | (codepoint & 0x3F));
    } else {
        out += char(0xE0 | (codepoint >> 12));
        out += char(0x80 | ((codepoint >> 6) & 0x3F));
        out += char(0x80 | (codepoint & 0x3F));
    }
    return out;
}

// Split a UTF-8 string into codepoint-sized chunks (input is well-formed UTF-8
// produced by byte_to_codepoint, so no resynchronisation is needed).
std::vector<std::string> utf8_chars(std::string_view s) {
    std::vector<std::string> out;
    size_t i = 0;
    while (i < s.size()) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        size_t len = 1;
        if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;
        if (i + len > s.size()) len = 1;
        out.push_back(std::string(s.substr(i, len)));
        i += len;
    }
    return out;
}

}  // namespace

Outcome<Vocab> Vocab::from_gguf(const gguf::ModelInfo& info) {
    Vocab v;
    auto model = info.get_str("tokenizer.ggml.model");
    if (!model) return Status::unavailable("tokenizer: file has no tokenizer.ggml.model");
    v.model = *model;
    if (auto pre = info.get_str("tokenizer.ggml.pre")) v.pre = *pre;
    auto toks = info.get_str_array("tokenizer.ggml.tokens");
    if (!toks) return Status::unavailable("tokenizer: file has no tokenizer.ggml.tokens");
    v.tokens = *toks;
    if (auto m = info.get_str_array("tokenizer.ggml.merges")) v.merges = *m;
    for (size_t i = 0; i < v.tokens.size(); ++i) v.index.emplace(v.tokens[i], uint32_t(i));
    auto id_of = [&](const char* key) -> uint32_t {
        auto id = info.get_u64(key);
        return id ? uint32_t(*id) : UINT32_MAX;
    };
    v.bos_id = id_of("tokenizer.ggml.bos_token_id");
    v.eos_id = id_of("tokenizer.ggml.eos_token_id");
    v.pad_id = id_of("tokenizer.ggml.padding_token_id");
    v.exact_bpe = (v.model == "gpt2") && !v.merges.empty();
    return v;
}

Outcome<Vocab> Vocab::load(const std::string& path) {
    auto info = gguf::read_model_info(path);
    if (!info) return info.status;
    return from_gguf(*info);
}

// ---------------------------------------------------------------------------
// Exact byte-level BPE
// ---------------------------------------------------------------------------
struct BpeTable {
    std::map<std::pair<std::string, std::string>, int> rank;
};

namespace {

BpeTable build_ranks(const Vocab& v) {
    BpeTable t;
    for (size_t i = 0; i < v.merges.size(); ++i) {
        const std::string& m = v.merges[i];
        const size_t sp = m.find(' ');
        if (sp == std::string::npos) continue;
        t.rank[{m.substr(0, sp), m.substr(sp + 1)}] = int(i);
    }
    return t;
}

// --- Unicode classification (approximation of llama.cpp's tables) ----------
// ASCII is exact; beyond ASCII we classify by codepoint ranges: known numeric
// symbols count as numbers, known space codepoints as whitespace, the symbol /
// emoji planes as "other", everything else as a letter. This is an
// approximation of the full Unicode tables and is reported as such.
bool cp_is_space(uint32_t c) {
    return c == 0x20 || c == '\t' || c == '\n' || c == '\r' || c == 0x0B || c == 0x0C ||
           c == 0x85 || c == 0xA0 || (c >= 0x2000 && c <= 0x200A) || c == 0x2028 ||
           c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}
bool cp_is_number(uint32_t c) {
    if (c >= '0' && c <= '9') return true;
    if (c < 0x80) return false;
    return c == 0xB2 || c == 0xB3 || c == 0xB9 || c == 0xBC || c == 0xBD || c == 0xBE ||
           (c >= 0x660 && c <= 0x669) || (c >= 0x6F0 && c <= 0x6F9) ||
           (c >= 0x966 && c <= 0x96F) || (c >= 0xFF10 && c <= 0xFF19) ||
           (c >= 0x1D7CE && c <= 0x1D7FF);
}
enum class CpClass : int { OutOfRange = 0, Letter = 1, Number = 2, Other = 3, Space = 4 };
CpClass cp_class(uint32_t c) {
    if (cp_is_space(c)) return CpClass::Space;
    if (cp_is_number(c)) return CpClass::Number;
    if (c < 0x80) return std::isalpha(int(c)) ? CpClass::Letter : CpClass::Other;
    if (c >= 0x1F000 && c <= 0x1FAFF) return CpClass::Other;   // emoji / symbols
    if (c >= 0x2000 && c <= 0x2BFF && !(c >= 0x3040 && c <= 0x30FF)) return CpClass::Other;
    return CpClass::Letter;
}

struct Cpt {
    uint32_t cp = 0;
    size_t   byte_start = 0;
};

std::vector<Cpt> decode_utf8(std::string_view text) {
    std::vector<Cpt> out;
    size_t i = 0;
    while (i < text.size()) {
        const unsigned char c = static_cast<unsigned char>(text[i]);
        uint32_t cp = c;
        size_t len = 1;
        if ((c & 0xE0) == 0xC0 && i + 1 < text.size()) {
            cp = ((c & 0x1Fu) << 6) | (static_cast<unsigned char>(text[i + 1]) & 0x3Fu);
            len = 2;
        } else if ((c & 0xF0) == 0xE0 && i + 2 < text.size()) {
            cp = ((c & 0x0Fu) << 12) | ((static_cast<unsigned char>(text[i + 1]) & 0x3Fu) << 6) |
                 (static_cast<unsigned char>(text[i + 2]) & 0x3Fu);
            len = 3;
        } else if ((c & 0xF8) == 0xF0 && i + 3 < text.size()) {
            cp = ((c & 0x07u) << 18) | ((static_cast<unsigned char>(text[i + 1]) & 0x3Fu) << 12) |
                 ((static_cast<unsigned char>(text[i + 2]) & 0x3Fu) << 6) |
                 (static_cast<unsigned char>(text[i + 3]) & 0x3Fu);
            len = 4;
        }
        out.push_back(Cpt{cp, i});
        i += len;
    }
    return out;
}

// GPT-2 pre-tokenisation, following llama.cpp's unicode_regex_split_custom_gpt2
// (the same alternation as the original GPT-2 regex):
//   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
std::vector<std::string> gpt2_pretokenize(std::string_view text) {
    const std::vector<Cpt> cpts = decode_utf8(text);
    const size_t n = cpts.size();
    std::vector<std::string> out;
    auto byte_at = [&](size_t idx) -> size_t { return idx < n ? cpts[idx].byte_start : text.size(); };
    size_t prev = 0;
    auto add = [&](size_t end_cpt) {
        if (end_cpt > prev) {
            const size_t a = byte_at(prev), b = byte_at(end_cpt);
            out.push_back(std::string(text.substr(a, b - a)));
        }
        prev = end_cpt;
    };
    auto cls = [&](size_t idx) -> CpClass {
        return idx < n ? cp_class(cpts[idx].cp) : CpClass::OutOfRange;
    };
    for (size_t pos = 0; pos < n;) {
        const uint32_t cpt = cpts[pos].cp;
        if (cpt == '\'' && pos + 1 < n) {
            const uint32_t nx = cpts[pos + 1].cp;
            if (nx == 's' || nx == 't' || nx == 'm' || nx == 'd') {
                pos += 2;
                add(pos);
                continue;
            }
            if (pos + 2 < n) {
                const uint32_t nn = cpts[pos + 2].cp;
                if ((nx == 'r' && nn == 'e') || (nx == 'v' && nn == 'e') || (nx == 'l' && nn == 'l')) {
                    pos += 3;
                    add(pos);
                    continue;
                }
            }
        }
        const CpClass flags2 = (cpt == ' ') ? cls(pos + 1) : cls(pos);
        if (flags2 == CpClass::Letter) {
            size_t p = pos + (cpt == ' ' ? 1 : 0);
            while (cls(p) == CpClass::Letter) ++p;
            pos = p;
            add(pos);
            continue;
        }
        if (flags2 == CpClass::Number) {
            size_t p = pos + (cpt == ' ' ? 1 : 0);
            while (cls(p) == CpClass::Number) ++p;
            pos = p;
            add(pos);
            continue;
        }
        if (flags2 == CpClass::Other) {
            size_t p = pos + (cpt == ' ' ? 1 : 0);
            while (cls(p) == CpClass::Other) ++p;
            pos = p;
            add(pos);
            continue;
        }
        size_t nw = 0;
        while (cls(pos + nw) == CpClass::Space) ++nw;
        if (nw > 1 && pos + nw < n) {
            pos += nw - 1;
            add(pos);
            continue;
        }
        if (nw > 0) {
            pos += nw;
            add(pos);
            continue;
        }
        add(++pos);
    }
    add(n);
    return out;
}

// Applies byte-level BPE to one pre-tokenised chunk.
Outcome<std::vector<uint32_t>> bpe_chunk(const Vocab& v, const BpeTable& ranks, std::string_view chunk) {
    const auto& table = byte_to_codepoint();
    std::string mapped;
    mapped.reserve(chunk.size() * 2);
    for (unsigned char c : chunk) mapped += utf8_of(table[c]);
    std::vector<std::string> symbols = utf8_chars(mapped);
    std::vector<uint32_t> ids;
    if (symbols.empty()) return ids;
    for (;;) {
        int best = INT32_MAX;
        size_t best_index = SIZE_MAX;
        std::pair<std::string, std::string> best_pair;
        for (size_t i = 0; i + 1 < symbols.size(); ++i) {
            const std::pair<std::string, std::string> key{symbols[i], symbols[i + 1]};
            auto it = ranks.rank.find(key);
            if (it != ranks.rank.end() && it->second < best) {
                best = it->second;
                best_index = i;
                best_pair = key;
            }
        }
        if (best_index == SIZE_MAX) break;
        std::vector<std::string> merged;
        merged.reserve(symbols.size());
        for (size_t i = 0; i < symbols.size();) {
            if (i + 1 < symbols.size() && symbols[i] == best_pair.first &&
                symbols[i + 1] == best_pair.second) {
                merged.push_back(symbols[i] + symbols[i + 1]);
                i += 2;
            } else {
                merged.push_back(symbols[i]);
                ++i;
            }
        }
        symbols = std::move(merged);
    }
    for (const auto& sym : symbols) {
        auto it = v.index.find(sym);
        if (it == v.index.end())
            return Status::internal("tokenizer: BPE produced a piece not in the vocabulary: '" + sym + "'");
        ids.push_back(it->second);
    }
    return ids;
}

Outcome<Encoding> bpe_encode(const Vocab& v, std::string_view text) {
    Encoding enc;
    // 'gpt2' is the pre-tokenizer pattern this implementation actually ports;
    // every other pattern name is reported as degraded, never silently assumed.
    const bool gpt2_pre = v.pre.empty() || v.pre == "gpt2" || v.pre == "gpt-2";
    enc.method = gpt2_pre ? "bpe.byte_level.gpt2.exact"
                          : "bpe.byte_level.unsupported_pre.degraded";
    if (!gpt2_pre) {
        enc.degraded = true;
        enc.warnings = "tokenizer.ggml.pre='" + v.pre +
                       "' is not implemented; GPT-2 pre-tokenisation is used instead and results are measured";
    }
    // Merge ranks are expensive to rebuild (tens of thousands of entries), so
    // they are built once per vocabulary and shared by its copies. The ranks
    // live in the Vocab itself: a static pointer-keyed cache would hand a
    // different vocabulary's ranks to a re-used address (found by test).
    if (!v.ranks) v.ranks = std::make_shared<const BpeTable>(build_ranks(v));
    const BpeTable& ranks = *v.ranks;

    for (const std::string& chunk : gpt2_pretokenize(text)) {
        auto ids = bpe_chunk(v, ranks, chunk);
        if (!ids) return ids.status;
        enc.ids.insert(enc.ids.end(), ids->begin(), ids->end());
    }
    return enc;
}


Outcome<Encoding> greedy_encode(const Vocab& v, std::string_view text) {
    Encoding enc;
    enc.method = "greedy.longest_match.degraded";
    enc.degraded = true;
    enc.warnings = "no merges in this vocabulary: greedy longest-match fallback, not the reference tokenizer";
    size_t i = 0;
    while (i < text.size()) {
        size_t best_len = 0;
        uint32_t best_id = UINT32_MAX;
        for (size_t len = std::min<size_t>(32, text.size() - i); len >= 1; --len) {
            const std::string piece(text.substr(i, len));
            auto it = v.index.find(piece);
            if (it != v.index.end()) {
                best_len = len;
                best_id = it->second;
                break;
            }
        }
        if (best_len == 0) {
            const auto& table = byte_to_codepoint();
            const std::string piece = utf8_of(table[static_cast<unsigned char>(text[i])]);
            auto it = v.index.find(piece);
            if (it == v.index.end())
                return Status::internal("tokenizer: cannot encode byte at offset " + std::to_string(i));
            enc.ids.push_back(it->second);
            ++i;
        } else {
            enc.ids.push_back(best_id);
            i += best_len;
        }
    }
    return enc;
}

}  // namespace

Outcome<Encoding> encode(const Vocab& v, std::string_view text) {
    if (v.exact_bpe) return bpe_encode(v, text);
    return greedy_encode(v, text);
}

Outcome<std::string> decode(const Vocab& v, std::span<const uint32_t> ids) {
    std::string out;
    for (uint32_t id : ids) {
        if (id >= v.tokens.size()) return Status::invalid("tokenizer: token id out of range");
        const std::string& piece = v.tokens[id];
        if (!v.exact_bpe) {   // sentencepiece-ish: '\u2581' marks a leading space
            for (size_t i = 0; i < piece.size();) {
                if (piece.compare(i, 3, "\xe2\x96\x81") == 0) {
                    out += ' ';
                    i += 3;
                } else {
                    out += piece[i++];
                }
            }
            continue;
        }
        for (size_t i = 0; i < piece.size();) {
            const unsigned char c = static_cast<unsigned char>(piece[i]);
            size_t len = 1;
            if ((c & 0xE0) == 0xC0) len = 2;
            else if ((c & 0xF0) == 0xE0) len = 3;
            int codepoint = c;
            if (len == 2 && i + 1 < piece.size())
                codepoint = ((c & 0x1F) << 6) | (static_cast<unsigned char>(piece[i + 1]) & 0x3F);
            else if (len == 3 && i + 2 < piece.size())
                codepoint = ((c & 0x0F) << 12) |
                            ((static_cast<unsigned char>(piece[i + 1]) & 0x3F) << 6) |
                            (static_cast<unsigned char>(piece[i + 2]) & 0x3F);
            int byte = -1;
            const auto& table = byte_to_codepoint();
            for (int b = 0; b < 256; ++b)
                if (table[size_t(b)] == codepoint) { byte = b; break; }
            if (byte < 0) return Status::internal("tokenizer: piece is not byte-level encoded");
            out += char(byte);
            i += len;
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Fixture comparison
// ---------------------------------------------------------------------------
Json ReferenceCheck::to_json() const {
    Json j;
    j.begin_object();
    j.field("cases", int64_t(cases));
    j.field("exact_cases", int64_t(exact_cases));
    j.field("case_match_rate", case_match_rate());
    j.field("tokens_matched", int64_t(token_matches));
    j.field("tokens_total", int64_t(token_total));
    j.field("token_match_rate", token_match_rate());
    j.end_object();
    return j;
}

Outcome<ReferenceCheck> check_against_fixture(const Vocab& v, const std::string& inp_path,
                                              const std::string& out_path) {
    std::ifstream in(inp_path, std::ios::binary);
    std::ifstream out(out_path, std::ios::binary);
    if (!in) return Status::unavailable("tokenizer: cannot open " + inp_path);
    if (!out) return Status::unavailable("tokenizer: cannot open " + out_path);
    std::string raw((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    std::vector<std::string> expected;
    for (std::string line; std::getline(out, line);) expected.push_back(line);
    const std::string sep = "\n__ggml_vocab_test__\n";
    std::vector<std::string> segments;
    size_t pos = 0;
    while (pos < raw.size()) {
        const size_t next = raw.find(sep, pos);
        if (next == std::string::npos) {
            segments.push_back(raw.substr(pos));
            break;
        }
        segments.push_back(raw.substr(pos, next - pos));
        pos = next + sep.size();
    }
    if (segments.size() != expected.size())
        return Status::invalid("tokenizer: fixture size mismatch (" + std::to_string(segments.size()) +
                               " vs " + std::to_string(expected.size()) + ")");
    ReferenceCheck rc;
    rc.cases = segments.size();
    for (size_t i = 0; i < segments.size(); ++i) {
        std::vector<uint32_t> want;
        std::istringstream os(expected[i]);
        int64_t id = 0;
        while (os >> id) want.push_back(uint32_t(id));
        auto got = encode(v, segments[i]);
        if (!got) {
            rc.token_total += want.size();
            continue;
        }
        size_t matches = 0;
        const size_t n = std::min(want.size(), got->ids.size());
        for (size_t k = 0; k < n; ++k) {
            ++rc.token_total;
            if (want[k] == got->ids[k]) ++matches;
        }
        rc.token_total += want.size() - n;
        rc.token_matches += matches;
        if (want == got->ids) ++rc.exact_cases;
    }
    return rc;
}

// ---------------------------------------------------------------------------
// Module wrapper
// ---------------------------------------------------------------------------
namespace {

class TokenizerModule final : public Module {
public:
    ModuleInfo info() const override {
        ModuleInfo i;
        i.name = "tokenizer.gguf";
        i.version = "1.0.0";
        i.language = "C++20";
        i.role = "GGUF vocabulary loader, byte-level BPE encoder, exact fixture comparison";
        i.trust = Trust::Core;
        i.compiled_in = true;
        i.capabilities = {"tokenizer.load", "tokenizer.encode", "tokenizer.decode",
                          "tokenizer.compare_fixture"};
        i.limitations = {
            "exact for GPT-2 byte-level BPE with merges; other tokenizer.ggml.pre patterns use GPT-2 pre-tokenisation and their match rate is measured, not assumed",
            "sentencepiece vocabularies (no merges) use a greedy longest-match fallback",
            "no added/special-token handling beyond bos/eos/pad ids; no chat templates",
        };
        i.build_id = std::string("c++20/") + __VERSION__;
        return i;
    }
    std::string describe() const override {
        return "Reads vocabularies from the GGUF file itself and encodes with the exact "
               "byte-level BPE algorithm when merge ranks are present.";
    }
    Status self_check() override {
        Vocab v;
        v.model = "gpt2";
        v.tokens = {"a", "b", "ab", "c"};
        v.merges = {"a b"};
        for (size_t i = 0; i < v.tokens.size(); ++i) v.index.emplace(v.tokens[i], uint32_t(i));
        v.exact_bpe = true;
        auto enc = encode(v, "ab");
        if (!enc || enc->ids.size() != 1 || enc->ids[0] != 2)
            return Status::internal("tokenizer: merge rank was not applied");
        auto dec = decode(v, enc->ids);
        if (!dec || *dec != "ab") return Status::internal("tokenizer: decode round-trip failed");
        Vocab g;
        g.model = "llama";
        g.tokens = {"\xe2\x96\x81hello", "hello", "!"};
        for (size_t i = 0; i < g.tokens.size(); ++i) g.index.emplace(g.tokens[i], uint32_t(i));
        auto ge = encode(g, "hello!");
        if (!ge || ge->ids.size() != 2) return Status::internal("tokenizer: greedy encode failed");
        auto gd = decode(g, ge->ids);
        if (!gd || *gd != "hello!") return Status::internal("tokenizer: greedy decode failed");
        return Status::ok();
    }
};

}  // namespace

std::shared_ptr<oct::Module> make_tokenizer_module() {
    return std::make_shared<TokenizerModule>();
}

}  // namespace oct::tokenizer
