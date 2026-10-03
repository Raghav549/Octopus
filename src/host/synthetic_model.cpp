// Octopus Hybrid AI Engine -- synthetic tiny-model generator (implementation).
// SPDX-License-Identifier: MIT
#include "octopus/synthetic_model.hpp"

#include "octopus/gguf.hpp"

#include <cstring>
#include <map>
#include <vector>

namespace oct::synthetic {

namespace {

// The GPT-2 byte <-> unicode table: byte-level BPE vocabularies store text as
// these printable stand-ins, so a vocabulary built here is a *valid* BPE
// vocabulary that llama.cpp can tokenize with.
std::map<uint32_t, uint32_t> byte_to_codepoint() {
    std::map<uint32_t, uint32_t> table;
    std::vector<int> direct;
    for (int b = '!'; b <= '~'; ++b) direct.push_back(b);
    for (int b = 0xA1; b <= 0xAC; ++b) direct.push_back(b);
    for (int b = 0xAE; b <= 0xFF; ++b) direct.push_back(b);
    std::vector<bool> used(256, false);
    for (int b : direct) used[size_t(b)] = true;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (!used[size_t(b)]) {
            table[uint32_t(b)] = uint32_t(256 + n);
            ++n;
        } else {
            table[uint32_t(b)] = uint32_t(b);
        }
    }
    return table;
}

std::string utf8_of(uint32_t cp) {
    std::string s;
    if (cp < 0x80) {
        s.push_back(char(cp));
    } else if (cp < 0x800) {
        s.push_back(char(0xC0 | (cp >> 6)));
        s.push_back(char(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        s.push_back(char(0xE0 | (cp >> 12)));
        s.push_back(char(0x80 | ((cp >> 6) & 0x3F)));
        s.push_back(char(0x80 | (cp & 0x3F)));
    } else {
        s.push_back(char(0xF0 | (cp >> 18)));
        s.push_back(char(0x80 | ((cp >> 12) & 0x3F)));
        s.push_back(char(0x80 | ((cp >> 6) & 0x3F)));
        s.push_back(char(0x80 | (cp & 0x3F)));
    }
    return s;
}

std::vector<uint8_t> f32_payload(size_t count, double scale, uint64_t seed) {
    std::vector<uint8_t> out;
    out.reserve(count * 4);
    Rng rng(seed);
    for (size_t i = 0; i < count; ++i) {
        const float value = float((rng.next_unit() * 2.0 - 1.0) * scale);
        uint32_t bits = 0;
        std::memcpy(&bits, &value, 4);
        for (int k = 0; k < 4; ++k) out.push_back(uint8_t((bits >> (8 * k)) & 0xFF));
    }
    return out;
}

std::vector<uint8_t> f32_ones(size_t count, float value) {
    std::vector<uint8_t> out;
    out.reserve(count * 4);
    for (size_t i = 0; i < count; ++i) {
        uint32_t bits = 0;
        std::memcpy(&bits, &value, 4);
        for (int k = 0; k < 4; ++k) out.push_back(uint8_t((bits >> (8 * k)) & 0xFF));
    }
    return out;
}

}  // namespace

Json WrittenModel::to_json() const {
    Json j;
    j.begin_object();
    j.field("file", path);
    j.field("bytes", int64_t(bytes));
    j.field("layers", layers);
    j.field("embd", embd);
    j.field("heads", heads);
    j.field("ff", ff);
    j.field("vocab", int64_t(vocab));
    j.field("weights", "synthetic-pseudo-random");
    j.field("sha256", sha256);
    j.field("warning", "not a trained model: for exercising the inference path only");
    j.end_object();
    return j;
}

Outcome<WrittenModel> write_model(const std::string& path, const ModelSpec& spec) {
    if (spec.n_embd % spec.n_head != 0)
        return Status::invalid("synthetic: embedding length must be divisible by head count");
    if (spec.n_head % spec.n_head_kv != 0)
        return Status::invalid("synthetic: head count must be a multiple of the KV head count");
    if (spec.n_layer < 1 || spec.n_embd < 1 || spec.n_ff < 1)
        return Status::invalid("synthetic: layer/embedding/feed-forward sizes must be positive");

    gguf::Writer w;
    w.set_str("general.architecture", "llama");
    w.set_str("general.name", "octopus-tiny-synthetic");
    w.set_u32("general.file_type", 0);                       // ALL_F32
    w.set_str("general.description",
              "Synthetic weights: for exercising the inference path only, not a usable model");
    w.set_str("octopus.weights", "synthetic-pseudo-random");
    w.set_str("octopus.purpose", "offline end-to-end inference-path test");
    w.set_u32("llama.block_count", uint32_t(spec.n_layer));
    w.set_u32("llama.context_length", uint32_t(spec.n_ctx));
    w.set_u32("llama.embedding_length", uint32_t(spec.n_embd));
    w.set_u32("llama.feed_forward_length", uint32_t(spec.n_ff));
    w.set_u32("llama.attention.head_count", uint32_t(spec.n_head));
    w.set_u32("llama.attention.head_count_kv", uint32_t(spec.n_head_kv));
    w.set_f32("llama.attention.layer_norm_rms_epsilon", 1e-5f);
    w.set_f32("llama.rope.freq_base", 10000.0f);

    // Vocabulary: the 256 GPT-2 byte-level tokens, then <unk>, <s>, </s>. This is
    // a valid BPE vocabulary with no merges, so it is exactly the byte alphabet.
    const auto table = byte_to_codepoint();
    std::vector<std::string> tokens;
    std::vector<int32_t> token_types;
    for (uint32_t b = 0; b < 256; ++b) {
        tokens.push_back(utf8_of(table.at(b)));
        token_types.push_back(1);            // 1 = NORMAL
    }
    const uint32_t unk_id = uint32_t(tokens.size());
    tokens.push_back("<unk>");
    token_types.push_back(2);                // 2 = UNKNOWN
    const uint32_t bos_id = uint32_t(tokens.size());
    tokens.push_back("<s>");
    token_types.push_back(3);                // 3 = CONTROL
    const uint32_t eos_id = uint32_t(tokens.size());
    tokens.push_back("</s>");
    token_types.push_back(3);
    const uint32_t n_vocab = uint32_t(tokens.size());

    w.set_u32("llama.vocab_size", n_vocab);
    w.set_str("tokenizer.ggml.model", "gpt2");
    // "llama-bpe" is a pre-tokenizer llama.cpp implements for llama-architecture
    // BPE vocabularies; a name it does not know makes the load fail.
    w.set_str("tokenizer.ggml.pre", "llama-bpe");
    w.set_str_array("tokenizer.ggml.tokens", tokens);
    w.set_str_array("tokenizer.ggml.merges", {});
    w.set_i32_array("tokenizer.ggml.token_type", token_types);
    w.set_u32("tokenizer.ggml.unknown_token_id", unk_id);
    w.set_u32("tokenizer.ggml.bos_token_id", bos_id);
    w.set_u32("tokenizer.ggml.eos_token_id", eos_id);
    w.set_bool("tokenizer.ggml.add_bos_token", true);
    w.set_bool("tokenizer.ggml.add_eos_token", false);

    // Tensors: a complete llama block set (the loader requires all of them).
    const uint64_t ne = uint64_t(spec.n_embd), nf = uint64_t(spec.n_ff), nv = uint64_t(n_vocab);
    const uint64_t n_kv_dim = uint64_t(spec.n_embd / spec.n_head * spec.n_head_kv);
    const double scale = spec.weight_scale;
    uint64_t seed = spec.seed;
    w.add_tensor("token_embd.weight", {ne, nv}, gguf::GgmlType::F32,
                 f32_payload(size_t(ne * nv), scale, seed++));
    w.add_tensor("output_norm.weight", {ne}, gguf::GgmlType::F32, f32_ones(size_t(ne), 1.0f));
    w.add_tensor("output.weight", {ne, nv}, gguf::GgmlType::F32,
                 f32_payload(size_t(ne * nv), scale, seed++));
    for (int64_t l = 0; l < spec.n_layer; ++l) {
        const std::string p = "blk." + std::to_string(l) + ".";
        w.add_tensor(p + "attn_norm.weight", {ne}, gguf::GgmlType::F32, f32_ones(size_t(ne), 1.0f));
        w.add_tensor(p + "attn_q.weight", {ne, ne}, gguf::GgmlType::F32,
                     f32_payload(size_t(ne * ne), scale, seed++));
        w.add_tensor(p + "attn_k.weight", {ne, n_kv_dim}, gguf::GgmlType::F32,
                     f32_payload(size_t(ne * n_kv_dim), scale, seed++));
        w.add_tensor(p + "attn_v.weight", {ne, n_kv_dim}, gguf::GgmlType::F32,
                     f32_payload(size_t(ne * n_kv_dim), scale, seed++));
        w.add_tensor(p + "attn_output.weight", {ne, ne}, gguf::GgmlType::F32,
                     f32_payload(size_t(ne * ne), scale, seed++));
        w.add_tensor(p + "ffn_norm.weight", {ne}, gguf::GgmlType::F32, f32_ones(size_t(ne), 1.0f));
        w.add_tensor(p + "ffn_gate.weight", {ne, nf}, gguf::GgmlType::F32,
                     f32_payload(size_t(ne * nf), scale, seed++));
        w.add_tensor(p + "ffn_down.weight", {nf, ne}, gguf::GgmlType::F32,
                     f32_payload(size_t(nf * ne), scale, seed++));
        w.add_tensor(p + "ffn_up.weight", {ne, nf}, gguf::GgmlType::F32,
                     f32_payload(size_t(ne * nf), scale, seed++));
    }

    const Status st = w.write_file(path);
    if (!st) return st;
    const auto buf = w.encode();

    WrittenModel m;
    m.path = path;
    m.layers = spec.n_layer;
    m.embd = spec.n_embd;
    m.heads = spec.n_head;
    m.ff = spec.n_ff;
    m.vocab = n_vocab;
    m.bytes = buf.size();
    m.sha256 = hash::sha256_hex(
        std::string_view(reinterpret_cast<const char*>(buf.data()), buf.size()));
    return m;
}

}  // namespace oct::synthetic
