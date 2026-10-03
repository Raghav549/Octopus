// Octopus Hybrid AI Engine -- GGUF fixture generator.
//
// Writes a tiny but *valid* GGUF v3 container with the metadata of a 2-layer
// llama-style model plus small real tensors. The fixture is used by the test
// suite and by CI so that GGUF parsing, model facts and the tokenizer path are
// exercised without shipping a multi-gigabyte model.
// SPDX-License-Identifier: MIT
#include "octopus/gguf.hpp"

#include <cstdio>
#include <cstring>
#include <string>

using namespace oct;

namespace {

// Deterministic little-endian float payload (no external data).
std::vector<uint8_t> f32_payload(size_t count, uint64_t seed) {
    std::vector<uint8_t> out;
    out.reserve(count * 4);
    Rng rng(seed);
    for (size_t i = 0; i < count; ++i) {
        const float value = float((rng.next_unit() - 0.5) * 0.02);
        uint32_t bits = 0;
        std::memcpy(&bits, &value, 4);
        out.push_back(uint8_t(bits & 0xFF));
        out.push_back(uint8_t((bits >> 8) & 0xFF));
        out.push_back(uint8_t((bits >> 16) & 0xFF));
        out.push_back(uint8_t((bits >> 24) & 0xFF));
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    std::string out_path = argc > 1 ? argv[1] : "octopus_fixture.gguf";
    const int64_t n_vocab = argc > 2 ? std::stoll(argv[2]) : 32;
    const int64_t n_embd = 16;

    gguf::Writer w;
    w.set_str("general.architecture", "llama");
    w.set_str("general.name", "octopus-ci-fixture");
    w.set_u32("general.file_type", 0);           // F32
    w.set_u32("llama.block_count", 2);
    w.set_u32("llama.context_length", 64);
    w.set_u32("llama.embedding_length", uint32_t(n_embd));
    w.set_u32("llama.feed_forward_length", 32);
    w.set_u32("llama.attention.head_count", 2);
    w.set_u32("llama.attention.head_count_kv", 2);
    w.set_u32("llama.vocab_size", uint32_t(n_vocab));
    w.set_f32("llama.rope.freq_base", 10000.0f);
    w.set_f32("llama.attention.layer_norm_rms_epsilon", 1e-5f);
    w.set_str("tokenizer.ggml.model", "llama");
    std::vector<std::string> tokens;
    for (int64_t i = 0; i < n_vocab; ++i) tokens.push_back("<tok" + std::to_string(i) + ">");
    tokens[0] = "<unk>";
    tokens[1] = "<s>";
    tokens[2] = "</s>";
    w.set_str_array("tokenizer.ggml.tokens", tokens);
    w.set_str_array("tokenizer.ggml.merges", {});

    w.add_tensor("token_embd.weight", {uint64_t(n_embd), uint64_t(n_vocab)}, gguf::GgmlType::F32,
                 f32_payload(size_t(n_embd) * size_t(n_vocab), 1));
    w.add_tensor("output_norm.weight", {uint64_t(n_embd)}, gguf::GgmlType::F32,
                 f32_payload(size_t(n_embd), 2));
    w.add_tensor("blk.0.attn_q.weight", {uint64_t(n_embd), uint64_t(n_embd)}, gguf::GgmlType::F32,
                 f32_payload(size_t(n_embd) * size_t(n_embd), 3));
    w.add_tensor("blk.0.attn_k.weight", {uint64_t(n_embd), uint64_t(n_embd)}, gguf::GgmlType::F32,
                 f32_payload(size_t(n_embd) * size_t(n_embd), 4));
    w.add_tensor("blk.0.attn_v.weight", {uint64_t(n_embd), uint64_t(n_embd)}, gguf::GgmlType::F32,
                 f32_payload(size_t(n_embd) * size_t(n_embd), 5));
    w.add_tensor("blk.0.ffn_gate.weight", {32, uint64_t(n_embd)}, gguf::GgmlType::F32,
                 f32_payload(size_t(n_embd) * 32, 6));

    Status st = w.write_file(out_path);
    if (!st) {
        std::fprintf(stderr, "ggufgen: %s\n", st.message.c_str());
        return 1;
    }
    const auto buf = w.encode();
    std::printf("{\"file\":\"%s\",\"bytes\":%zu,\"tensors\":6,\"vocab\":%lld,\"sha256\":\"%s\"}\n",
                out_path.c_str(), buf.size(), (long long)n_vocab,
                hash::sha256_hex(std::string_view(reinterpret_cast<const char*>(buf.data()),
                                                  buf.size()))
                    .c_str());
    return 0;
}
