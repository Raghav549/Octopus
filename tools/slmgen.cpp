// Octopus Hybrid AI Engine -- tiny SLM generator CLI.
//
// Writes a complete, loadable llama-architecture GGUF model with synthetic
// weights so the real inference path can be exercised offline (see
// include/octopus/synthetic_model.hpp for the honesty contract: the weights are
// pseudo-random, the output is nonsense, and the file says so in its metadata).
//
// Usage: slmgen <out.gguf> [--layers N] [--embd N] [--heads N] [--kv-heads N]
//                             [--ff N] [--ctx N] [--seed N]
// SPDX-License-Identifier: MIT
#include "octopus/synthetic_model.hpp"

#include <cstdio>
#include <string>

using namespace oct;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: slmgen <out.gguf> [--layers N] [--embd N] [--heads N] "
                    "[--kv-heads N] [--ff N] [--ctx N] [--seed N]\n");
        return 2;
    }
    synthetic::ModelSpec spec;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (i + 1 >= argc) {
            std::fprintf(stderr, "slmgen: %s needs a value\n", a.c_str());
            return 2;
        }
        const int64_t v = std::stoll(argv[++i]);
        if (a == "--layers") spec.n_layer = v;
        else if (a == "--embd") spec.n_embd = v;
        else if (a == "--heads") spec.n_head = v;
        else if (a == "--kv-heads") spec.n_head_kv = v;
        else if (a == "--ff") spec.n_ff = v;
        else if (a == "--ctx") spec.n_ctx = v;
        else if (a == "--seed") spec.seed = uint64_t(v);
        else {
            std::fprintf(stderr, "slmgen: unknown option %s\n", a.c_str());
            return 2;
        }
    }
    auto written = synthetic::write_model(argv[1], spec);
    if (!written) {
        std::fprintf(stderr, "slmgen: %s\n", written.status.message.c_str());
        return 1;
    }
    std::printf("%s\n", written->to_json().str().c_str());
    return 0;
}
