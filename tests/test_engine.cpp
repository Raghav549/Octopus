// Engine-level tests: the capability registry, the module catalogue, hash
// reference vectors, reproducibility of results and the Result JSON contract.
// SPDX-License-Identifier: MIT
#include "harness.hpp"

#include "octopus/catalog.hpp"
#include "octopus/numerics.hpp"

#include <string>
#include <vector>

using namespace oct;

OCT_TEST(engine, sha256_and_fnv_match_published_reference_vectors) {
    // FIPS 180-4 test vectors, checked against the standard, not against
    // ourselves: a self-consistent hash would pass a round-trip test too.
    OCT_CHECK(hash::sha256_hex("abc") ==
              "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    OCT_CHECK(hash::sha256_hex("") ==
              "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    OCT_CHECK(hash::sha256_hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") ==
              "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    OCT_EQ(hash::fnv1a64(""), 0xcbf29ce484222325ull);   // publishable reference value
    const uint64_t a = hash::fnv1a64("octopus");
    const uint64_t b = hash::fnv1a64("octopus");
    OCT_EQ(a, b);                                  // stable
    OCT_CHECK(hash::fnv1a64("octopus") != hash::fnv1a64("octopuS"));
}

OCT_TEST(engine, registry_exposes_every_builtin_module) {
    Registry reg;
    const size_t n = register_builtin_modules(reg);
    OCT_EQ(n, size_t(14));                 // Fortran 2023 + language DNA + neural router + universe
    OCT_EQ(reg.size(), size_t(14));

    OCT_CHECK(reg.resolve("numeric.kernel.heat2d") != nullptr);
    OCT_CHECK(reg.resolve("numeric.kernel.sod1d") != nullptr);
    OCT_CHECK(reg.resolve("numeric.kernel.tensor_field") != nullptr);
    OCT_CHECK(reg.resolve("universe.frame") != nullptr);
    // Capabilities are the dotted operation names modules advertise.
    OCT_CHECK(reg.resolve("array.eval") != nullptr);
    OCT_CHECK(reg.resolve("logic.verify") != nullptr);
    OCT_CHECK(reg.resolve("tokenizer.encode") != nullptr);
    // A module name is not a capability: the lookup must not invent one.
    OCT_CHECK(reg.resolve("apl.arrays") == nullptr);
    OCT_CHECK(reg.resolve("no.such.capability") == nullptr);

    // The router is a module like any other and advertises its capabilities.
    OCT_CHECK(reg.resolve("route.classify") != nullptr);
    OCT_CHECK(reg.resolve("route.dispatch") != nullptr);

    // dispatch() falls back to progressively shorter prefixes.
    Capability matched;
    auto m = reg.dispatch("numeric.kernel.heat2d.units", &matched);
    OCT_CHECK(m != nullptr);
    if (m) OCT_CHECK(matched == "numeric.kernel.heat2d");

    const auto inv = reg.inventory();
    OCT_EQ(inv.size(), size_t(14));
    for (const auto& i : inv) {
        OCT_CHECK(!i.name.empty());
        OCT_CHECK(!i.version.empty());
        OCT_CHECK(!i.language.empty());
        OCT_CHECK(!i.role.empty());
        OCT_CHECK(!i.capabilities.empty());
        OCT_CHECK(!i.limitations.empty());     // every module must document limits
    }
}

OCT_TEST(engine, every_builtin_module_self_check_executes_and_passes) {
    Registry reg;
    register_builtin_modules(reg);
    const auto results = reg.self_check_all();
    OCT_EQ(results.size(), size_t(14));
    for (const auto& r : results) {
        OCT_NOTE(r.module << " -> " << (r.status.is_ok() ? "ok" : r.status.message));
        OCT_CHECK_MSG(r.status.is_ok() || r.status.is_hardware_skip(),
                      r.module << ": " << r.status.message);
    }
}

OCT_TEST(engine, results_are_reproducible_and_fingerprints_bind_the_spec) {
    auto k = numerics::KernelLibrary::instance().find("sod1d");
    OCT_CHECK(k != nullptr);
    if (!k) OCT_SKIP("kernel missing");

    numerics::ProblemSpec s;
    s.kernel = "sod1d";
    s.params["n"] = 200;
    s.t_end = 0.1;

    const numerics::Result a = k->run(s);
    const numerics::Result b = k->run(s);
    OCT_CHECK(!a.fingerprint.empty());
    OCT_CHECK(a.fingerprint == b.fingerprint);        // bit-identical rerun
    OCT_EQ(a.data.size(), b.data.size());
    for (size_t i = 0; i < a.data.size() && i < 16; ++i) OCT_CHECK(a.data[i] == b.data[i]);

    numerics::ProblemSpec other = s;
    other.params["n"] = 400;
    const numerics::Result c = k->run(other);
    OCT_CHECK(c.fingerprint != a.fingerprint);        // the spec is part of the identity
}

OCT_TEST(engine, result_json_contract_is_honest) {
    auto k = numerics::KernelLibrary::instance().find("poisson2d");
    OCT_CHECK(k != nullptr);
    if (!k) OCT_SKIP("kernel missing");
    numerics::ProblemSpec s;
    s.kernel = "poisson2d";
    s.params["n"] = 16;
    // Request the Fortran backend explicitly: without a Fortran compiler the
    // result must still run, and must say that it ran degraded.
    s.backend = numerics::Backend::Fortran;
    const numerics::Result r = k->run(s);
    const std::string js = r.to_json().str();

    for (const char* key : {"\"kernel\"", "\"method\"", "\"backend\"", "\"units\"",
                            "\"fingerprint\"", "\"diagnostics\"", "\"elements\""})
        OCT_CHECK_MSG(js.find(key) != std::string::npos, "missing key " << key);

    // Without a Fortran compiler the backend string and status must explicitly
    // report UNSUPPORTED_HARDWARE_SKIP and refuse C++ fallback.
    if (!numerics::KernelLibrary::instance().fortran_available()) {
        OCT_CHECK(r.hardware_skipped);
        OCT_CHECK(r.status.is_hardware_skip());
        OCT_CHECK(r.backend.find("UNSUPPORTED_HARDWARE_SKIP") != std::string::npos);
    }
}

OCT_TEST(engine, json_emits_strings_for_const_char_fields) {
    // Regression: a `const char*` argument to Json::field used to bind to the
    // bool overload and print true/false instead of the name.
    Json j;
    j.begin_object();
    j.field("action", "restart");
    j.field("severity", "critical");
    j.field("healthy", true);
    j.field("count", int64_t(3));
    j.end_object();
    const std::string out = j.str();
    OCT_CHECK(out.find("\"action\":\"restart\"") != std::string::npos);
    OCT_CHECK(out.find("\"severity\":\"critical\"") != std::string::npos);
    OCT_CHECK(out.find("\"healthy\":true") != std::string::npos);
    OCT_CHECK(out.find("\"count\":3") != std::string::npos);
}

OCT_TEST(engine, host_info_and_build_fingerprint_are_available) {
    const HostInfo h = host_info();
    OCT_CHECK(h.hw_threads >= 1);
    OCT_CHECK(!h.arch.empty());
    OCT_CHECK(!h.os.empty());
    const std::string fp = build_fingerprint();
    OCT_CHECK(!fp.empty());
    OCT_CHECK(fp == build_fingerprint());              // deterministic
}
