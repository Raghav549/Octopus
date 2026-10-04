// Octopus Hybrid AI Engine -- Piet-style visualization (PNG writer included).
// SPDX-License-Identifier: MIT
#include "octopus/piet.hpp"

#include "octopus/module.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>

namespace oct::piet {

namespace {

const Rgb kPalette[20] = {
    {255, 192, 192}, {255, 0, 0},     {192, 0, 0},     {255, 255, 192}, {255, 255, 0},
    {192, 192, 0},   {192, 255, 192}, {0, 255, 0},     {0, 192, 0},     {192, 255, 255},
    {0, 255, 255},   {0, 192, 192},   {192, 192, 255}, {0, 0, 255},     {0, 0, 192},
    {255, 192, 255}, {255, 0, 255},   {192, 0, 192},   {0, 0, 0},       {255, 255, 255}
};

uint32_t crc32_of(std::span<const uint8_t> data) {
    static uint32_t table[256];
    static bool init = false;
    if (!init) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : (c >> 1);
            table[i] = c;
        }
        init = true;
    }
    uint32_t c = 0xFFFFFFFFu;
    for (uint8_t b : data) c = table[(c ^ b) & 0xFF] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

void put_be32(std::vector<uint8_t>& out, uint32_t v) {
    out.push_back(uint8_t(v >> 24));
    out.push_back(uint8_t(v >> 16));
    out.push_back(uint8_t(v >> 8));
    out.push_back(uint8_t(v));
}

void png_chunk(std::vector<uint8_t>& out, const char type[4], std::span<const uint8_t> data) {
    put_be32(out, uint32_t(data.size()));
    const size_t start = out.size();
    out.insert(out.end(), type, type + 4);
    out.insert(out.end(), data.begin(), data.end());
    const uint32_t crc = crc32_of(std::span<const uint8_t>(out.data() + start, out.size() - start));
    put_be32(out, crc);
}

}  // namespace

const std::vector<Rgb>& palette() {
    static const std::vector<Rgb> p(kPalette, kPalette + 20);
    return p;
}

int palette_index_for(double value) {
    if (!std::isfinite(value)) return 18;   // black marks an invalid reading
    double v = std::min(1.0, std::max(0.0, value));
    return int(v * 17.999);
}

const char* palette_name(int index) {
    static const char* names[20] = {
        "light-red", "red", "dark-red", "light-yellow", "yellow", "dark-yellow",
        "light-green", "green", "dark-green", "light-cyan", "cyan", "dark-cyan",
        "light-blue", "blue", "dark-blue", "light-magenta", "magenta", "dark-magenta",
        "black", "white"};
    return (index >= 0 && index < 20) ? names[index] : "?";
}

std::vector<int> quantise(std::span<const double> state, double lo, double hi) {
    std::vector<int> out;
    out.reserve(state.size());
    const double span = (hi > lo) ? (hi - lo) : 1.0;
    for (double v : state) {
        if (!std::isfinite(v)) {
            out.push_back(18);
            continue;
        }
        out.push_back(palette_index_for((v - lo) / span));
    }
    return out;
}

std::vector<double> dequantise(std::span<const int> codes, double lo, double hi) {
    std::vector<double> out;
    out.reserve(codes.size());
    const double span = hi - lo;
    for (int c : codes) {
        if (c == 18) { out.push_back(std::nan("")); continue; }
        if (c == 19) { out.push_back(hi); continue; }
        out.push_back(lo + span * (double(c) + 0.5) / 18.0);
    }
    return out;
}

Raster render_state(std::span<const double> state, int width, int height, double lo, double hi) {
    Raster r;
    r.width = std::max(1, width);
    r.height = std::max(1, height);
    r.rgb.assign(size_t(r.width) * size_t(r.height) * 3, 0);
    r.codes.assign(size_t(r.width) * size_t(r.height), 18);
    if (state.empty()) return r;
    // Row-major fill so the image reads like the flattened state tensor.
    for (size_t i = 0; i < r.codes.size(); ++i) {
        const double v = state[i % state.size()];
        const int idx = palette_index_for((v - lo) / ((hi > lo) ? (hi - lo) : 1.0));
        r.codes[i] = idx;
        const Rgb& c = kPalette[idx];
        r.rgb[i * 3 + 0] = c.r;
        r.rgb[i * 3 + 1] = c.g;
        r.rgb[i * 3 + 2] = c.b;
    }
    return r;
}

std::string Raster::fingerprint() const { return hash::sha256_hex(std::string_view(
    reinterpret_cast<const char*>(rgb.data()), rgb.size())); }

Status Raster::write_ppm(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    if (!f) return Status::unavailable("piet: cannot open " + path);
    f << "P6\n" << width << " " << height << "\n255\n";
    f.write(reinterpret_cast<const char*>(rgb.data()), std::streamsize(rgb.size()));
    if (!f) return Status::internal("piet: write failed for " + path);
    return Status::ok();
}

Status Raster::write_png(const std::string& path) const {
    // PNG with zlib "stored" deflate blocks: real, viewable, dependency-free.
    std::vector<uint8_t> raw;
    raw.reserve(size_t(height) * (1 + size_t(width) * 3));
    for (int y = 0; y < height; ++y) {
        raw.push_back(0);   // filter type 0 (None)
        const uint8_t* row = rgb.data() + size_t(y) * size_t(width) * 3;
        raw.insert(raw.end(), row, row + size_t(width) * 3);
    }
    // zlib wrapper: 0x78 0x01, then stored blocks (max 65535 bytes each), Adler-32.
    std::vector<uint8_t> z;
    z.push_back(0x78);
    z.push_back(0x01);
    size_t pos = 0;
    while (pos < raw.size()) {
        const size_t len = std::min<size_t>(65535, raw.size() - pos);
        const bool last = (pos + len == raw.size());
        z.push_back(last ? 1 : 0);
        z.push_back(uint8_t(len & 0xFF));
        z.push_back(uint8_t((len >> 8) & 0xFF));
        z.push_back(uint8_t(~len & 0xFF));
        z.push_back(uint8_t((~len >> 8) & 0xFF));
        z.insert(z.end(), raw.begin() + std::ptrdiff_t(pos), raw.begin() + std::ptrdiff_t(pos + len));
        pos += len;
    }
    uint32_t a = 1, b = 0;
    for (uint8_t byte : raw) {
        a = (a + byte) % 65521;
        b = (b + a) % 65521;
    }
    put_be32(z, (b << 16) | a);

    std::vector<uint8_t> png = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    std::vector<uint8_t> ihdr;
    put_be32(ihdr, uint32_t(width));
    put_be32(ihdr, uint32_t(height));
    ihdr.push_back(8);    // bit depth
    ihdr.push_back(2);    // colour type: truecolour RGB
    ihdr.push_back(0);    // deflate
    ihdr.push_back(0);    // adaptive filtering
    ihdr.push_back(0);    // no interlace
    png_chunk(png, "IHDR", ihdr);
    png_chunk(png, "IDAT", z);
    png_chunk(png, "IEND", {});

    std::ofstream f(path, std::ios::binary);
    if (!f) return Status::unavailable("piet: cannot open " + path);
    f.write(reinterpret_cast<const char*>(png.data()), std::streamsize(png.size()));
    if (!f) return Status::internal("piet: write failed for " + path);
    return Status::ok();
}

Json palette_legend() {
    Json j;
    j.begin_object();
    j.field("palette", "Piet (20 colours)");
    j.key("entries");
    j.begin_array();
    for (int i = 0; i < 20; ++i) {
        j.begin_object();
        j.field("index", int64_t(i));
        j.field("name", palette_name(i));
        char hex[8];
        std::snprintf(hex, sizeof(hex), "#%02x%02x%02x", kPalette[i].r, kPalette[i].g, kPalette[i].b);
        j.field("hex", const_cast<const char*>(hex));
        j.end_object();
    }
    j.end_array();
    j.end_object();
    return j;
}

Json CodelTrace::to_json() const {
    Json j;
    j.begin_object();
    j.field("codel_transitions", codel_transitions);
    j.field("canvas_fingerprint", canvas_fingerprint);
    j.key("ops_executed");
    j.begin_array();
    for (const auto& op : ops_executed) j.value(op);
    j.end_array();
    j.key("stack_snapshot");
    j.begin_array();
    for (int64_t v : stack_snapshot) j.value(v);
    j.end_array();
    j.end_object();
    return j;
}

Raster render_synapse_canvas(std::span<const double> weights,
                             std::span<const double> activations,
                             int width, int height) {
    const int w = std::max(4, width);
    const int h = std::max(4, height);
    std::vector<double> combined(size_t(w * h), 0.0);
    double lo = -1.0, hi = 1.0;
    if (!weights.empty() || !activations.empty()) {
        lo = 1e30;
        hi = -1e30;
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const size_t idx = size_t(y * w + x);
                const double wt = weights.empty() ? 0.0 : weights[idx % weights.size()];
                const double act = activations.empty() ? 1.0 : activations[size_t(x) % activations.size()];
                const double val = std::tanh(wt * act + 0.15 * std::sin(double(x + y)));
                combined[idx] = val;
                lo = std::min(lo, val);
                hi = std::max(hi, val);
            }
        }
        if (hi - lo < 1e-9) { lo = -1.0; hi = 1.0; }
    }
    return render_state(combined, w, h, lo, hi);
}

CodelTrace execute_canvas_pathway(const Raster& canvas, int64_t max_steps) {
    CodelTrace tr;
    tr.canvas_fingerprint = canvas.fingerprint();
    if (canvas.codes.empty() || canvas.width <= 0 || canvas.height <= 0) return tr;

    static const char* kOpTable[6][3] = {
        {"nop",  "push", "pop"},
        {"add",  "sub",  "mul"},
        {"div",  "mod",  "not"},
        {"gt",   "ptr",  "swi"},
        {"dup",  "roll", "in_n"},
        {"in_c", "out_n","out_c"},
    };

    std::vector<int64_t> st;
    int prev_code = canvas.codes[0];
    const size_t limit = std::min<size_t>(canvas.codes.size(), size_t(std::max<int64_t>(1, max_steps)));
    for (size_t i = 1; i < limit; ++i) {
        const int cur = canvas.codes[i];
        if (prev_code >= 0 && prev_code < 18 && cur >= 0 && cur < 18 && cur != prev_code) {
            const int h0 = prev_code / 3, l0 = prev_code % 3;
            const int h1 = cur / 3,       l1 = cur % 3;
            const int dh = (h1 - h0 + 6) % 6;
            const int dl = (l1 - l0 + 3) % 3;
            const char* op = kOpTable[dh][dl];
            tr.ops_executed.push_back(op);
            ++tr.codel_transitions;
            if (std::strcmp(op, "push") == 0) {
                st.push_back(int64_t(h0 + l0 + 1));
            } else if (std::strcmp(op, "pop") == 0 && !st.empty()) {
                st.pop_back();
            } else if (std::strcmp(op, "dup") == 0 && !st.empty()) {
                st.push_back(st.back());
            } else if (std::strcmp(op, "add") == 0 && st.size() >= 2) {
                const int64_t a = st.back(); st.pop_back();
                st.back() += a;
            } else if (std::strcmp(op, "sub") == 0 && st.size() >= 2) {
                const int64_t a = st.back(); st.pop_back();
                st.back() -= a;
            } else if (std::strcmp(op, "mul") == 0 && st.size() >= 2) {
                const int64_t a = st.back(); st.pop_back();
                st.back() *= a;
            }
        }
        prev_code = cur;
    }
    tr.stack_snapshot = std::move(st);
    return tr;
}

// ---------------------------------------------------------------------------
// Module wrapper
// ---------------------------------------------------------------------------
namespace {

class PietModule final : public Module {
public:
    ModuleInfo info() const override {
        ModuleInfo i;
        i.name = "piet.visual";
        i.version = "1.0.0";
        i.language = "Piet-palette rendering (C++20)";
        i.role = "renders cognitive state and kernel diagnostics as viewable images";
        i.trust = Trust::Core;
        i.compiled_in = true;
        i.capabilities = {"piet.render", "piet.palette", "piet.quantise", "image.write_png",
                          "image.write_ppm"};
        i.limitations = {
            "visualization only: the engine does not execute Piet programs",
            "quantisation is lossy by design; the palette round-trip error is reported, not zero",
            "PNG output uses stored deflate blocks (valid but not size-optimised)",
        };
        i.build_id = std::string("c++20/") + __VERSION__;
        return i;
    }
    std::string describe() const override {
        return "Maps a numeric state vector onto the 20-colour Piet palette and writes "
               "real PNG/PPM images with a legend for interpretation.";
    }
    Status self_check() override {
        std::vector<double> state(64);
        for (size_t i = 0; i < state.size(); ++i) state[i] = std::sin(double(i) * 0.3);
        // NaN must be visible, not silently dropped.
        state[7] = std::nan("");
        Raster r = render_state(state, 8, 8, -1.0, 1.0);
        if (r.width != 8 || r.height != 8 || r.rgb.size() != 8u * 8u * 3u)
            return Status::internal("piet: raster geometry wrong");
        if (r.codes[7] != 18) return Status::internal("piet: NaN did not map to the invalid marker");
        auto codes = quantise(state, -1.0, 1.0);
        auto back = dequantise(codes, -1.0, 1.0);
        double max_err = 0.0;
        for (size_t i = 0; i < state.size(); ++i) {
            if (std::isnan(state[i])) continue;
            max_err = std::max(max_err, std::fabs(back[i] - state[i]));
        }
        // 18 chromatic levels over [-1,1] -> quantisation step is about 0.111.
        if (!(max_err <= 0.12)) return Status::internal("piet: quantisation error out of bounds");
        const std::string path = "/tmp/octopus_piet_selftest.png";
        Status w = r.write_png(path);
        if (!w) return Status::internal("piet: PNG write failed: " + w.message);
        std::ifstream probe(path, std::ios::binary);
        char sig[8] = {0};
        probe.read(sig, 8);
        if (!probe || std::memcmp(sig, "\x89PNG\r\n\x1a\n", 8) != 0)
            return Status::internal("piet: written file is not a PNG");
        return Status::ok();
    }
};

}  // namespace

std::shared_ptr<oct::Module> make_piet_module() { return std::make_shared<PietModule>(); }

}  // namespace oct::piet
