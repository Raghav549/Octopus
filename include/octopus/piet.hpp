// Octopus Hybrid AI Engine -- Piet-style cognitive-state visualization.
//
// Role in the engine: turn a numeric cognitive state (attention vector, kernel
// diagnostics, module health) into an image in the 20-colour Piet palette, plus
// a legend that maps colours back to states. Output is a real image file (PNG
// written with stored deflate blocks, no external dependency) and/or PPM.
//
// Honest scope: this is a *visualization* layer. The engine does not execute
// Piet programs (no Piet interpreter is implemented), and colour mapping is a
// documented, reversible projection of the state vector -- not a claim about
// cognition.
// SPDX-License-Identifier: MIT
#pragma once

#include "octopus/core.hpp"

#include <span>

namespace oct { class Module; }

namespace oct::piet {

struct Rgb { uint8_t r = 0, g = 0, b = 0; };

// The 18 chromatic Piet colours in hue/lightness order, then black and white.
const std::vector<Rgb>& palette();
// Index 0..17 = chromatic; 18 = black; 19 = white.
int palette_index_for(double value);
const char* palette_name(int index);

struct Raster {
    int width = 0;
    int height = 0;
    std::vector<uint8_t> rgb;      // width*height*3
    std::vector<int>    codes;     // palette indices, for round-trip checks

    Status write_ppm(const std::string& path) const;
    Status write_png(const std::string& path) const;   // stored deflate, no zlib
    std::string fingerprint() const;                   // SHA-256 over the pixel data
};

// Map a state vector onto a width x height Piet block grid. Values are
// normalised against `lo`/`hi`; NaN/Inf are mapped to black (a documented
// "invalid" marker rather than silently dropping them).
Raster render_state(std::span<const double> state, int width, int height, double lo, double hi);

// Quantise a state vector through the palette and back (lossy, documented).
std::vector<int> quantise(std::span<const double> state, double lo, double hi);
std::vector<double> dequantise(std::span<const int> codes, double lo, double hi);

// Visual Canvas Logic Engine: render cognitive execution pathways and neural
// synapse configurations into multi-tonal, color-metric Piet pixel arrays and
// trace the Piet (dHue, dLightness) codel transition instructions.
struct CodelTrace {
    int64_t                  codel_transitions = 0;
    std::vector<std::string> ops_executed;
    std::vector<int64_t>     stack_snapshot;
    std::string              canvas_fingerprint;
    Json                     to_json() const;
};

Raster render_synapse_canvas(std::span<const double> weights,
                             std::span<const double> activations,
                             int width, int height);

CodelTrace execute_canvas_pathway(const Raster& canvas, int64_t max_steps = 128);

Json palette_legend();
std::shared_ptr<oct::Module> make_piet_module();

}  // namespace oct::piet
