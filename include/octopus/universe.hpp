// Octopus Hybrid AI Engine -- Physical Universe Image/Video Generation Engine.
//
// Requests raw 4-channel coordinate structural matrices from the Fortran 2023
// Physics layer (`oct_f_universe_field`: real light tracking, Fresnel spatial
// reflection, Navier-Stokes fluid inertia, and binary Kepler/N-body mass
// gravity), projects them through APL right-to-left matrix operations, verifies
// physical conservation invariants via Prolog SLD resolution, and translates
// those mathematical matrices directly into the 20-colour structural color
// spaces of the Piet Canvas interface (PNG, PPM, and YUV4MPEG2 .y4m video).
//
// FORCED MULTI-LANGUAGE BINDING REGIME:
// Live physical universe coordinate generation requires the native Fortran 2023
// compiler (`gfortran`/`lfortran`). When absent on the host, `synthesize_frame`
// and `synthesize_video` return UNSUPPORTED_HARDWARE_SKIP rather than
// substituting pseudo-C++ physics approximations.
//
// SPDX-License-Identifier: MIT
#pragma once

#include "octopus/apl.hpp"
#include "octopus/core.hpp"
#include "octopus/module.hpp"
#include "octopus/numerics.hpp"
#include "octopus/piet.hpp"
#include "octopus/prolog.hpp"

#include <span>
#include <string>
#include <vector>

namespace oct::universe {

struct SceneSpec {
    int         width = 64;
    int         height = 64;
    int         frames = 1;
    double      time_t = 0.25;
    double      dt = 0.08;
    double      light_x = -0.4;
    double      light_y = 0.5;
    double      light_z = 1.0;
    double      refl_ior = 1.52;     // Fresnel glass index of refraction
    double      reynolds = 250.0;    // Navier-Stokes vortex fluid inertia Re
    double      gm1 = 1.0;           // Primary mass G*M1
    double      gm2 = 0.45;          // Secondary mass G*M2
    double      orbit_e = 0.35;      // Binary orbital eccentricity
    std::string output_path;         // .png, .ppm, or .y4m
};

struct FrameResult {
    int                  width = 0;
    int                  height = 0;
    double               time_t = 0.0;
    bool                 hardware_skipped = false;
    Status               status = Status::ok();
    std::string          backend;
    // Raw 4-channel coordinate matrix [width * height * 4]:
    // ch0 = light_radiance, ch1 = spatial_reflection,
    // ch2 = fluid_inertia,  ch3 = gravity_potential
    std::vector<double>  raw_coordinates;
    // Physical diagnostics from Fortran 2023 layer:
    double               total_radiance = 0.0;
    double               max_reflection = 0.0;
    double               fluid_kinetic_energy = 0.0;
    double               gravity_potential_energy = 0.0;
    // APL + Prolog + Piet translation outputs:
    std::vector<double>  projected_field;
    prolog::SoundnessReport sld_report;
    piet::Raster         canvas;
    piet::CodelTrace     codel_trace;
    std::string          output_file;

    Json to_json() const;
};

struct VideoResult {
    int                      width = 0;
    int                      height = 0;
    int                      frame_count = 0;
    bool                     hardware_skipped = false;
    Status                   status = Status::ok();
    std::string              backend;
    std::vector<FrameResult> frames;
    std::string              output_file;
    std::string              stream_sha256;

    Json to_json() const;
};

// Translate a 4-channel coordinate matrix [width * height * 4]
// (light_radiance, spatial_reflection, fluid_inertia, gravity_potential)
// through APL matrix projection and Prolog SLD verification directly into a
// Piet 20-colour structural canvas.
Outcome<FrameResult> translate_coordinates_to_piet(std::span<const double> coords_4ch,
                                                   int width, int height,
                                                   double time_t = 0.0,
                                                   const double diag_4[4] = nullptr);

// Request raw coordinate structural data from the Fortran 2023 Physics layer
// (`oct_f_universe_field`) and translate into the Piet Canvas interface.
// Returns hardware_skipped=true (UNSUPPORTED_HARDWARE_SKIP) if Fortran 2023
// compiler was absent at build time.
FrameResult synthesize_frame(const SceneSpec& spec);

// Synthesize a multi-frame physical universe video sequence and optionally
// write a YUV4MPEG2 (.y4m) stream derived from the Piet Canvas frames.
VideoResult synthesize_video(const SceneSpec& spec);

// Write a sequence of Piet Rasters as an uncompressed YUV4MPEG2 (.y4m) video file.
Status write_y4m_video(const std::string& path,
                       const std::vector<piet::Raster>& frames,
                       int fps = 24);

std::shared_ptr<oct::Module> make_universe_module();

}  // namespace oct::universe
