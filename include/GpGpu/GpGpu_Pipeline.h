#ifndef GPGPU_PIPELINE_H
#define GPGPU_PIPELINE_H

/// In-process GPU MEC pipeline control (gpu-instream plan).
///
/// Env:
///   MICMAC_GPU_PIPELINE=0|1     1 = in-process multi-stream GPU (default after P-G)
///                               0 = legacy process-per-box GPU
///   MICMAC_GPU_LEGACY_PROCESS=1 force make -j GPU boxes even if pipeline on
///
/// Phase G (post soak): default pipeline-on. Escape with PIPELINE=0 or LEGACY_PROCESS=1.

#include <cstdlib>
#include <cstdio>
#include <cstring>
#if !defined(_WIN32)
#include <unistd.h>
#endif

namespace gpgpu_pipeline {

inline int SelfPid()
{
#if defined(_WIN32)
    return 0;
#else
    return (int)getpid();
#endif
}

inline bool EnvTruthy(const char * e)
{
    if (!e || !e[0])
        return false;
    if (e[0] == '0' || !std::strcmp(e, "off") || !std::strcmp(e, "false")
        || !std::strcmp(e, "no"))
        return false;
    return true; // "1", "on", "true", "yes", anything else non-false
}

/// Default **on** after P-G soak (2026-07-30). Explicit 0/off restores legacy process path.
inline bool PipelineEnabled()
{
    static int s = -1;
    if (s >= 0)
        return s != 0;
    const char * e = std::getenv("MICMAC_GPU_PIPELINE");
    if (!e || !e[0])
        s = 1; // Phase G production default
    else
        s = EnvTruthy(e) ? 1 : 0;
    return s != 0;
}

/// Emergency / A/B escape: restore process fan-out for GPU boxes.
inline bool LegacyProcessForced()
{
    static int s = -1;
    if (s >= 0)
        return s != 0;
    const char * e = std::getenv("MICMAC_GPU_LEGACY_PROCESS");
    s = EnvTruthy(e) ? 1 : 0;
    return s != 0;
}

/// True when GPU boxes should run in the master process (no make -j).
inline bool UseInProcessBoxes()
{
    if (LegacyProcessForced())
        return false;
    return PipelineEnabled();
}

inline void LogModeOnce(bool gpuStage, int userByProcess)
{
    static bool done = false;
    if (done)
        return;
    done = true;
    const char * mode =
        !gpuStage ? "n/a-cpu-etape"
        : UseInProcessBoxes() ? "inprocess"
        : LegacyProcessForced() ? "legacy-forced"
        : "legacy-process";
    std::fprintf(stderr,
        "[GPGPU][PIPELINE] mode=%s pipeline=%d legacy_process=%d "
        "ByProcess=%d pid=%d (MICMAC_GPU_PIPELINE / MICMAC_GPU_LEGACY_PROCESS)\n",
        mode,
        PipelineEnabled() ? 1 : 0,
        LegacyProcessForced() ? 1 : 0,
        userByProcess,
        SelfPid());
    std::fflush(stderr);
}

} // namespace gpgpu_pipeline

#endif // GPGPU_PIPELINE_H
