#ifndef GPGPU_DIAG_H
#define GPGPU_DIAG_H

/// Runtime GpGpu / MICMAC CUDA diagnostic verbosity.
///
/// Env: MICMAC_GPGPU_DIAG
///   0 / off / error  — errors only (and fatal aborts)
///   1 / min / minimal — STAGE lines, long-wait WARNINGs, errors (default)
///   2 / full / all   — every correl/optim cell, simpleJob SPAWN, heartbeats
///
/// Also accepted: true/yes → 2, false/no → 0.

#include <cstdlib>
#include <cstdio>
#include <cstring>

inline int GpgpuDiagLevel()
{
    static int sLevel = -1;
    if (sLevel >= 0)
        return sLevel;

    const char * e = std::getenv("MICMAC_GPGPU_DIAG");
    if (!e || !e[0])
    {
        sLevel = 1; // production-friendly default: stages only
        return sLevel;
    }

    if (e[0] == '0' || !std::strcmp(e, "off") || !std::strcmp(e, "error")
        || !std::strcmp(e, "false") || !std::strcmp(e, "no"))
        sLevel = 0;
    else if (e[0] == '2' || !std::strcmp(e, "full") || !std::strcmp(e, "all")
             || !std::strcmp(e, "true") || !std::strcmp(e, "yes"))
        sLevel = 2;
    else
        sLevel = 1; // "1", "min", "minimal", anything else

    // One-time banner so operators know which mode is active.
    static const char * names[] = {"error-only", "minimal", "full"};
    std::fprintf(stderr,
                 "[GPGPU][RUNPOD_GPGPU_DIAG] log level=%d (%s) "
                 "via MICMAC_GPGPU_DIAG (0=error,1=min,2=full)\n",
                 sLevel, names[sLevel < 0 ? 1 : (sLevel > 2 ? 2 : sLevel)]);
    std::fflush(stderr);

    return sLevel;
}

inline bool GpgpuDiagMin()  { return GpgpuDiagLevel() >= 1; }
inline bool GpgpuDiagFull() { return GpgpuDiagLevel() >= 2; }

/// Always print (errors / hard failures).
#define GPGPU_DIAG_ERR(...)                                                         \
    do {                                                                            \
        std::fprintf(stderr, __VA_ARGS__);                                          \
        std::fflush(stderr);                                                        \
    } while (0)

/// Stage / warning level (default on).
#define GPGPU_DIAG_MIN(...)                                                         \
    do {                                                                            \
        if (GpgpuDiagMin()) {                                                       \
            std::fprintf(stderr, __VA_ARGS__);                                      \
            std::fflush(stderr);                                                    \
        }                                                                           \
    } while (0)

/// Verbose spam (default off).
#define GPGPU_DIAG_FULL(...)                                                        \
    do {                                                                            \
        if (GpgpuDiagFull()) {                                                      \
            std::fprintf(stderr, __VA_ARGS__);                                      \
            std::fflush(stderr);                                                    \
        }                                                                           \
    } while (0)

#endif // GPGPU_DIAG_H
