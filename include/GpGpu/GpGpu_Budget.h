#ifndef GPGPU_BUDGET_H
#define GPGPU_BUDGET_H

/// VRAM / stream-slot budget (successor to process-count auto NbProc for GPU pipeline).
///
/// Env:
///   MICMAC_GPU_MAX_SLOTS=N   cap concurrent stream slots (default min(NSTREAM,4))
///   MICMAC_GPU_SAFETY=0.80   VRAM fraction (falls back to MICMAC_GPU_AUTO_SAFETY)
///   MICMAC_GPU_PREFETCH=0|1  cross-box host prefetch (default 0)

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <algorithm>

// Avoid pulling GpGpu_Defines.h (CUDA host/device attributes). Mirror NSTREAM for host tests.
#ifndef NSTREAM
#define GPGPU_BUDGET_NSTREAM_DEFAULT 2
#else
#define GPGPU_BUDGET_NSTREAM_DEFAULT NSTREAM
#endif

namespace gpgpu_budget {

inline int MaxSlotsEnv()
{
    const char * e = std::getenv("MICMAC_GPU_MAX_SLOTS");
    int n = GPGPU_BUDGET_NSTREAM_DEFAULT;
    if (e && e[0])
    {
        int v = std::atoi(e);
        if (v >= 1)
            n = v;
    }
    if (n > GPGPU_BUDGET_NSTREAM_DEFAULT)
        n = GPGPU_BUDGET_NSTREAM_DEFAULT;
    if (n < 1)
        n = 1;
    return n;
}

inline double SafetyFrac()
{
    const char * e = std::getenv("MICMAC_GPU_SAFETY");
    if (!e || !e[0])
        e = std::getenv("MICMAC_GPU_AUTO_SAFETY");
    double s = 0.80;
    if (e && e[0])
    {
        double v = std::atof(e);
        if (v > 0.3 && v <= 1.0)
            s = v;
    }
    return s;
}

inline bool PrefetchEnabled()
{
    const char * e = std::getenv("MICMAC_GPU_PREFETCH");
    if (!e || !e[0])
        return false;
    if (e[0] == '0' || !std::strcmp(e, "off") || !std::strcmp(e, "false") || !std::strcmp(e, "no"))
        return false;
    return true;
}

/// Pure formula: given total VRAM and peak used bytes for one slot, return slots in [1,maxSlots].
/// Host-unit-testable without a CUDA device.
inline int ComputeMaxSlots(size_t totalBytes, size_t peakUsedBytes, int maxSlots, double safety)
{
    if (maxSlots < 1)
        maxSlots = 1;
    if (peakUsedBytes == 0)
        return 1;
    if (peakUsedBytes < 64ull * 1024 * 1024)
        peakUsedBytes = 64ull * 1024 * 1024;
    if (safety <= 0.3)
        safety = 0.80;
    if (safety > 1.0)
        safety = 1.0;
    if (totalBytes == 0)
        return 1;
    size_t budget = (size_t)(totalBytes * safety);
    int n = (int)(budget / peakUsedBytes);
    if (n < 1)
        n = 1;
    if (n > maxSlots)
        n = maxSlots;
    return n;
}

inline int ClampActiveSlots(int desired)
{
    int cap = MaxSlotsEnv();
    if (desired < 1)
        desired = 1;
    if (desired > cap)
        desired = cap;
    return desired;
}

} // namespace gpgpu_budget

#endif // GPGPU_BUDGET_H
