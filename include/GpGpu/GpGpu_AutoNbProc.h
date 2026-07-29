#ifndef GPGPU_AUTO_NBPROC_H
#define GPGPU_AUTO_NBPROC_H

/// Adaptive GPU box parallelism for MicMac MEC.
///
/// Env:
///   MICMAC_GPU_AUTO_NBPROC=1|0     enable (default 1)
///   MICMAC_GPU_AUTO_NBPROC_MAX=N   hard cap concurrent GPU workers (default 2)
///   MICMAC_GPU_AUTO_SAFETY=0.80    use at most this fraction of total VRAM
///
/// Workflow (master process, per DeZoom etape):
///   1) ResetPeak()
///   2) Run box 0 in-process (DoOneBloc) — SampleGpuNow() during correl/opt
///   3) N = ComputeNbProc(userMax)
///   4) ExeProcessParallelisable remaining boxes with ByProcess=N

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>

#if CUDA_ENABLED
#include <cuda_runtime.h>
#endif

namespace gpgpu_auto {

inline bool Enabled()
{
    const char * e = std::getenv("MICMAC_GPU_AUTO_NBPROC");
    if (!e || !e[0])
        return true; // default on
    if (e[0] == '0' || !std::strcmp(e, "off") || !std::strcmp(e, "false") || !std::strcmp(e, "no"))
        return false;
    return true;
}

inline int MaxCap()
{
    const char * e = std::getenv("MICMAC_GPU_AUTO_NBPROC_MAX");
    int n = 2;
    if (e && e[0])
    {
        int v = std::atoi(e);
        if (v >= 1)
            n = v;
    }
    // Hard safety for single consumer GPU without MPS
    if (n > 3)
        n = 3;
    return n;
}

inline double SafetyFrac()
{
    const char * e = std::getenv("MICMAC_GPU_AUTO_SAFETY");
    double s = 0.80;
    if (e && e[0])
    {
        double v = std::atof(e);
        if (v > 0.3 && v <= 1.0)
            s = v;
    }
    return s;
}

struct PeakState
{
    bool     active;
    size_t   totalBytes;
    size_t   minFreeBytes;
    size_t   sampleCount;
    long     hostRssKb;
};

inline PeakState & State()
{
    static PeakState s = { false, 0, (size_t)-1, 0, 0 };
    return s;
}

inline void ResetPeak()
{
    PeakState & s = State();
    s.active = true;
    s.totalBytes = 0;
    s.minFreeBytes = (size_t)-1;
    s.sampleCount = 0;
    s.hostRssKb = 0;
}

inline void StopPeak()
{
    State().active = false;
}

inline long ReadHostRssKb()
{
#if defined(__linux__)
    FILE * f = std::fopen("/proc/self/status", "r");
    if (!f)
        return 0;
    char line[256];
    long rss = 0;
    while (std::fgets(line, sizeof(line), f))
    {
        if (std::strncmp(line, "VmRSS:", 6) == 0)
        {
            std::sscanf(line + 6, "%ld", &rss);
            break;
        }
    }
    std::fclose(f);
    return rss;
#else
    return 0;
#endif
}

inline long ReadMemAvailableKb()
{
#if defined(__linux__)
    FILE * f = std::fopen("/proc/meminfo", "r");
    if (!f)
        return 0;
    char line[256];
    long avail = 0;
    while (std::fgets(line, sizeof(line), f))
    {
        if (std::strncmp(line, "MemAvailable:", 13) == 0)
        {
            std::sscanf(line + 13, "%ld", &avail);
            break;
        }
    }
    std::fclose(f);
    return avail;
#else
    return 0;
#endif
}

/// Call while GPU correl/opt buffers are still allocated (peak window).
inline void SampleGpuNow()
{
    PeakState & s = State();
    if (!s.active)
        return;
#if CUDA_ENABLED
    size_t freeB = 0, totB = 0;
    if (cudaMemGetInfo(&freeB, &totB) != cudaSuccess)
        return;
    if (s.sampleCount == 0 || totB > s.totalBytes)
        s.totalBytes = totB;
    if (freeB < s.minFreeBytes)
        s.minFreeBytes = freeB;
    s.sampleCount++;
#else
    (void)s;
#endif
    long rss = ReadHostRssKb();
    if (rss > s.hostRssKb)
        s.hostRssKb = rss;
}

/// userMax: requested ByProcess/NbProc from CLI (>=1). Returns N in [1, min(userMax, MaxCap())].
inline int ComputeNbProc(int userMax)
{
    if (userMax < 1)
        userMax = 1;
    int cap = MaxCap();
    int nMax = std::min(userMax, cap);

    PeakState & s = State();
    if (s.sampleCount == 0 || s.totalBytes == 0 || s.minFreeBytes == (size_t)-1)
    {
        std::fprintf(stderr,
            "[GPGPU][AUTO_NBPROC] no GPU samples — keep NbProc=%d\n", nMax);
        std::fflush(stderr);
        return nMax;
    }

    size_t peakUsed = (s.totalBytes > s.minFreeBytes)
                          ? (s.totalBytes - s.minFreeBytes)
                          : s.totalBytes;
    if (peakUsed < 64ull * 1024 * 1024)
        peakUsed = 64ull * 1024 * 1024; // floor 64MB

    double safety = SafetyFrac();
    size_t budget = (size_t)(s.totalBytes * safety);
    int nGpu = (int)(budget / peakUsed);
    if (nGpu < 1)
        nGpu = 1;

    // Host RSS cap (optional): concurrent workers ~ N * rss of probe process
    int nHost = nMax;
    if (s.hostRssKb > 0)
    {
        long availKb = ReadMemAvailableKb();
        if (availKb > 0)
        {
            // leave 4GB headroom for OS / parent
            long usable = availKb - 4L * 1024 * 1024;
            if (usable < s.hostRssKb)
                nHost = 1;
            else
                nHost = (int)(usable / s.hostRssKb);
            if (nHost < 1)
                nHost = 1;
        }
    }

    int n = std::min(nMax, std::min(nGpu, nHost));
    if (n < 1)
        n = 1;

    std::fprintf(stderr,
        "[GPGPU][AUTO_NBPROC] peakUsed=%.1f MiB total=%.1f MiB minFree=%.1f MiB "
        "rss=%.1f MiB → N_gpu=%d N_host=%d Nmax=%d → **NbProc=%d**\n",
        peakUsed / (1024.0 * 1024.0),
        s.totalBytes / (1024.0 * 1024.0),
        s.minFreeBytes / (1024.0 * 1024.0),
        s.hostRssKb / 1024.0,
        nGpu, nHost, nMax, n);
    std::fflush(stderr);

    return n;
}

} // namespace gpgpu_auto

#endif // GPGPU_AUTO_NBPROC_H
