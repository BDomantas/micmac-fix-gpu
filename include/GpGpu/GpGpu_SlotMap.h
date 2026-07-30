#ifndef GPGPU_SLOT_MAP_H
#define GPGPU_SLOT_MAP_H

/// PR-D: pure host-buffer-id → device stream/slot mapping (unit-testable).
///
/// Rules:
///   - active_slots clamped to [1, nstream]
///   - slot = idBuf % active_slots  (no collapse of both ids to 0 when active_slots≥2)
///   - when active_slots==1, all traffic uses slot 0

#include <algorithm>

namespace gpgpu_slot {

inline int ClampActive(int activeSlots, int nstream)
{
    if (nstream < 1)
        nstream = 1;
    if (activeSlots < 1)
        activeSlots = 1;
    if (activeSlots > nstream)
        activeSlots = nstream;
    return activeSlots;
}

/// Map host ring id to device slot.
inline int MapHostToSlot(int idBuf, int activeSlots, int nstream = 2)
{
    const int a = ClampActive(activeSlots, nstream);
    if (a <= 1)
        return 0;
    // Normalize id to non-negative
    int id = idBuf;
    if (id < 0)
        id = -id;
    return id % a;
}

/// True when two consecutive host ids use distinct device slots (dual depth live).
inline bool DistinctSlotsForRing(int activeSlots, int nstream = 2)
{
    const int a = ClampActive(activeSlots, nstream);
    if (a < 2)
        return false;
    return MapHostToSlot(0, a, nstream) != MapHostToSlot(1, a, nstream);
}

/// Max prep-ahead depth for host MT loop (bounded by ring size and active slots).
inline int PrepAheadDepth(int activeSlots, int ringSize = 2)
{
    const int a = ClampActive(activeSlots, ringSize);
    int d = a;
    if (d > ringSize)
        d = ringSize;
    if (d < 1)
        d = 1;
    return d;
}

} // namespace gpgpu_slot

#endif // GPGPU_SLOT_MAP_H
