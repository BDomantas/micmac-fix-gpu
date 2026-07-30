#ifndef GPGPU_IMAGE_LRU_H
#define GPGPU_IMAGE_LRU_H

/// Phase H / C1: host LRU of pyramid **windows** across boxes in the same etape.
///
/// Env:
///   MICMAC_GPU_IMG_LRU=0|1     default **0** until P-H green
///   MICMAC_GPU_IMG_LRU_MB=N    capacity in MiB (default 2048)
///
/// Pure policy (no Elise / CUDA). Cover hit: same image+dezoom and stored clip covers request.
/// Single-threaded — not MT-safe.

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <list>
#include <unordered_map>
#include <cstdint>
#include <algorithm>

namespace gpgpu_img_lru {

struct ClipRect
{
    int x0, y0, x1, y1;

    ClipRect() : x0(0), y0(0), x1(0), y1(0) {}
    ClipRect(int a, int b, int c, int d) : x0(a), y0(b), x1(c), y1(d) {}

    bool valid() const { return x1 > x0 && y1 > y0; }

    /// True if *this* fully covers `req` (stored window usable for req without reload).
    bool covers(const ClipRect & req) const
    {
        return valid() && req.valid()
            && x0 <= req.x0 && y0 <= req.y0
            && x1 >= req.x1 && y1 >= req.y1;
    }

    bool exact(const ClipRect & o) const
    {
        return x0 == o.x0 && y0 == o.y0 && x1 == o.x1 && y1 == o.y1;
    }

    int64_t area() const
    {
        if (!valid())
            return 0;
        return (int64_t)(x1 - x0) * (int64_t)(y1 - y0);
    }
};

struct Key
{
    std::string image;
    int dezoom;

    Key() : dezoom(0) {}
    Key(const std::string & img, int dz) : image(img), dezoom(dz) {}

    bool operator==(const Key & o) const
    {
        return dezoom == o.dezoom && image == o.image;
    }
};

struct KeyHash
{
    size_t operator()(const Key & k) const
    {
        return std::hash<std::string>()(k.image) ^ (std::hash<int>()(k.dezoom) * 0x9e3779b9u);
    }
};

inline bool EnvTruthy(const char * e)
{
    if (!e || !e[0])
        return false;
    if (e[0] == '0' || !std::strcmp(e, "off") || !std::strcmp(e, "false")
        || !std::strcmp(e, "no"))
        return false;
    return true;
}

/// Default **off** until P-H green (plan).
inline bool Enabled()
{
    const char * e = std::getenv("MICMAC_GPU_IMG_LRU");
    if (!e || !e[0])
        return false;
    return EnvTruthy(e);
}

inline size_t CapBytes()
{
    const char * e = std::getenv("MICMAC_GPU_IMG_LRU_MB");
    long mb = 2048;
    if (e && e[0])
    {
        long v = std::atol(e);
        if (v >= 16)
            mb = v;
    }
    return (size_t)mb * 1024ull * 1024ull;
}

/// Pure cache: metadata + opaque payload pointer (ownership external or via free_fn).
template<class Payload>
class WindowLru
{
public:
    using FreeFn = void (*)(Payload);

    explicit WindowLru(size_t capBytes = 0, FreeFn freeFn = nullptr)
        : _cap(capBytes ? capBytes : CapBytes())
        , _free(freeFn)
        , _used(0)
        , _hits(0)
        , _misses(0)
        , _bytes_saved(0)
    {}

    ~WindowLru() { clear(); }

    void set_capacity(size_t capBytes) { _cap = capBytes; }
    size_t capacity() const { return _cap; }
    size_t used_bytes() const { return _used; }
    uint64_t hits() const { return _hits; }
    uint64_t misses() const { return _misses; }
    uint64_t bytes_saved() const { return _bytes_saved; }
    size_t size() const { return _map.size(); }

    void clear()
    {
        for (auto & e : _list)
        {
            if (_free)
                _free(e.payload);
        }
        _list.clear();
        _map.clear();
        _used = 0;
    }

    /// Clear when DeZoom / etape changes (plan: never share across etapes without re-key).
    void clear_etape()
    {
        clear();
        _hits = 0;
        _misses = 0;
        _bytes_saved = 0;
    }

    /// Lookup: hit if same key and stored clip covers req. Touches LRU order.
    Payload * find_cover(const Key & key, const ClipRect & req, ClipRect * outStored = nullptr)
    {
        auto it = _map.find(key);
        if (it == _map.end())
        {
            _misses++;
            return nullptr;
        }
        auto lit = it->second;
        if (!lit->clip.covers(req))
        {
            _misses++;
            return nullptr;
        }
        // move to front
        _list.splice(_list.begin(), _list, lit);
        _hits++;
        _bytes_saved += lit->bytes;
        if (outStored)
            *outStored = lit->clip;
        return &lit->payload;
    }

    /// Insert / replace entry for key. Evicts LRU until capacity allows.
    void put(const Key & key, const ClipRect & clip, Payload payload, size_t bytes)
    {
        auto it = _map.find(key);
        if (it != _map.end())
        {
            auto lit = it->second;
            _used -= lit->bytes;
            if (_free)
                _free(lit->payload);
            lit->clip = clip;
            lit->payload = payload;
            lit->bytes = bytes;
            _used += bytes;
            _list.splice(_list.begin(), _list, lit);
        }
        else
        {
            Entry e;
            e.key = key;
            e.clip = clip;
            e.payload = payload;
            e.bytes = bytes;
            _list.push_front(e);
            _map[key] = _list.begin();
            _used += bytes;
        }
        evict_to_cap();
    }

    void log_diag(const char * tag = "IMG_LRU") const
    {
        std::fprintf(stderr,
            "[GPGPU][%s] hits=%llu misses=%llu used_MiB=%.1f cap_MiB=%.1f entries=%zu bytes_saved=%llu\n",
            tag,
            (unsigned long long)_hits,
            (unsigned long long)_misses,
            _used / (1024.0 * 1024.0),
            _cap / (1024.0 * 1024.0),
            _map.size(),
            (unsigned long long)_bytes_saved);
        std::fflush(stderr);
    }

private:
    struct Entry
    {
        Key key;
        ClipRect clip;
        Payload payload {};
        size_t bytes = 0;
    };

    void evict_to_cap()
    {
        while (_used > _cap && !_list.empty())
        {
            auto last = std::prev(_list.end());
            _used -= last->bytes;
            if (_free)
                _free(last->payload);
            _map.erase(last->key);
            _list.pop_back();
        }
    }

    size_t _cap;
    FreeFn _free;
    size_t _used;
    uint64_t _hits, _misses, _bytes_saved;
    std::list<Entry> _list;
    std::unordered_map<Key, typename std::list<Entry>::iterator, KeyHash> _map;
};

/// Process-wide cache of "last clip covered" markers for PDV-local reuse policy tests
/// and lightweight integration (stores only whether a key had a covering load).
/// Integration in LoadImageMM uses ClipMeta only (no payload ownership of Elise objects).
struct ClipMeta
{
    ClipRect clip;
    int dezoom = 0;
    size_t bytes = 0;
};

inline WindowLru<ClipMeta> & GlobalMetaCache()
{
    static WindowLru<ClipMeta> s(CapBytes(), nullptr);
    return s;
}

/// Record a successful load window (meta only — actual cLoadedImage stays on PDV).
inline void RecordLoad(const std::string & image, int dezoom, const ClipRect & clip, size_t bytesEst)
{
    if (!Enabled())
        return;
    Key k{image, dezoom};
    ClipMeta m;
    m.clip = clip;
    m.dezoom = dezoom;
    m.bytes = bytesEst ? bytesEst : (size_t)std::max<int64_t>(1, clip.area());
    GlobalMetaCache().put(k, clip, m, m.bytes);
}

/// True if a prior load for this image@dezoom covers req (diag / gate helper).
inline bool WouldHit(const std::string & image, int dezoom, const ClipRect & req)
{
    if (!Enabled())
        return false;
    Key k{image, dezoom};
    ClipMeta * p = GlobalMetaCache().find_cover(k, req);
    return p != nullptr;
}

/// PDV-local reuse: same dezoom and **exact** clip match.
/// Cover-only reuse is unsafe with cLoadedImage (offsets bound to alloc clip).
inline bool LocalReuseOk(int prevDezoom, const ClipRect & prevClip,
                         int reqDezoom, const ClipRect & reqClip)
{
    if (prevDezoom != reqDezoom)
        return false;
    return prevClip.exact(reqClip);
}

inline void ClearEtape()
{
    GlobalMetaCache().clear_etape();
}

inline void LogIfEnabled()
{
    if (!Enabled())
        return;
    GlobalMetaCache().log_diag();
}

} // namespace gpgpu_img_lru

#endif // GPGPU_IMAGE_LRU_H
