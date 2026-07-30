#ifndef __GPGPU_MULTITHREADING_CPU_H__
#define __GPGPU_MULTITHREADING_CPU_H__

#include <stdio.h>
#include <unistd.h> // RUNPOD_GPGPU_DIAG

#include "GpGpu/GpGpu_Data.h"
#include "GpGpu/GpGpu_Diag.h"

#ifdef CPP11THREAD_NOBOOSTTHREAD
#define CPP11_THREAD
#endif

#ifdef CPP11_THREAD
    #ifdef NOCUDA_X11
        #include <chrono>
        #include <thread>
        #include <mutex>
        #include <condition_variable>
        #include <cstdlib>
    #endif
#else
#include <boost/thread/thread.hpp>
#include <boost/thread/condition_variable.hpp>
#include <boost/progress.hpp>
#include <boost/timer.hpp>
#include <cstdlib>
#endif


template< class T >
///
/// \brief The CSimpleJobCpuGpu class
///
class CSimpleJobCpuGpu
{
public:

    ///
    /// \brief CSimpleJobCpuGpu
    /// \param useMultiThreading
    ///
    CSimpleJobCpuGpu(bool useMultiThreading = true);
    ~CSimpleJobCpuGpu();

    ///
    /// \brief SetCompute indique au thread Gpu s'il doit traiter les donn�es
    /// \param toBeComputed
    ///
    void            SetCompute(T toBeComputed);
    ///
    /// \brief GetCompute : savoir si le Gpu doit traiter des donn�es
    /// \return
    ///
    T               GetCompute();

    ///
    /// \brief SetDataToCopy
    /// \param toBeCopy
    ///
    void            SetDataToCopy(T toBeCopy);

    ///
    /// \brief GetDataToCopy
    /// \return
    ///
    T               GetDataToCopy();

    ///
    /// \brief SetPreComp
    /// \param canBePreCompute
    ///
    void            SetPreComp(bool canBePreCompute);

    ///
    /// \brief GetPreComp
    /// \return
    ///
    bool            GetPreComp();

	///
	/// \brief UseMultiThreading
	/// \return La valeur de l'option sur l'utilisation du parall�lisme CPU
	///
    bool            UseMultiThreading();

	///
	/// \brief GetIdBuf
	/// \return L'identifiant du buffer courrant
	///
    bool            GetIdBuf();
	///
	/// \brief SwitchIdBuffer
	/// Changer de buffer
    void            SwitchIdBuffer();
	///
	/// \brief ResetIdBuffer
	/// R�initialise l'identifiant du buffer
    void            ResetIdBuffer();

	///
	/// \brief freezeCompute
	/// Stoppe le calcul
    virtual void    freezeCompute() = 0;


	///
	/// \brief SetProgress
	/// \param expected_count
	/// D�finir la progression
    void            SetProgress(unsigned long expected_count);

	///
	/// \brief IncProgress Incr�menter la progression
	/// \param inc
	///
    void            IncProgress(uint inc = 1);

	///
	/// \brief simpleJob
	/// Lance le processus de gpu
    void            simpleJob();

private:


    void            simpleCompute();

    virtual void    simpleWork()    = 0;

    bool            _useMultiThreading;

#ifdef CPP11_THREAD
    #ifdef NOCUDA_X11
    std::mutex    _mutexCompu;
    std::mutex    _mutexCopy;
	std::mutex    _mutexPreCompute;
    std::mutex    _mutexWork;          // PR-D: serialize simpleWork enqueue
    std::condition_variable _cvCompu;  // PR-E
    std::condition_variable _cvCopy;   // PR-E
    #endif
#else
    boost::mutex    _mutexCompu;
    boost::mutex    _mutexCopy;
    boost::mutex    _mutexPreCompute;
    boost::mutex    _mutexWork;
    boost::condition_variable _cvCompu;
    boost::condition_variable _cvCopy;
#endif

    T               _compute;
    T               _copy;
    bool            _precompute;

    bool            _idBufferHostIn;
#ifndef CPP11_THREAD
    boost::progress_display *_show_progress;
#endif
    bool            _show_progress_console;

};

template< class T >
CSimpleJobCpuGpu<T>::CSimpleJobCpuGpu(bool useMultiThreading):
    _useMultiThreading(useMultiThreading),
    _compute(false),
    _copy(false),
    _precompute(false),
    _idBufferHostIn(false),
    #ifndef CPP11_THREAD
    _show_progress(NULL),
    #endif
    _show_progress_console(false)
{}

template< class T >
CSimpleJobCpuGpu<T>::~CSimpleJobCpuGpu()
{
#ifndef CPP11_THREAD
    if(_show_progress)
        delete _show_progress;
#endif
}

template< class T >
void CSimpleJobCpuGpu<T>::SetCompute(T toBeComputed)
{
#ifdef CPP11_THREAD
    #ifdef NOCUDA_X11
    {
        std::lock_guard<std::mutex> guard(_mutexCompu);
        _compute = toBeComputed;
    }
    _cvCompu.notify_all(); // PR-E
    #else
    _compute = toBeComputed;
    #endif
#else
    {
        boost::lock_guard<boost::mutex> guard(_mutexCompu);
        _compute = toBeComputed;
    }
    _cvCompu.notify_all();
#endif
}

template< class T >
T CSimpleJobCpuGpu<T>::GetCompute()
{
#ifdef CPP11_THREAD
    #ifdef NOCUDA_X11
    std::lock_guard<std::mutex> guard(_mutexCompu);
    #endif
#else
    boost::lock_guard<boost::mutex> guard(_mutexCompu);
#endif
    return _compute;
}

template< class T >
void CSimpleJobCpuGpu<T>::SetDataToCopy(T toBeCopy)
{
#ifdef CPP11_THREAD
    #ifdef NOCUDA_X11
    {
        std::lock_guard<std::mutex> guard(_mutexCopy);
        _copy = toBeCopy;
    }
    _cvCopy.notify_all(); // PR-E
    #else
    _copy = toBeCopy;
    #endif
#else
    {
        boost::lock_guard<boost::mutex> guard(_mutexCopy);
        _copy = toBeCopy;
    }
    _cvCopy.notify_all();
#endif
}

template< class T >
T CSimpleJobCpuGpu<T>::GetDataToCopy()
{
#ifdef CPP11_THREAD
    #ifdef NOCUDA_X11
    std::lock_guard<std::mutex> guard(_mutexCopy);
    #endif
#else
    boost::lock_guard<boost::mutex> guard(_mutexCopy);
#endif
    return _copy;
}

template< class T >
void CSimpleJobCpuGpu<T>::SetPreComp(bool canBePreCompute)
{
#ifdef CPP11_THREAD
    #ifdef NOCUDA_X11
    std::lock_guard<std::mutex> guard(_mutexPreCompute);
    #endif
#else
    boost::lock_guard<boost::mutex> guard(_mutexPreCompute);
#endif

    _precompute = canBePreCompute;
}

template< class T >
bool CSimpleJobCpuGpu<T>::GetPreComp()
{
#ifdef CPP11_THREAD
    #ifdef NOCUDA_X11
    std::lock_guard<std::mutex> guard(_mutexPreCompute);
    #endif
#else
    boost::lock_guard<boost::mutex> guard(_mutexPreCompute);
#endif
    return _precompute;
}

template< class T >
bool CSimpleJobCpuGpu<T>::UseMultiThreading()
{
    return _useMultiThreading;
}

template< class T >
bool CSimpleJobCpuGpu<T>::GetIdBuf()
{
    return _idBufferHostIn;
}

template< class T >
void CSimpleJobCpuGpu<T>::SwitchIdBuffer()
{
    _idBufferHostIn = !_idBufferHostIn;
}

template< class T >
void CSimpleJobCpuGpu<T>::ResetIdBuffer()
{
    _idBufferHostIn = false;
}

template< class T >
void CSimpleJobCpuGpu<T>::SetProgress(unsigned long expected_count)
{
#ifndef CPP11_THREAD
    if(_show_progress_console)
    {
        if(_show_progress == NULL)
            _show_progress = new boost::progress_display(expected_count);
        else
            _show_progress->restart(expected_count);
    }
#endif
}

template< class T >
void CSimpleJobCpuGpu<T>::IncProgress(uint inc)
{
#ifndef CPP11_THREAD
    if(_show_progress_console)
        (*_show_progress) += inc;
#endif
}

template< class T >
void CSimpleJobCpuGpu<T>::simpleCompute()
{
    // PR-E: primary wait path = condition_variable; optional MICMAC_GPU_POLL_US fallback.
    auto _gpgpu_now = []() -> double {
#if defined(CPP11_THREAD) && defined(NOCUDA_X11)
        using clock = std::chrono::steady_clock;
        return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
#else
        return 0.0;
#endif
    };
    auto _poll_us = []() -> int {
        // 0 (default) => CV wait; >0 => debug poll period in microseconds
        const char * e = std::getenv("MICMAC_GPU_POLL_US");
        if (!e || !e[0])
            return 0;
        int v = std::atoi(e);
        return v > 0 ? v : 0;
    };
    auto _gpgpu_sleep_us = [](int us) {
#if defined(CPP11_THREAD) && defined(NOCUDA_X11)
        std::this_thread::sleep_for(std::chrono::microseconds(us));
#elif !defined(CPP11_THREAD)
        boost::this_thread::sleep(boost::posix_time::microsec(us));
#else
        (void)us; // nvcc host path without threads — no-op
#endif
    };
    GPGPU_DIAG_FULL("[GPGPU][RUNPOD_GPGPU_DIAG] simpleCompute ENTER pid=%d\n", (int)getpid());

    double t0 = _gpgpu_now();
    double t_last = t0;
    unsigned long wait_iters = 0;
    const int pollUs = _poll_us();
    while(!GetCompute())
    {
        wait_iters++;
        if (pollUs > 0)
        {
            _gpgpu_sleep_us(pollUs);
        }
        else
        {
#if defined(CPP11_THREAD) && defined(NOCUDA_X11)
            std::unique_lock<std::mutex> lk(_mutexCompu);
            _cvCompu.wait_for(lk, std::chrono::milliseconds(50), [this]{ return (bool)_compute; });
#elif !defined(CPP11_THREAD)
            boost::unique_lock<boost::mutex> lk(_mutexCompu);
            _cvCompu.timed_wait(lk, boost::posix_time::milliseconds(50));
#else
            // CPP11 without NOCUDA_X11 (e.g. nvcc parse): short sleep poll
            _gpgpu_sleep_us(200);
#endif
        }
        double t = _gpgpu_now();
        if (GpgpuDiagFull() && (t - t_last >= 2.0))
        {
            GPGPU_DIAG_FULL(
                "[GPGPU][RUNPOD_GPGPU_DIAG] simpleCompute WAIT_COMPUTE pid=%d elapsed=%.1fs iters=%lu "
                "compute=%d copy=%d pre=%d idBuf=%d poll_us=%d\n",
                (int)getpid(), t - t0, wait_iters,
                (int)GetCompute(), (int)GetDataToCopy(), (int)GetPreComp(), (int)GetIdBuf(),
                pollUs);
            t_last = t;
        }
        if (t - t0 > 600.0 && wait_iters % 5000 == 0)
        {
            GPGPU_DIAG_MIN(
                "[GPGPU][RUNPOD_GPGPU_DIAG] WARNING simpleCompute still WAIT_COMPUTE after %.0fs pid=%d\n",
                t - t0, (int)getpid());
        }
    }
    SetCompute(false);

    GPGPU_DIAG_FULL("[GPGPU][RUNPOD_GPGPU_DIAG] simpleCompute WORK_BEGIN pid=%d wait_compute=%.2fs\n",
            (int)getpid(), _gpgpu_now() - t0);
    double t_work0 = _gpgpu_now();
    // PR-D: serialize enqueue so dual workers do not race GetIdBuf; GPU work can still overlap via streams.
    {
#if defined(CPP11_THREAD) && defined(NOCUDA_X11)
        std::lock_guard<std::mutex> workGuard(_mutexWork);
        simpleWork();
#elif !defined(CPP11_THREAD)
        boost::lock_guard<boost::mutex> workGuard(_mutexWork);
        simpleWork();
#else
        simpleWork();
#endif
    }
    GPGPU_DIAG_FULL("[GPGPU][RUNPOD_GPGPU_DIAG] simpleCompute WORK_END pid=%d work=%.2fs\n",
            (int)getpid(), _gpgpu_now() - t_work0);

    t0 = _gpgpu_now();
    t_last = t0;
    wait_iters = 0;
    while(GetDataToCopy())
    {
        wait_iters++;
        if (pollUs > 0)
        {
            _gpgpu_sleep_us(pollUs);
        }
        else
        {
#if defined(CPP11_THREAD) && defined(NOCUDA_X11)
            std::unique_lock<std::mutex> lk(_mutexCopy);
            _cvCopy.wait_for(lk, std::chrono::milliseconds(50), [this]{ return !_copy; });
#elif !defined(CPP11_THREAD)
            boost::unique_lock<boost::mutex> lk(_mutexCopy);
            _cvCopy.timed_wait(lk, boost::posix_time::milliseconds(50));
#else
            _gpgpu_sleep_us(200);
#endif
        }
        double t = _gpgpu_now();
        if (GpgpuDiagFull() && (t - t_last >= 2.0))
        {
            GPGPU_DIAG_FULL(
                "[GPGPU][RUNPOD_GPGPU_DIAG] simpleCompute WAIT_COPY_CLEAR pid=%d elapsed=%.1fs iters=%lu "
                "compute=%d copy=%d pre=%d idBuf=%d\n",
                (int)getpid(), t - t0, wait_iters,
                (int)GetCompute(), (int)GetDataToCopy(), (int)GetPreComp(), (int)GetIdBuf());
            t_last = t;
        }
        if (t - t0 > 600.0 && wait_iters % 5000 == 0)
        {
            GPGPU_DIAG_MIN(
                "[GPGPU][RUNPOD_GPGPU_DIAG] WARNING simpleCompute WAIT_COPY_CLEAR after %.0fs (possible host stall) pid=%d\n",
                t - t0, (int)getpid());
        }
    }

    SwitchIdBuffer();
    SetDataToCopy(true);
    SetCompute(true);
    GPGPU_DIAG_FULL("[GPGPU][RUNPOD_GPGPU_DIAG] simpleCompute EXIT pid=%d total=%.2fs idBuf=%d\n",
            (int)getpid(), _gpgpu_now() - t_work0, (int)GetIdBuf());
}

template< class T >
void CSimpleJobCpuGpu<T>::simpleJob()
{
    GPGPU_DIAG_FULL("[GPGPU][RUNPOD_GPGPU_DIAG] simpleJob SPAWN pid=%d compute=%d copy=%d pre=%d idBuf=%d\n",
            (int)getpid(),
            (int)GetCompute(), (int)GetDataToCopy(), (int)GetPreComp(), (int)GetIdBuf());
#ifdef CPP11_THREAD
    #ifdef NOCUDA_X11
        std::thread tOpti(&CSimpleJobCpuGpu<T>::simpleCompute,this);
        tOpti.detach();
    #else
        GPGPU_DIAG_ERR("[GPGPU][RUNPOD_GPGPU_DIAG] ERROR simpleJob: CPP11_THREAD without NOCUDA_X11 — NO WORKER THREAD\n");
    #endif
#else
        boost::thread tOpti(&CSimpleJobCpuGpu<T>::simpleCompute,this);
        tOpti.detach();
#endif
}


#endif //__GPGPU_MULTITHREADING_CPU_H__

