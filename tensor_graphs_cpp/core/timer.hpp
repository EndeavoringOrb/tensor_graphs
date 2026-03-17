#pragma once
#include <chrono>
#include <iostream>

#define DEBUG_TIMING

#ifdef DEBUG_TIMING

struct ProgressTimer
{
    using clock = std::chrono::steady_clock;

    clock::time_point start;
    size_t total;
    size_t current = 0;
    const char *label;

    ProgressTimer(size_t total_, const char *label_ = "")
        : start(clock::now()), total(total_), label(label_) {}

    inline void tick(size_t increment = 1)
    {
        current += increment;

        auto now = clock::now();
        double elapsed = std::chrono::duration<double>(now - start).count();
        double rate = current / (elapsed > 0.0 ? elapsed : 1e-9);
        double eta = (total > current) ? (total - current) / rate : 0.0;

        std::cout << label
                  << current << "/" << total
                  << " ETA: " << eta << "s\r" << std::flush;
    }

    ~ProgressTimer()
    {
        auto end = clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();
        std::cout << "\n"
                  << label << "done in " << elapsed << "s\n";
    }
};

#else

// Fully compiled out (no overhead, no branches)
struct ProgressTimer
{
    ProgressTimer(size_t, const char * = "") {}
    inline void tick(size_t = 1) {}
};

#endif