#pragma once
#include <chrono>
#include <iostream>

#define DEBUG_TIMING

#ifdef DEBUG_TIMING

struct ProgressTimer
{
    using clock = std::chrono::steady_clock;

    clock::time_point start;
    clock::time_point last_print;

    size_t total;
    size_t current = 0;

    double minInterval; // seconds
    const char *label;

    bool has_total;
    bool disable;

    ProgressTimer(size_t total_, const char *label_ = "", bool disable_ = false, double minInterval_ = 2)
        : start(clock::now()),
          last_print(start),
          total(total_),
          minInterval(minInterval_),
          label(label_),
          has_total(total_ > 0),
          disable(disable_) {}

    inline void tick(size_t increment = 1)
    {
        if (disable)
            return;
        current += increment;

        auto now = clock::now();
        double since_last = std::chrono::duration<double>(now - last_print).count();

        // If total is known: always print on completion, otherwise throttle
        if (has_total)
        {
            if (since_last < minInterval && current < total)
                return;
        }
        else
        {
            if (since_last < minInterval)
                return;
        }

        last_print = now;

        double elapsed = std::chrono::duration<double>(now - start).count();
        double rate = current / (elapsed > 0.0 ? elapsed : 1e-9);

        std::cout << label;

        if (has_total)
        {
            double eta = (total > current) ? (total - current) / rate : 0.0;

            std::cout << current << "/" << total
                      << " ETA: " << eta << "s";
        }
        else
        {
            std::cout << current
                      << " (" << rate << " it/s, "
                      << elapsed << "s)";
        }

        std::cout << "\r" << std::flush;
    }

    ~ProgressTimer()
    {
        if (disable)
            return;
        auto end = clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();

        std::cout << "\n"
                  << label;

        if (has_total)
        {
            std::cout << "done " << current << "/" << total;
        }
        else
        {
            std::cout << "done " << current;
        }

        std::cout << " in " << elapsed << "s\n";
    }
};

#else

struct ProgressTimer
{
    ProgressTimer(size_t, const char * = "", double = 0.0) {}
    inline void tick(size_t = 1) {}
};

#endif