#pragma once
#include <chrono>
#include <cstdint>
#include <sstream>
#include <string>

#include "core/logging.hpp"

#ifdef ERROR
#undef ERROR
#endif

#define TG_DEBUG_TIMING

#ifdef TG_DEBUG_TIMING

struct ProgressTimer
{
    using clock = std::chrono::steady_clock;

    clock::time_point start;
    clock::time_point last_print;

    uint64_t total;
    uint64_t current = 0;

    double minInterval; // seconds
    std::string label;
    LogLevel logLevel;

    bool has_total;
    bool disable;
    bool disable_tick;

    ProgressTimer(uint64_t total_ = 0, std::string label_ = "", bool disable_ = false, bool disable_tick_ = false,
                  double minInterval_ = 2, LogLevel logLevel_ = LogLevel::DEBUG)
        : start(clock::now()), last_print(start), total(total_), minInterval(minInterval_), label(label_),
          logLevel(logLevel_), has_total(total_ > 0), disable(disable_), disable_tick(disable_tick_)
    {
        if (label.size() > 0)
        {
            label += " ";
        }
    }

    void reset()
    {
        start = clock::now();
        last_print = start;
        current = 0;
        has_total = total > 0;
    }

    // LOG takes a compile-time level token, so dispatch the runtime level through a switch
    void logMessage(const std::string &msg) const
    {
        switch (logLevel)
        {
        case LogLevel::DEBUG:
            LOG(DEBUG) << msg;
            break;
        case LogLevel::INFO:
            LOG(INFO) << msg;
            break;
        case LogLevel::WARNING:
            LOG(WARNING) << msg;
            break;
        case LogLevel::ERROR:
            LOG(ERROR) << msg;
            break;
        case LogLevel::CRITICAL:
            LOG(CRITICAL) << msg;
            break;
        case LogLevel::OFF:
            break;
        }
    }

    inline void tick(uint64_t increment = 1)
    {
        if (disable || disable_tick)
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

        std::ostringstream oss;
        oss << label;

        if (has_total)
        {
            double eta = (total > current) ? (total - current) / rate : 0.0;

            oss << current << "/" << total << " ETA: " << eta << "s";
        }
        else
        {
            oss << current << " (" << rate << " it/s, " << elapsed << "s)";
        }

        logMessage(oss.str());
    }

    // returns elapsed time in seconds
    double getElapsed()
    {
        auto end = clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();
        return elapsed;
    }

    ~ProgressTimer()
    {
        if (disable)
            return;
        auto end = clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();

        std::ostringstream oss;
        oss << label;

        if (has_total)
        {
            oss << "done " << current << "/" << total;
        }
        else
        {
            oss << "done " << current;
        }

        oss << " in " << elapsed << "s";

        logMessage(oss.str());
    }
};

#else

struct ProgressTimer
{
    ProgressTimer(uint64_t, const char * = "", double = 0.0)
    {
    }
    inline void tick(uint64_t = 1)
    {
    }
};

#endif
