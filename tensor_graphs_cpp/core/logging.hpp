#pragma once

#include <atomic>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>

#include "core/types.hpp"

#ifdef ERROR
#undef ERROR
#endif

// Numeric representations for log levels
#define LOG_LEVEL_DEBUG 0
#define LOG_LEVEL_INFO 1
#define LOG_LEVEL_WARNING 2
#define LOG_LEVEL_ERROR 3
#define LOG_LEVEL_CRITICAL 4
#define LOG_LEVEL_OFF 5

// Default log level if not supplied at compile time via -DTG_LOG_LEVEL
#ifndef TG_LOG_LEVEL
#define TG_LOG_LEVEL LOG_LEVEL_INFO
#endif

enum class LogLevel : int
{
    DEBUG = LOG_LEVEL_DEBUG,
    INFO = LOG_LEVEL_INFO,
    WARNING = LOG_LEVEL_WARNING,
    ERROR = LOG_LEVEL_ERROR,
    CRITICAL = LOG_LEVEL_CRITICAL,
    OFF = LOG_LEVEL_OFF
};

namespace tg_log
{

inline const char *logLevelToString(LogLevel level)
{
    switch (level)
    {
    case LogLevel::DEBUG:
        return "DEBUG";
    case LogLevel::INFO:
        return "INFO";
    case LogLevel::WARNING:
        return "WARNING";
    case LogLevel::ERROR:
        return "ERROR";
    case LogLevel::CRITICAL:
        return "CRITICAL";
    default:
        return "UNKNOWN";
    }
}

class LogMessage
{
  public:
    LogMessage(LogLevel level, SourceLocation loc = SourceLocation::current()) : level_(level)
    {
        stream_ << "[" << logLevelToString(level) << "] " << loc.file_name() << ":" << loc.line() << " - ";
    }

    ~LogMessage()
    {
        stream_ << "\n";
        static std::mutex log_mutex;
        std::lock_guard<std::mutex> lock(log_mutex);
        if (level_ >= LogLevel::ERROR)
        {
            std::cerr << stream_.str() << std::flush;
        }
        else
        {
            std::cout << stream_.str() << std::flush;
        }
    }

    template <typename T> LogMessage &operator<<(const T &val)
    {
        stream_ << val;
        return *this;
    }

    LogMessage &operator<<(std::ostream &(*pf)(std::ostream &))
    {
        stream_ << pf;
        return *this;
    }

    LogMessage &operator<<(std::ios_base &(*pf)(std::ios_base &))
    {
        stream_ << pf;
        return *this;
    }

    LogMessage &operator<<(
        std::basic_ios<char, std::char_traits<char>> &(*pf)(std::basic_ios<char, std::char_traits<char>> &))
    {
        stream_ << pf;
        return *this;
    }

  private:
    LogLevel level_;
    std::ostringstream stream_;
};

// Keep a representative sample of hot-path diagnostics in profile builds.
// The first events show the initial failure modes; periodic events show whether
// the same paths continue to dominate during a long search.
inline bool shouldLogHotPath()
{
    static std::atomic<uint64_t> event_count{0};
    const uint64_t event = event_count.fetch_add(1, std::memory_order_relaxed);
    return event < 100 || event % 1000 == 0;
}

} // namespace tg_log

#define LOG(level)                                                                                                     \
    for (bool _tg_log_cond = (LOG_LEVEL_##level >= TG_LOG_LEVEL); _tg_log_cond; _tg_log_cond = false)                  \
    ::tg_log::LogMessage(::LogLevel::level)

// Per-branch diagnostics can dominate profiled search runs because every log
// flushes the stream. Keep these details in ordinary debug builds, while
// profiling builds rely on aggregate search and propagator counters instead.
#ifdef TG_PROFILE
#define LOG_HOT_PATH(level) for (bool _tg_hot_log_cond = (LOG_LEVEL_##level >= TG_LOG_LEVEL && ::tg_log::shouldLogHotPath()); \
                                _tg_hot_log_cond; _tg_hot_log_cond = false)                                           \
    ::tg_log::LogMessage(::LogLevel::level)
#else
#define LOG_HOT_PATH(level) LOG(level)
#endif
