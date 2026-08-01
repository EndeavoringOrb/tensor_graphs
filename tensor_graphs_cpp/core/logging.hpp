#pragma once

#include <iomanip>
#include <iostream>
#include <mutex>
#include <source_location>
#include <sstream>
#include <string>

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
        // std::source_location::current() automatically captures call site location
        LogMessage(LogLevel level, std::source_location loc = std::source_location::current()) : level_(level)
        {
            stream_ << "[" << logLevelToString(level) << "] " << loc.file_name() << ":" << loc.line() << " ("
                    << loc.function_name() << ") - ";
        }

        // Destructor flushes the complete line when statement ends at the semicolon
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

        template <typename T>
        LogMessage &operator<<(const T &val)
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

        LogMessage &operator<<(std::basic_ios<char, std::char_traits<char>> &(*pf)(
            std::basic_ios<char, std::char_traits<char>> &))
        {
            stream_ << pf;
            return *this;
        }

    private:
        LogLevel level_;
        std::ostringstream stream_;
    };

} // namespace tg_log

// Macro short-circuits logging at compile-time/run-time if below TG_LOG_LEVEL threshold.
// Uses a 1-pass for-loop to safely avoid dangling-else syntax bugs in if/else blocks.
#define LOG(level)                                                                                    \
    for (bool _tg_log_cond = (LOG_LEVEL_##level >= TG_LOG_LEVEL); _tg_log_cond; _tg_log_cond = false) \
    ::tg_log::LogMessage(::LogLevel::level)