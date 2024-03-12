// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#if defined(_GLIBCXX_USE_CXX11_ABI) && _GLIBCXX_USE_CXX11_ABI == 0
#undef UTILS_LOGGER_PYTHON_OSTREAM_REDIRECT
#endif

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <type_traits>
#include <utility>
#if defined(UTILS_LOGGER_PYTHON_OSTREAM_REDIRECT) && (UTILS_LOGGER_PYTHON_OSTREAM_REDIRECT == 1)
#include <pybind11/iostream.h>
#endif

#include "fmt/color.h"
#include "fmt/core.h"
#include "fmt/ostream.h"
#include "fmt/ranges.h"
#include "utils/env.hpp"

namespace tt
{
enum class LoggerABI
{
    PreCXX11,
    CXX11,
};

#if defined(_GLIBCXX_USE_CXX11_ABI) && _GLIBCXX_USE_CXX11_ABI == 0
constexpr LoggerABI kLoggerABI = LoggerABI::PreCXX11;
#else
constexpr LoggerABI kLoggerABI = LoggerABI::CXX11;
#endif

#define LOGGER_TYPES   \
    X(Always)          \
    X(Test)            \
    X(Placer)          \
    X(Autograd)        \
    X(Netlist)         \
    X(Balancer)        \
    X(KernelBroadcast) \
    X(Padding)         \
    X(GraphSolver)     \
    X(Reportify)       \
    X(Eval)            \
    X(ConstEval)       \
    X(GraphCompiler)   \
    X(TStream)         \
    X(Scheduler)       \
    X(PatternMatcher)  \
    X(PerfModel)       \
    X(Fuser)           \
    X(Fracture)        \
    X(Profile)         \
    X(TMFusion)        \
    X(TTDevice)        \
    X(TorchDevice)

enum LogType : uint32_t
{
// clang-format off
#define X(a) Log ## a,
    LOGGER_TYPES
#undef X
    LogType_Count,
    // clang-format on
};
static_assert(LogType_Count < 64, "Exceeded number of log types");

#pragma GCC visibility push(hidden)
template <LoggerABI abi>
class Logger
{
   public:
    static constexpr char const* type_names[LogType_Count] = {
    // clang-format off
#define X(a) #a,
      LOGGER_TYPES
#undef X
        // clang-format on
    };

    enum class Level
    {
        Trace = 0,
        Debug = 1,
        Profile = 2,
        Info = 3,
        Warning = 4,
        Error = 5,
        Fatal = 6,

        Count,
    };

    static constexpr char const* level_names[] = {
        "TRACE",
        "DEBUG",
        "PROFILE",
        "INFO",
        "WARNING",
        "ERROR",
        "FATAL",
    };

    static_assert(
        (sizeof(level_names) / sizeof(level_names[0])) == static_cast<std::underlying_type_t<Level>>(Level::Count));

    static constexpr fmt::color level_color[] = {
        fmt::color::cornflower_blue,
        fmt::color::cornflower_blue,
        fmt::color::black,
        fmt::color::orange_red,
        fmt::color::orange,
        fmt::color::red,
        fmt::color::red,
    };

    static_assert(
        (sizeof(level_color) / sizeof(level_color[0])) == static_cast<std::underlying_type_t<Level>>(Level::Count));

    // TODO: we should sink this into some common cpp file, marking inline so maybe it's per lib instead of per TU
    static inline Logger<abi>& get()
    {
        static Logger<abi> logger;
        return logger;
    }

    template <typename... Args>
    void log_level_type(Level level, LogType type, char const* fmt, Args&&... args)
    {
        if (static_cast<std::underlying_type_t<Level>>(level) < static_cast<std::underlying_type_t<Level>>(min_level))
            return;

        if ((1 << type) & mask)
        {
#if defined(UTILS_LOGGER_PYTHON_OSTREAM_REDIRECT) && (UTILS_LOGGER_PYTHON_OSTREAM_REDIRECT == 1)
//            pybind11::scoped_ostream_redirect stream(*fd);
#endif
            std::string timestamp_str = get_current_time();
            fmt::terminal_color timestamp_color = fmt::terminal_color::green;
            timestamp_str = fmt::format(fmt::fg(timestamp_color), "{}", timestamp_str);

            std::string level_str = fmt::format(
                fmt::fg(level_color[static_cast<std::underlying_type_t<Level>>(level)]) | fmt::emphasis::bold,
                "{:8}",
                level_names[static_cast<std::underlying_type_t<Level>>(level)]);

            fmt::terminal_color type_color = fmt::terminal_color::cyan;
            std::string type_str = fmt::format(fmt::fg(type_color), "{:15}", type_names[type]);

            fmt::print(*fd, "{} | {} | {} - ", timestamp_str, level_str, type_str);
            fmt::print(*fd, fmt, std::forward<Args>(args)...);
            *fd << std::endl;
        }
    }

    void flush() { *fd << std::flush; }

    bool trace_enabled() const
    {
        return static_cast<std::underlying_type_t<Level>>(Level::Trace) >=
               static_cast<std::underlying_type_t<Level>>(min_level);
    }

    bool debug_enabled() const
    {
        return static_cast<std::underlying_type_t<Level>>(Level::Debug) >=
               static_cast<std::underlying_type_t<Level>>(min_level);
    }

   private:
    Logger()
    {
        static char const* env = env_as<char const*>("LOGGER_TYPES");
        if (env)
        {
            if (strstr(env, "All"))
            {
                mask = 0xFFFFFFFFFFFFFFFF;
            }
            else
            {
                std::uint32_t mask_index = 0;
                for (char const* type_name : type_names)
                {
                    mask |= (strstr(env, type_name) != nullptr) << mask_index;
                    mask_index++;
                }
            }
        }
        else
        {
            // For now default to all
            mask = 0xFFFFFFFFFFFFFFFF;
        }

        if (auto level_env = env_as_optional<std::string>("LOGGER_LEVEL"))
        {
            std::string level_str = *level_env;
            std::transform(
                level_str.begin(), level_str.end(), level_str.begin(), [](unsigned char c) { return std::toupper(c); });
            std::underlying_type_t<Level> level_index = 0;
            for (char const* level_name : level_names)
            {
                if (level_str == level_name)
                {
                    min_level = static_cast<Level>(level_index);
                }
                level_index++;
            }
        }

#if !defined(UTILS_LOGGER_PYTHON_OSTREAM_REDIRECT) || (UTILS_LOGGER_PYTHON_OSTREAM_REDIRECT == 0)
        static char const* file_env = env_as<char const*>("LOGGER_FILE");
        if (file_env)
        {
            log_file.open(file_env);
            if (log_file.is_open())
            {
                fd = &log_file;
            }
        }
#endif
    }

    // Returns the current timestamp in 'YYYY-MM-DD HH:MM:SS.MMM' format, e.g. 2023-06-26 11:39:38.432
    static std::string get_current_time()
    {
        auto now = std::chrono::system_clock::now();
        auto current_time = std::chrono::system_clock::to_time_t(now);

        auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now_ms.time_since_epoch()) % 1000;

        std::stringstream ss;
        ss << std::put_time(std::localtime(&current_time), "%F %T");
        ss << '.' << std::setfill('0') << std::setw(3) << ms.count();
        return ss.str();
    }

    std::ofstream log_file;
    std::ostream* fd = &std::cout;
    std::uint64_t mask = (1 << LogAlways);
    Level min_level = Level::Info;
};
#pragma GCC visibility pop

#ifdef DEBUG
template <typename... Args>
static void log_debug_(LogType type, char const* fmt, Args&&... args)
{
    Logger<tt::kLoggerABI>::get().log_level_type(Logger<tt::kLoggerABI>::Level::Debug, type, fmt, std::forward<Args>(args)...);
}

template <typename... Args>
static void log_debug_(char const* fmt, Args&&... args)
{
    log_debug_(LogAlways, fmt, std::forward<Args>(args)...);
}

template <typename... Args>
static void log_trace_(LogType type, std::string const& src_info, char const* fmt, Args&&... args)
{
    Logger<tt::kLoggerABI>::get().log_level_type(Logger<tt::kLoggerABI>::Level::Trace, type, fmt, src_info, std::forward<Args>(args)...);
}

#define log_trace(log_type, ...)           \
    if (tt::Logger<tt::kLoggerABI>::get().trace_enabled()) \
    log_trace_(log_type, fmt::format(fmt::fg(fmt::color::green), "{}:{}", __FILE__, __LINE__), "{} - " __VA_ARGS__)

#define log_debug(...)                     \
    if (tt::Logger<tt::kLoggerABI>::get().debug_enabled()) \
    log_debug_(__VA_ARGS__)

#else
template <typename... Args>
static void log_debug(LogType, char const*, Args&&...)
{
}
template <typename... Args>
static void log_debug(char const*, Args&&...)
{
}
template <typename... Args>
static void log_trace(Args&&...)
{
}
#endif

template <typename... Args>
static void log_info(LogType type, char const* fmt, Args&&... args)
{
    Logger<tt::kLoggerABI>::get().log_level_type(Logger<tt::kLoggerABI>::Level::Info, type, fmt, std::forward<Args>(args)...);
}

template <typename... Args>
static void log_info(char const* fmt, Args&&... args)
{
    log_info(LogAlways, fmt, std::forward<Args>(args)...);
}

template <typename... Args>
static void log_warning(LogType type, char const* fmt, Args&&... args)
{
    Logger<tt::kLoggerABI>::get().log_level_type(Logger<tt::kLoggerABI>::Level::Warning, type, fmt, std::forward<Args>(args)...);
}

template <typename... Args>
static void log_warning(char const* fmt, Args&&... args)
{
    log_warning(LogAlways, fmt, std::forward<Args>(args)...);
}

template <typename... Args>
static void log_error(LogType type, char const* fmt, Args&&... args)
{
    Logger<tt::kLoggerABI>::get().log_level_type(Logger<tt::kLoggerABI>::Level::Error, type, fmt, std::forward<Args>(args)...);
}

template <typename... Args>
static void log_error(char const* fmt, Args&&... args)
{
    log_error(LogAlways, fmt, std::forward<Args>(args)...);
}

template <typename... Args>
static void log_fatal_(LogType type, char const* fmt, Args&&... args)
{
    Logger<tt::kLoggerABI>::get().log_level_type(Logger<kLoggerABI>::Level::Fatal, type, fmt, std::forward<Args>(args)...);
    Logger<tt::kLoggerABI>::get().flush();
    throw std::runtime_error(fmt::format(fmt, std::forward<Args>(args)...));
}

template <typename... Args>
static void log_fatal_(char const* fmt, Args&&... args)
{
    log_fatal_(LogAlways, fmt, std::forward<Args>(args)...);
}

#define log_fatal(...)           \
    do                           \
    {                            \
        log_fatal_(__VA_ARGS__); \
        __builtin_unreachable(); \
    } while (false)

#undef LOGGER_TYPES

}  // namespace tt
