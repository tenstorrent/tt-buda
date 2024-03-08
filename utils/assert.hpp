// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cxxabi.h>
#include <execinfo.h>
#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <sstream>
#include <vector>

#include "fmt/core.h"
#include "utils/env.hpp"

namespace tt {
template <typename A, typename B>
struct OStreamJoin {
    OStreamJoin(A const& a, B const& b, char const* delim = " ") : a(a), b(b), delim(delim) {}
    A const& a;
    B const& b;
    char const* delim;
};

template <typename A, typename B>
std::ostream& operator<<(std::ostream& os, tt::OStreamJoin<A, B> const& join) {
    os << join.a << join.delim << join.b;
    return os;
}
}  // namespace tt

namespace tt::assert {

inline std::string demangle(const char* str) {
    size_t size = 0;
    int status = 0;
    std::string rt(256, '\0');
    if (1 == sscanf(str, "%*[^(]%*[^_]%255[^)+]", &rt[0])) {
        char* v = abi::__cxa_demangle(&rt[0], nullptr, &size, &status);
        if (v) {
            std::string result(v);
            free(v);
            return result;
        }
    }
    return str;
}

/**
 * @brief Get the current call stack
 * @param[out] bt Save Call Stack
 * @param[in] size Maximum number of return layers
 * @param[in] skip Skip the number of layers at the top of the stack
 */
inline std::vector<std::string> backtrace(int size, int skip) {
    std::vector<std::string> bt;
    void** array = (void**)malloc((sizeof(void*) * size));
    size_t s = ::backtrace(array, size);
    char** strings = backtrace_symbols(array, s);
    if (strings == NULL) {
        std::cout << "backtrace_symbols error." << std::endl;
        return bt;
    }
    for (size_t i = skip; i < s; ++i) {
        bt.push_back(demangle(strings[i]));
    }
    free(strings);
    free(array);

    return bt;
}

/**
 * @brief String to get current stack information
 * @param[in] size Maximum number of stacks
 * @param[in] skip Skip the number of layers at the top of the stack
 * @param[in] prefix Output before stack information
 */
inline std::string backtrace_to_string(int size, int skip, const std::string& prefix) {
    std::vector<std::string> bt = backtrace(size, skip);
    std::stringstream ss;
    for (size_t i = 0; i < bt.size(); ++i) {
        ss << prefix << bt[i] << std::endl;
    }
    return ss.str();
}

inline void tt_assert_message(std::ostream&) {}

template <typename T, typename... Ts>
void tt_assert_message(std::ostream& os, T const& t, Ts const&... ts) {
    os << t << std::endl;
    tt_assert_message(os, ts...);
}

template <bool fmt_present, typename... Ts>
void tt_assert(
    char const* file,
    int line,
    char const* assert_type,
    char const* condition_str,
    std::string_view format_str,
    Ts const&... messages)
{
    (void)format_str;  // Fix warning about unused

    std::stringstream trace_message_ss = {};
    trace_message_ss << assert_type << " @ " << file << ":" << line << ": " << condition_str << std::endl;
    if constexpr (fmt_present)
    {
        trace_message_ss << fmt::format(fmt::runtime(format_str), messages...);
    }
    else if constexpr (sizeof...(messages) > 0)
    {
        trace_message_ss << "info:" << std::endl;
        tt_assert_message(trace_message_ss, messages...);
    }
    trace_message_ss << "backtrace:\n";
    trace_message_ss << tt::assert::backtrace_to_string(100, 3, " --- ");
    trace_message_ss << std::flush;
    if (env_as<bool>("TT_ASSERT_ABORT"))
        abort();
    throw std::runtime_error(trace_message_ss.str());
}

}  // namespace tt::assert

#define TT_ASSERT(condition, ...) \
    __builtin_expect(not (condition), 0) ? \
    ::tt::assert::tt_assert<false>(__FILE__, __LINE__, "TT_ASSERT", #condition, std::string_view{}, ##__VA_ARGS__) : void()
#define TT_LOG_ASSERT(condition, f, ...) \
    __builtin_expect(not (condition), 0) ? \
    ::tt::assert::tt_assert<true>(__FILE__, __LINE__, "TT_ASSERT", #condition, f, ##__VA_ARGS__) : void()
#define TT_THROW(...) \
    ::tt::assert::tt_assert<false>(__FILE__, __LINE__, "TT_THROW", "tt::exception", std::string_view{}, ##__VA_ARGS__)

#ifndef DEBUG
// Do nothing in release mode.
#define TT_DBG_ASSERT(condition, ...) ((void)0)
#else
#define TT_DBG_ASSERT(condition, ...) \
    __builtin_expect(not (condition), 0) ? \
    ::tt::assert::tt_assert<false>(__FILE__, __LINE__, "TT_DBG_ASSERT", #condition, std::string_view{}, ##__VA_ARGS__) : void()
#endif
