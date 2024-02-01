// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <chrono>

#include "fmt/chrono.h"
#include "utils/logger.hpp"

namespace tt
{
#if DEBUG
struct ProfileScope
{
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    char const* tag;
    int line;

    ProfileScope(char const* tag, int line = 0) : start(std::chrono::high_resolution_clock::now()), tag(tag), line(line)
    {
    }

    ~ProfileScope()
    {
        std::chrono::duration<double> elapsed = std::chrono::high_resolution_clock::now() - start;
        std::string info = line ? fmt::format(fmt::fg(fmt::color::green), "{}:{}", tag, line) : tag;
        Logger<kLoggerABI>::get().log_level_type(
            Logger<kLoggerABI>::Level::Profile, LogProfile, "{} - elapsed {}", info, elapsed);
    }
};

#define CONCAT(a, b) CONCAT_INNER(a, b)
#define CONCAT_INNER(a, b) a##b
#define PROFILE_SCOPE() ProfileScope CONCAT(tt_PROFILE_SCOPE_id, __COUNTER__)(__FILE__, __LINE__)
#else
struct ProfileScope
{
};

#define PROFILE_SCOPE() ((void)0)
#endif
}  // namespace tt
