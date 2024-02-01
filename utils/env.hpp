// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdlib>
#include <cstring>
#include <optional>
#include <vector>

namespace tt
{
template <typename T>
inline T env_cast(char const*)
{
    static_assert(sizeof(T) == 0, "No viable specialization for typename T");
}

template <>
inline const char* env_cast<const char*>(char const* value)
{
    return value;
}

template <>
inline std::string env_cast<std::string>(char const* value)
{
    return value;
}

template <>
inline std::size_t env_cast<std::size_t>(char const* value)
{
    return std::atoll(value);
}

template <>
inline int env_cast<int>(char const* value)
{
    return std::atoi(value);
}

template <>
inline float env_cast<float>(char const* value)
{
    return std::stof(std::string(value));
}

template <>
inline double env_cast<double>(char const* value)
{
    return std::stod(std::string(value));
}

template <>
inline bool env_cast<bool>(char const* value)
{
    return env_cast<int>(value) != 0;
}

template <typename T>
inline std::optional<T> env_as_optional(char const* env_var)
{
    char const* value = std::getenv(env_var);
    return value ? env_cast<T>(value) : std::optional<T>{};
}

template <typename T>
inline T env_as(char const* env_var, T default_value = T{})
{
    char const* value = std::getenv(env_var);
    return value ? env_cast<T>(value) : default_value;
}

template <typename T>
inline std::vector<T> env_as_vector(char const* env_var, std::string const& delimiter = ",")
{
    char const* value = std::getenv(env_var);
    if (not value)
        return {};

    std::vector<T> v;
    std::string s = value;
    char const* sub = s.c_str();
    for (std::string::size_type pos = s.find(delimiter); pos != std::string::npos; pos = s.find(delimiter))
    {
        s[pos] = '\0';
        v.push_back(env_cast<T>(sub));
        sub = s.c_str() + pos + delimiter.size();
    }
    v.push_back(env_cast<T>(sub));
    return v;
}
}  // namespace tt
