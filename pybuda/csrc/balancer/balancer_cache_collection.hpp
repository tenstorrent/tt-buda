// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "balancer/types.hpp"

namespace tt::balancer
{

// typename std::enable_if<std::is_same<T, std::unordered_map<>>::value, int>::type

// Generic template to check whether a type is a map in stdlib
//
template <typename T>
struct is_std_map
{
    static constexpr bool value = false;
};

// Specialization for std::unordered_map
//
template <typename K, typename V>
struct is_std_map<std::unordered_map<K, V>>
{
    static constexpr bool value = true;
};

// Specialization for std::map
//
template <typename K, typename V>
struct is_std_map<std::map<K, V>>
{
    static constexpr bool value = true;
};

// Returns (very) approximate size of map in bytes
//
template <typename T>
typename std::enable_if<is_std_map<T>::value, int>::type get_map_size_bytes_approx(T map)
{
    // sizeof all keys and values + sizeof map overhead
    //
    return (sizeof(typename T::key_type) + sizeof(typename T::mapped_type)) * map.size() + sizeof map;
}

// Container for various caches used throughout the balancing process.
// Currently a very simple structure with a lot of space for improvements, to be done as needed...
//
struct BalancerCacheCollection
{
    std::unordered_map<Pipe, int> pipe_to_kb_len_cache;                    // Cache Pipe object to kernel broadcast len
    std::unordered_map<Pipe, ResourceUsage> pipe_to_resource_usage_cache;  // Cache Pipe object to ResourceUsage

    BalancerCacheCollection() { log_debug(tt::LogBalancer, "BalancerCacheCollection: Cache collection initialized"); }

    ~BalancerCacheCollection()
    {
        log_debug(tt::LogBalancer, "BalancerCacheCollection: Cache collection destroyed");

        // Stats
        //
        log_debug(tt::LogBalancer, "BalancerCacheCollection: Cache collection stats:");

        log_debug(tt::LogBalancer, "  pipe_to_kb_len_cache size (elems): {}", pipe_to_kb_len_cache.size());
        log_debug(
            tt::LogBalancer,
            "  pipe_to_kb_len_cache size approx (bytes): {} b",
            get_map_size_bytes_approx(pipe_to_kb_len_cache));

        log_debug(
            tt::LogBalancer, "  pipe_to_resource_usage_cache size (elems): {}", pipe_to_resource_usage_cache.size());
        log_debug(
            tt::LogBalancer,
            "  pipe_to_resource_usage_cache size approx (bytes): {} b",
            get_map_size_bytes_approx(pipe_to_resource_usage_cache));
    }
};

}  // namespace tt::balancer
