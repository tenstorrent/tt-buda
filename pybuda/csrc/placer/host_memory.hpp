// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <vector>

#include "placer/host_memory_allocator.hpp"

namespace tt
{
struct DeviceConfig;
namespace graphlib
{
class Graph;
class Node;
}  // namespace graphlib
namespace balancer
{
struct BalancerSolution;
}

namespace placer
{

class HostChannelMemoryRegion
{
    std::uint32_t host_channel_id;
    std::uint32_t host_channel_start_addr;
    std::uint32_t host_channel_size;

   public:
    HostChannelMemoryRegion(
        std::uint32_t host_channel_id, std::uint32_t host_channel_start_addr, std::uint32_t host_channel_size) :
        host_channel_id(host_channel_id),
        host_channel_start_addr(host_channel_start_addr),
        host_channel_size(host_channel_size)
    {
    }
    std::uint32_t get_host_channel_id() const { return host_channel_id; }
    std::uint32_t get_host_channel_start_addr() const { return host_channel_start_addr; }
    std::uint32_t get_host_channel_size() const { return host_channel_size; }
};

// HostMemory is system memory that is memory mapped onto the device.
// Host memory is divided into host channels, which are contiguous regions of memory.
struct HostMemoryPlacerConfig
{
    const DeviceConfig& device_config;
    std::vector<HostChannelMemoryRegion> host_memory_regions;
    bool input_queues_on_host;
    bool output_queues_on_host;

   public:
    HostMemoryPlacerConfig(const DeviceConfig& device_config, bool input_queues_on_host, bool output_queues_on_host);
    bool place_input_queues_on_host() const;
    bool place_output_queues_on_host() const;
    std::size_t get_num_host_channels() const { return host_memory_regions.size(); }
};

}  // namespace placer
}  // namespace tt
