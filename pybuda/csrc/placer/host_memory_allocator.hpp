// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "placer/host_memory.hpp"
#include "placer/placer.hpp"

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

struct HostMemoryPlacerConfig;
class HostMemoryAllocator
{
    const HostMemoryPlacerConfig &config;
    std::uint32_t current_allocation_channel;
    std::uint32_t current_allocation_address;

   public:
    HostMemoryAllocator(const HostMemoryPlacerConfig &config, std::uint32_t current_allocation_address)
    : config(config), current_allocation_channel(0), current_allocation_address(current_allocation_address) {}

    std::uint32_t get_current_allocation_channel() const { return current_allocation_channel; }
    std::uint32_t get_current_allocation_address() const;
    void increment_allocation_address(const std::uint32_t size);

    std::pair<std::uint32_t, std::uint32_t> allocate_memory(const graphlib::Node* node, std::uint32_t queue_size);
    std::vector<QueueHostBufferPlacement> allocate_queue(
        const graphlib::Node *node, CoordRange const &queue_grid, std::uint32_t queue_size);
};
void place_host_queues(
    const HostMemoryPlacerConfig &host_memory_config,
    HostMemoryAllocator &host_memory_allocator,
    const graphlib::Graph *graph,
    PlacerSolution &placer_solution,
    balancer::BalancerSolution &balancer_solution);

}  // namespace placer
}  // namespace tt
