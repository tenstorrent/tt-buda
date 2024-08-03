// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "placer/host_memory_allocator.hpp"

#include "backend_api/device_config.hpp"
#include "balancer/balancer.hpp"
#include "graph_lib/node.hpp"
#include "placer/allocator_utils.hpp"
#include "placer/host_memory.hpp"
#include "third_party/budabackend/common/param_lib.hpp"

namespace tt::placer
{

inline static std::uint32_t align_address(std::uint32_t address, std::uint32_t alignment) 
{
    return (address + alignment - 1) & ~static_cast<std::uint32_t>(alignment - 1);
}

inline static std::uint32_t align_host_address(std::uint32_t address, const DeviceConfig& device_config) {
    if (device_config.is_blackhole())
    {
        // On blackhole starting addresses of host queues need to be 64 byte aligned.
        //
        address = align_address(address, 64 /* alignment */);
    }
    else
    {
        // NB: To ensure device->host writes are 64B aligned(PCIE controller w/ 512-bit interface), we need to allocate 
        // addresses that are odd multiples of 32 bytes because we need to include the 32 byte tile header.
        // See BBE#2175 for more details.
        //
        constexpr std::uint32_t alignment = 32;
        address = align_address(address, alignment);

        // Check if the result is an odd multiple; if not, add another alignment.
        //
        if ((address / alignment) % 2 == 0)
        {
            address += alignment;
        }
    }

    return address;
}

std::uint32_t HostMemoryAllocator::get_current_allocation_address() const
{
    return align_host_address(this->current_allocation_address, config.device_config);
}

void HostMemoryAllocator::increment_allocation_address(const std::uint32_t size)
{
    this->current_allocation_address = align_host_address(this->get_current_allocation_address() + size, config.device_config);
}

std::pair<std::uint32_t, std::uint32_t> HostMemoryAllocator::allocate_memory(const graphlib::Node* node, std::uint32_t queue_size)
{
    std::uint32_t allocated_channel = this->get_current_allocation_channel();
    std::uint32_t allocated_address = this->get_current_allocation_address();

    if (allocated_address + queue_size > this->config.host_memory_regions.at(allocated_channel).get_host_channel_size())
    {
        // Fallback to existing allocation scheme: allocate on next channel until we run out of channels
        if (allocated_channel >= this->config.host_memory_regions.size() - 1)
        {
            log_fatal(tt::LogPlacer, "Host queue {} of address {} + size {} = {} exceeds maximum allocatable address {} on host channel {}",
                node->name(), allocated_address, queue_size, this->current_allocation_address, this->config.host_memory_regions.at(allocated_channel).get_host_channel_size(), allocated_channel);
        }
        allocated_channel++;
        this->current_allocation_channel = allocated_channel;
        this->current_allocation_address =
            this->config.host_memory_regions.at(allocated_channel).get_host_channel_start_addr();
        return allocate_memory(node, queue_size);
    }
    this->increment_allocation_address(queue_size);

    return {allocated_channel, allocated_address};
}

std::vector<QueueHostBufferPlacement> HostMemoryAllocator::allocate_queue(
    const graphlib::Node *node, CoordRange const &queue_grid, std::uint32_t queue_size)
{
    std::vector<QueueHostBufferPlacement> buffer_placement;
    for (std::uint32_t row = queue_grid.start.row; row < queue_grid.end.row; row++)
    {
        for (std::uint32_t col = queue_grid.start.col; col < queue_grid.end.col; col++)
        {
            auto [allocated_channel, allocated_address] = this->allocate_memory(node, queue_size);

            buffer_placement.push_back(QueueHostBufferPlacement{
                .channel = allocated_channel,
                .address = allocated_address,
                .buffer_size = queue_size,
            });
            log_debug(
                tt::LogPlacer,
                "Placing host queue {} of size {}, channel {} address {}",
                node->name(),
                queue_size,
                allocated_channel,
                allocated_address);
        }
    }

    return buffer_placement;
}

}  // namespace tt::placer