// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "placer/dram_allocator.hpp"

#include <unordered_map>

#include "best_fit_allocator.hpp"
#include "placer/dram.hpp"
#include "reportify/paths.hpp"
#include "third_party/budabackend/common/param_lib.hpp"
#include "placer/exceptions.hpp"

// from backend
namespace tt::backend
{
extern uint32_t get_next_aligned_address(const uint32_t address);
}

namespace tt::placer
{
class RoundRobinPicker : public ChannelPicker
{
    const bool skip_ch0 = env_as<bool>("PYBUDA_DISABLE_DRAM0");
    std::uint32_t next_channel = skip_ch0 ? 1 : 0;

   public:
    virtual std::uint32_t pick_channel(
        const QueueDRAMPlacementParameters & /* parameters */,
        Coord /*c*/,
        const std::vector<std::unique_ptr<ChannelAllocator>> &channel_allocators) override
    {
        std::uint32_t selected_channel = next_channel;
        TT_ASSERT(selected_channel < channel_allocators.size());

        next_channel++;
        if (next_channel >= channel_allocators.size())
            next_channel = skip_ch0 ? 1 : 0;

        return selected_channel;
    }
    // we have to reset the next_channel
    virtual void reset() override { next_channel = skip_ch0 ? 1 : 0; }
};

class RoundRobinFlipFlopPicker : public ChannelPicker
{
    const bool skip_ch0 = env_as<bool>("PYBUDA_DISABLE_DRAM0");
    struct group
    {
        std::uint32_t next_channel;
        std::uint32_t min, max;
    };
    std::vector<group> groups;

   public:
    virtual std::uint32_t pick_channel(
        const QueueDRAMPlacementParameters &parameters,
        Coord /*c*/,
        const std::vector<std::unique_ptr<ChannelAllocator>> &channel_allocators) override
    {
        // Round robin between two groups of channels, based on producer epoch mod 2
        // If 'is_input' is set, then full set of channels is used
        std::uint32_t group = parameters.is_input ? 2 : parameters.producer_epoch % 2;
        if (groups.size() == 0)
        {
            for (std::uint32_t i = 0; i < 3; i++)
            {
                RoundRobinFlipFlopPicker::group g;
                std::uint32_t mid = channel_allocators.size() / 2;
                g.min = ((i == 0) || (i == 2)) ? (skip_ch0 ? 1 : 0) : mid;
                g.max = (i == 0) ? mid : channel_allocators.size();
                g.next_channel = g.min;
                groups.push_back(g);
            }
        }

        // Pick channel from the appropriate group
        std::uint32_t selected_channel = groups[group].next_channel;
        TT_ASSERT(selected_channel < channel_allocators.size());

        // Increment next channel
        groups[group].next_channel++;
        if (groups[group].next_channel >= groups[group].max)
            groups[group].next_channel = groups[group].min;

        return selected_channel;
    }
    // we have to reset the groups
    virtual void reset() override { groups.clear(); }
};

class GreatestCapacityPicker : public ChannelPicker
{
   public:
    virtual std::uint32_t pick_channel(
        const QueueDRAMPlacementParameters & /*parameters*/,
        Coord /*c*/,
        const std::vector<std::unique_ptr<ChannelAllocator>> &channel_allocators) override
    {
        std::uint32_t largest_capacity = 0;
        std::uint32_t selected_channel = 0;
        for (std::uint32_t i = 0; i < channel_allocators.size(); i++)
        {
            auto capacity = channel_allocators[i]->get_capacity();
            if (capacity > largest_capacity)
            {
                largest_capacity = capacity;
                selected_channel = i;
            }
        }
        TT_ASSERT(selected_channel < channel_allocators.size());

        return selected_channel;
    }
    // nothing to reset
    virtual void reset() override {}
};

class ClosestPicker : public ChannelPicker
{
    const Node *current_node = nullptr;
    std::unordered_map<std::uint32_t, std::unordered_map<std::uint32_t, std::uint32_t>>
        solution;  // coord to channel choice for the node
    std::unordered_map<std::uint32_t, std::unordered_set<std::uint32_t>>
        unused_channels_in_epoch;  // per epoch, keep track of channels that are not used, in order to avoid splitting
                                   // DRAM bw
    // Prologue has a separate pool of channels because it doesn't run at the same time as the epoch
    std::unordered_map<std::uint32_t, std::unordered_set<std::uint32_t>>
        unused_channels_in_epoch_prologue;  // per epoch, keep track of channels that are not used in prologue
    // Check if channel is unused, and modify it if an alias is available
    static bool is_channel_unused(
        std::uint32_t &channel, bool is_grayskull, const std::unordered_set<std::uint32_t> &unused_channels);

   public:
    virtual std::uint32_t pick_channel(
        const QueueDRAMPlacementParameters &parameters,
        Coord /*c*/,
        const std::vector<std::unique_ptr<ChannelAllocator>> &channel_allocators) override;
    virtual void reset() override;
};

DramAllocator::DramAllocator(
    const DramPlacerConfig &dram_config,
    const std::string &graph_name,
    std::uint32_t chip_id,
    std::vector<Blocks> &allocated_blocks,
    DRAMPlacementAlgorithm placement_algorithm,
    AllocationAlgorithm allocator_algorithm) :
    dram_config(dram_config), graph_name(graph_name), chip_id(chip_id)
{
    dram_logger = std::make_unique<DramLogger>();

    switch (placement_algorithm)
    {
        case ROUND_ROBIN: channel_picker = std::make_unique<RoundRobinPicker>(); break;
        case ROUND_ROBIN_FLIP_FLOP: channel_picker = std::make_unique<RoundRobinFlipFlopPicker>(); break;
        case GREATEST_CAPACITY: channel_picker = std::make_unique<GreatestCapacityPicker>(); break;
        case CLOSEST: channel_picker = std::make_unique<ClosestPicker>(); break;
        default: TT_THROW("Unknown placement algorithm");
    }

    std::unordered_set<std::uint32_t> allocated_channel_blocks;
    if (allocated_blocks.size() == 0)
    {
        for (std::size_t i = 0; i < dram_config.dram_config.size(); i++)
        {
            // Only once per channel, skip sub-channels
            if (allocated_channel_blocks.count(dram_config.dram_config[i].channel))
                continue;

            allocated_channel_blocks.insert(dram_config.dram_config[i].channel);

            allocated_blocks.push_back(Blocks());
            if (dram_config.device_config.is_wormhole_b0())
            {
                allocated_blocks.push_back(Blocks());
            }
        }
        allocated_blocks.push_back(Blocks());
    }

    std::unordered_set<std::uint32_t> allocated_channels;
    switch (allocator_algorithm)
    {
        case BEST_FIT: {
            std::size_t p2p_offset;
            std::size_t p2p_size;

            if (chip_id == 0)
            {
                // if chip is first in a set of chips (chip_id == 0), then, it communicates with host, and we use
                // host_mmio offset and range. this memory is allocated for communication with host.
                p2p_offset = dram_config.host_mmio_range_offset;
                p2p_size = dram_config.host_mmio_range_size;
            }
            else
            {
                // if chip is not first in a set of chips (chip_id != 0), then, it communicates with chip, and we use
                // p2p offset and range. this memory is allocated for communication with other chip.
                p2p_offset = dram_config.p2p_offset;
                p2p_size = dram_config.p2p_size;
            }

            // first channel contains memory allocated for communication with host or other device, so first channel
            // allocator has smaller space than other chanels. It ends where p2p region starts, so, its last available
            // address is p2p_offset - 1.
            channel_allocators.push_back(std::make_unique<BestFitAllocator>(
                dram_config.dram_config[0].initial_dram_offset, p2p_offset - 1, allocated_blocks[0]));

            std::size_t limit_top_address = 0;
            if (dram_config.device_config.is_blackhole()) 
            {
                // The top 16MB (0xFF00_0000 - 0xFFFF_FFFF) of DRAM are not accessible through the NOC on blackhole.
                //
                limit_top_address = 16 * 1024 * 1024;
            }

            if (dram_config.device_config.is_wormhole_b0())
            {
                channel_allocators.push_back(std::make_unique<BestFitAllocator>(
                    std::max(p2p_offset + p2p_size, dram_config.dram_config[0].initial_dram_offset),
                    dram_config.dram_config[0].channel_size - 1 - limit_top_address,
                    allocated_blocks[0]));
            }
            allocated_channels.insert(0);  // 0 is done

            for (std::size_t i = 0; i < dram_config.dram_config.size(); i++)
            {
                // Don't create extra allocators for subchannels
                if (allocated_channels.count(dram_config.dram_config[i].channel) > 0)
                    continue;

                allocated_channels.insert(dram_config.dram_config[i].channel);

                // channels from second to last don not contain memory space for communication with other host/device.
                // They use all space from initial_dram_offset to channel_size - 1. in wormhole, however, we split that
                // memory in two parts, since wormhole has separate bandwidths for two halves of dram.
                if (dram_config.device_config.is_wormhole_b0())
                {
                    // in wormhole we split one channel to two 1GB channels because they have separate bandwidths.
                    channel_allocators.push_back(std::make_unique<BestFitAllocator>(
                        dram_config.dram_config[i].initial_dram_offset,
                        dram_config.dram_config[i].channel_size / 2 - 1,
                        allocated_blocks[2 * dram_config.dram_config[i].channel]));
                    channel_allocators.push_back(std::make_unique<BestFitAllocator>(
                        std::max(dram_config.dram_config[i].initial_dram_offset, dram_config.dram_config[i].channel_size / 2),
                        dram_config.dram_config[i].channel_size - 1 - limit_top_address,
                        allocated_blocks[(2 * dram_config.dram_config[i].channel) + 1]));
                }
                else
                {
                    channel_allocators.push_back(std::make_unique<BestFitAllocator>(
                        dram_config.dram_config[i].initial_dram_offset,
                        dram_config.dram_config[i].channel_size - 1,
                        allocated_blocks[dram_config.dram_config[i].channel]));
                }
            }
            p2p_allocator =
                std::make_unique<BestFitAllocator>(p2p_offset, p2p_offset + p2p_size - 1, allocated_blocks.back());
            break;
        }
        default: TT_THROW("Unknown placement algorithm");
    }
}

std::vector<Blocks> DramAllocator::get_blocks()
{
    std::vector<Blocks> blocks;
    for (const auto &allocator : channel_allocators) blocks.push_back(allocator->get_blocks());
    blocks.push_back(p2p_allocator->get_blocks());
    return blocks;
}

// Gets dram free space, both in p2p region (managed by p2p_allocator) and in regular part of dram (managed by channel_allocators)
std::pair<std::size_t, size_t> DramAllocator::get_dram_free_space()
{
    size_t regular_free_space = 0;
    size_t p2p_free_space = 0;
    for (std::size_t i = 0; i < channel_allocators.size(); i++)
    {
        regular_free_space += channel_allocators.at(i)->get_capacity();
    }
    p2p_free_space += p2p_allocator->get_capacity();
    return std::make_pair(regular_free_space, p2p_free_space);
}

// clears allocated blocks from all channel allocators, and resets channel picker
void DramAllocator::reset_dram_allocator()
{
    // clear allocator for each channel
    for (auto &channel_allocator : channel_allocators)
    {
        channel_allocator->clear_allocated_blocks();
    }
    // clear also the p2p allocator
    p2p_allocator->clear_allocated_blocks();

    // reset channel picker
    channel_picker->reset();
}

bool DramAllocator::allocate_queues(
    std::vector<DRAMScheduleData> &scheduled_queue_placements, bool disable_dynamic_dram, int microbatch_size)
{
    // print start and end addresses for all channels
    for (std::size_t i = 0; i < channel_allocators.size(); i++)
    {
        log_debug("DRAM channel: {} ", i);

        auto free_blocks_start = channel_allocators[i]->get_blocks().free_blocks_start;
        for (auto &block : free_blocks_start)
        {
            log_debug(
                "\t Free block starts at address: {}, and ends at address {}",
                block.second.addr,
                block.second.addr + block.second.size);
        }
    }
    auto is_cross_epoch_type = [](const Node *q) -> bool
    {
        if (q->node_type() != graphlib::NodeType::kQueue)
            return false;
        if (q->as<graphlib::QueueNode>()->queue_type() != graphlib::QueueNodeType::EpochToEpoch)
            return false;
        return q->as<graphlib::EpochToEpochQueueNode>()->is_cross_epoch_type();
    };

    auto is_cross_chip_type = [](const Node *q) -> bool
    {
        if (q->node_type() != graphlib::NodeType::kQueue)
            return false;
        if (q->as<graphlib::QueueNode>()->queue_type() != graphlib::QueueNodeType::EpochToEpoch)
            return false;
        return q->as<graphlib::EpochToEpochQueueNode>()->is_cross_chip_type();
    };

    auto is_static_queue =
        [disable_dynamic_dram, is_cross_epoch_type, is_cross_chip_type, microbatch_size](const Node *node, bool is_input)
    {
        bool force_dynamic_dram = false;
        // if there is a buffering queue with num_entries < microbatch_size it has to be allocated dynamically. 
        if (node->as<graphlib::QueueNode>()->is_buffering() && node->as<graphlib::BufferingQueueNode>()->get_num_entries() < microbatch_size)
        {
            force_dynamic_dram = true;
        }
        bool disable_dynamic_dram_for_node = !force_dynamic_dram && disable_dynamic_dram;
        return disable_dynamic_dram_for_node || is_input || is_cross_epoch_type(node) || is_cross_chip_type(node) ||
               node->as<graphlib::QueueNode>()->is_grad_accumulator();
    };

    // Sort by queue size descending. When we allocate queues statically, we want to allocate the biggest queues first
    sort(
        scheduled_queue_placements.begin(),
        scheduled_queue_placements.end(),
        [](const DRAMScheduleData &i1, const DRAMScheduleData &i2)
        {
            return i1.second.queue_size > i2.second.queue_size;
        });

    // Allocate all static queues first
    std::vector<DRAMScheduleData*> dynamic_queues;
    for (std::size_t i = 0; i < scheduled_queue_placements.size(); i++)
    {
        auto &[queue_placement, parameters] = scheduled_queue_placements[i];
        if (is_static_queue(parameters.node, parameters.is_input))
        {
            auto [is_allocated, dram_buffers] = allocate_buffers(parameters);
            if (!is_allocated) 
            {
                return false;
            }

            queue_placement.dram_buffers = dram_buffers;
            log_debug(
                "\tstatic queue {}: {} buffers allocated, on channel {}, address {} ",
                queue_placement.name,
                queue_placement.dram_buffers.size(),
                queue_placement.dram_buffers[0].dram_channel,
                queue_placement.dram_buffers[0].dram_address);
        }
        else
        {
            dynamic_queues.push_back(&scheduled_queue_placements[i]);
        }
    }

    // When allocating queues we keep them sorted by last consumer epoch so that queues that 
    // are deallocated at the same time are more likely to be allocated next to each other.
    //
    struct ConsumerEpochComparator {
        bool operator()(const DRAMScheduleData* lhs, const DRAMScheduleData* rhs) const {
            return lhs->second.last_consumer_epoch < rhs->second.last_consumer_epoch;
        }
    };

    // For each epoch keep track of the dynamic queues to allocate and deallocate.
    //
    using QueueScheduleData = std::pair<std::multiset<DRAMScheduleData*, ConsumerEpochComparator>, std::vector<DRAMScheduleData*>>;
    std::map<std::uint32_t, QueueScheduleData> epoch_to_queue_schedule;
    for (DRAMScheduleData* queue: dynamic_queues) 
    {
        const QueueDRAMPlacementParameters& placement_parameters = queue->second;
        epoch_to_queue_schedule[placement_parameters.producer_epoch].first.insert(queue);
        epoch_to_queue_schedule[placement_parameters.last_consumer_epoch].second.push_back(queue);
    }

    // For each epoch, we first allocate dynamic queues starting at that epoch and then deallocate queues ending at that epoch,
    // since queues ending at that epoch persist until its end.
    //
    for (auto& [epoch, queue_schedule_data]: epoch_to_queue_schedule) 
    {
        auto& [queues_to_allocate, queues_to_deallocate] = queue_schedule_data;
        for (DRAMScheduleData* queue_to_allocate : queues_to_allocate) 
        {
            auto& [queue_placement, placement_parameters] = *queue_to_allocate;
            auto [is_allocated, dram_buffers] = allocate_buffers(placement_parameters);
            if (!is_allocated) 
            {
                return false;
            }

            queue_placement.dram_buffers = dram_buffers;
            queue_placement.epoch_allocate = placement_parameters.producer_epoch;
            log_debug(
                "\tdynamic queue {}: {} buffers allocated, on channel {}, address {} ",
                queue_placement.name,
                queue_placement.dram_buffers.size(),
                queue_placement.dram_buffers[0].dram_channel,
                queue_placement.dram_buffers[0].dram_address);
        }

        for (DRAMScheduleData* queue_to_deallocate : queues_to_deallocate) 
        {
            auto& [queue_placement, placement_parameters] = *queue_to_deallocate;
            deallocate_buffers(queue_placement, placement_parameters);
            queue_placement.epoch_deallocate = placement_parameters.last_consumer_epoch;
        }
    }

    // pass through scheduled queue placements and divide channel by two to get real dram channel index.
    // we do this here because we need virtual channel indices in above lines, for allocator manipulation
    if (dram_config.device_config.is_wormhole_b0())
    {
        for (std::size_t i = 0; i < scheduled_queue_placements.size(); i++)
        {
            auto &[queue_placement, parameters] = scheduled_queue_placements[i];
            for (std::size_t j = 0; j < queue_placement.dram_buffers.size(); j++)
            {
                int channel = queue_placement.dram_buffers[j].dram_channel;
                queue_placement.dram_buffers[j].dram_channel = channel / 2;
            }
        }
    }

    dram_logger->dump_to_reportify(
        reportify::get_default_reportify_path(graph_name) + reportify::get_memory_report_relative_directory(),
        graph_name);

    return true;
}

QueueBufferPlacement DramAllocator::create_buffer_placement(
    std::uint32_t virtual_channel,
    std::size_t channel_address,
    std::size_t buffer_size,
    bool in_p2p_region)
{
    std::uint32_t real_channel = virtual_channel;
    if (dram_config.device_config.is_wormhole_b0())
    {
        // On wormhole, each channel is divided into 2 virtual channels, so to get the real channel index
        // we have to divide the virtual channel by 2.
        //
        real_channel = virtual_channel / 2;
    }

    return QueueBufferPlacement{
        .dram_channel = virtual_channel,
        .dram_address = channel_address,
        .dram_channel_location = dram_config.dram_config[real_channel].location,
        .buffer_size = buffer_size,
        .allocated_in_p2p_region = in_p2p_region,
    };
}

std::pair<bool, std::vector<QueueBufferPlacement>> DramAllocator::allocate_buffers(const QueueDRAMPlacementParameters &parameters)
{
    const std::size_t num_channels = channel_allocators.size();
    const std::size_t buffer_size = parameters.queue_size;
    TT_ASSERT(buffer_size > 0, "Buffer size for queue {} must be larger than 0", parameters.node->name());

    std::vector<QueueBufferPlacement> buffer_placement;
    for (std::uint32_t row = 0; row < parameters.grid_shape.rows; row++)
    {
        for (std::uint32_t col = 0; col < parameters.grid_shape.columns; col++)
        {
            std::size_t allocated_address;

            if (parameters.in_p2p_region_soft or parameters.in_p2p_region_hard) 
            {
                // Try to allocate the queue first in the p2p region.
                //
                if (p2p_allocator->allocate(buffer_size, allocated_address))
                {
                    buffer_placement.push_back(create_buffer_placement(0 /* virtual_channel */, allocated_address, buffer_size, true /* in_p2p_region */));
                    dram_logger->log_allocate(parameters.node, 0 /* dram_channel */, allocated_address, buffer_size, parameters.producer_epoch);

                    continue;
                }
                else if (parameters.in_p2p_region_hard) 
                {
                    // Queue must be allocated in the p2p region but the allocation has failed.
                    //
                    return std::make_pair(false, buffer_placement);
                }
            }

            const DramQueueMap& manual_queue_placement = this->dram_config.manual_dram_queue_placement;
            std::optional<std::uint32_t> channel_override;
            auto it = manual_queue_placement.find(parameters.node->name());
            if (it != manual_queue_placement.end()) 
            {
                channel_override = it->second.channel;
            }

            std::uint32_t selected_channel;
            if (channel_override.has_value()) 
            {
                selected_channel = channel_override.value();
                log_debug(tt::LogPlacer, "Manually placing DRAM queue {} to channel: {}", parameters.node->name(), selected_channel);
            } 
            else 
            {
                selected_channel = channel_picker->pick_channel(parameters, Coord{row, col}, channel_allocators);
            }
            TT_ASSERT(selected_channel < channel_allocators.size(), "Chosen DRAM channel {} for queue allocation is invalid", selected_channel);

            for (std::size_t attempt = 0; attempt < num_channels; attempt++)
            {
                // Try to allocate the queue in regular DRAM channels.
                //
                if (channel_allocators.at(selected_channel)->allocate(buffer_size, allocated_address))
                {
                    buffer_placement.push_back(create_buffer_placement(selected_channel, allocated_address, buffer_size, false /* in_p2p_region */));
                    dram_logger->log_allocate(parameters.node, selected_channel, allocated_address, buffer_size, parameters.producer_epoch);

                    break;
                }
                
                if (attempt == num_channels - 1) 
                {
                    // Allocation has been tried on all channel.
                    //
                    return std::make_pair(false, buffer_placement);
                }

                // Use round robin method to choose the next possible channel.
                //
                selected_channel = (selected_channel + 1) % num_channels;
            }
        }
    }
    
    return std::make_pair(true, buffer_placement);
}

void DramAllocator::deallocate_buffers(
    const QueuePlacement& queue_placement,
    const QueueDRAMPlacementParameters& placement_parameters) 
{
    TT_ASSERT(!queue_placement.dram_buffers.empty(), "Queue {} does not have any buffers allocated.", queue_placement.name);

    for (auto &buffer : queue_placement.dram_buffers)
    {    
        if (buffer.allocated_in_p2p_region) 
        {
            p2p_allocator->deallocate(buffer.dram_address);
        }
        else 
        {
            channel_allocators.at(buffer.dram_channel)->deallocate(buffer.dram_address);
        }
        dram_logger->log_deallocate(
            buffer.dram_channel, buffer.dram_address, placement_parameters.last_consumer_epoch);
    }
}

std::uint32_t noc_distance(const Coord &start, const Coord &end, const tt::DeviceGrid &grid_size, std::uint32_t noc)
{
    // NOC0 goes right and down, NOC1 goes left and up.
    auto grid_r = grid_size.r + 2;  // physical grid size
    auto grid_c = grid_size.c + 2;

    if (noc == 0)
    {
        // Check for wrap
        auto x_dist = (start.col <= end.col) ? end.col - start.col : grid_c - start.col + end.col;
        auto y_dist = (start.row <= end.row) ? end.row - start.row : grid_r - start.row + end.row;
        return x_dist + y_dist;
    }
    else
    {
        // Check for wrap
        auto x_dist = (start.col >= end.col) ? start.col - end.col : grid_c - end.col + start.col;
        auto y_dist = (start.row >= end.row) ? start.row - end.row : grid_r - end.row + start.row;
        return x_dist + y_dist;
    }
}

// Check if channel is unused, and modify it if an alias is available
bool ClosestPicker::is_channel_unused(
    std::uint32_t &channel, bool is_grayskull, const std::unordered_set<std::uint32_t> &unused_channels)
{
    if (unused_channels.count(channel) == 0)
    {
        // Check the other one for wormhole
        if (!is_grayskull && (channel % 2 == 0) && unused_channels.count(channel + 1) > 0)
        {
            channel += 1;
            return true;
        }
        return false;
    }
    return true;
}

// Pick the closest DRAM channel, to minimize the distance on either NOC0 or NOC1
// For each channel, we will calculate the average distance, on each noc, to all consumers, and the producer. If the
// channel has more than one location, all will be calculated, and lowest picked. Finally, the channel with the lowest
// average distance is picked.
std::uint32_t ClosestPicker::pick_channel(
    const QueueDRAMPlacementParameters &parameters,
    Coord c,
    const std::vector<std::unique_ptr<ChannelAllocator>> &channel_allocators)
{
    const auto &config = *parameters.config;
    const auto *node = parameters.node;

    if (node == current_node)
    {
        // We already have a solution, just return it.
        TT_ASSERT(solution.count(c.row) > 0);
        TT_ASSERT(solution.at(c.row).count(c.col) > 0);
        log_trace(
            tt::LogPlacer, "Picking channel {} for queue {} at {}", solution.at(c.row).at(c.col), node->name(), c);
        return solution.at(c.row).at(c.col);
    }

    // We don't have a solution yet, so we need to calculate it.
    bool is_grayskull = parameters.config->device_config.is_grayskull();

    std::unordered_set<std::uint32_t> unused_channels;
    auto reset_channels = [&config, is_grayskull](std::unordered_set<std::uint32_t> &channels)
    {
        channels.clear();
        for (std::size_t i = 0; i < config.dram_config.size(); i++)
        {
            if (is_grayskull)
            {
                channels.insert(config.dram_config[i].channel);
            }
            else
            {
                channels.insert(config.dram_config[i].channel * 2);
                channels.insert(config.dram_config[i].channel * 2 + 1);
            }
        }
    };
    reset_channels(unused_channels);

    //
    // Each buffer has to be allocated a single channel. However, each epoch can have a different subchannel, i.e. a
    // differe core to read from, or to write to. So, we need to go through all the epochs and pick the channel that
    // minimizes the distance to all consumers, from whatever the best core is in each epoch.
    //
    // There's currently no way to specify this in the netlist, one we pick the channel, but we will eventually have it.
    // In the meantime, we rely on backend to pick the closest core whenever possible.
    //

    for (std::uint32_t row = 0; row < parameters.grid_shape.rows; row++)
    {
        for (std::uint32_t col = 0; col < parameters.grid_shape.columns; col++)
        {
            // Get producers and consumers for this buffer in the queue
            TT_LOG_ASSERT(
                parameters.consumer_loc.count(row) > 0,
                "Missing consumer location for queue {} for row {}",
                node->name(),
                row);
            TT_LOG_ASSERT(
                parameters.consumer_loc.at(row).count(col) > 0,
                "No consumers for queue {} at row={}, col={}",
                node->name(),
                row,
                col);
            std::vector<std::pair<Coord, std::uint32_t>> consumers = parameters.consumer_loc.at(row).at(col);

            // Per channel, per epoch, record best distance and subchannel that has it
            std::
                unordered_map<std::uint32_t, std::unordered_map<std::uint32_t, std::pair<std::uint32_t, std::uint32_t>>>
                    best_distance;

            // Record used epochs so we can check for unused channels per epoch
            std::unordered_set<std::uint32_t> used_epochs;
            for (auto &consumer : consumers)
            {
                used_epochs.insert(consumer.second);
            }

            // Producer is optional - some queues are filled from the host before the epoch runs (i.e. constants/parameters)
            if ((parameters.producer_loc.count(row) > 0) && (parameters.producer_loc.at(row).count(col) > 0))
            {
                std::pair<Coord, std::uint32_t> producer = parameters.producer_loc.at(row).at(col);
                used_epochs.insert(producer.second);
            }

            // Go through all dram locations, and find the distance to all consumers and the producer. If it's shorter
            // than current best_distance, update best_distance and best_channel.
            for (auto &dram_c : config.dram_config)
            {
                // Only consider channels that haven't been picked. Grayskull has 1-1 mapping, but Wormhole actually has
                // 2 channels for each one in dram config.
                auto real_channel = is_grayskull ? dram_c.channel : dram_c.channel * 2;
                if (!is_channel_unused(real_channel, is_grayskull, unused_channels))
                {
                    continue;
                }

                // Also check per-epoch unused channels, and skip the channel if it's been used
                auto &epoch_map = parameters.is_prologue ? unused_channels_in_epoch_prologue : unused_channels_in_epoch;
                for (auto &epoch : used_epochs)
                {
                    if (epoch_map.count(epoch) == 0)
                    {
                        epoch_map[epoch] = std::unordered_set<std::uint32_t>();
                        reset_channels(epoch_map[epoch]);
                    }
                    if (!is_channel_unused(real_channel, is_grayskull, epoch_map[epoch]))
                    {
                        continue;
                    }
                }

                std::unordered_map<std::uint32_t, std::uint32_t> distance;  // per epoch

                for (auto &consumer : consumers)
                {
                    auto noc0 = noc_distance(dram_c.location, consumer.first, config.device_config.grid_size, 0);
                    auto noc1 = noc_distance(dram_c.location, consumer.first, config.device_config.grid_size, 1);
                    auto d = (noc0 < noc1) ? noc0 : noc1;
                    distance[consumer.second] += d;
                }

                if ((parameters.producer_loc.count(row) > 0) && (parameters.producer_loc.at(row).count(col) > 0))
                {
                    std::pair<Coord, std::uint32_t> producer = parameters.producer_loc.at(row).at(col);
                    auto noc0 = noc_distance(producer.first, dram_c.location, config.device_config.grid_size, 0);
                    auto noc1 = noc_distance(producer.first, dram_c.location, config.device_config.grid_size, 1);
                    auto d = (noc0 < noc1) ? noc0 : noc1;
                    distance[producer.second] += d;

                    for (auto it : distance)
                    {
                        if (best_distance.count(real_channel) == 0 || best_distance.at(real_channel).count(it.first) == 0 ||
                            it.second < best_distance.at(real_channel).at(it.first).second)
                        {
                            best_distance[real_channel][it.first] = std::make_pair(real_channel, it.second);
                        }
                    }
                }
            }

            // For each channel, sum up the epoch distances, and find the best channel
            std::uint32_t total_best_distance = std::numeric_limits<std::uint32_t>::max();
            std::uint32_t total_best_channel = 0;

            for (auto &[channel, epoch_distances] : best_distance)
            {
                std::uint32_t total_distance = 0;
                for (auto &[epoch, distance] : epoch_distances)
                {
                    total_distance += distance.second;
                }
                if (total_distance < total_best_distance)
                {
                    total_best_distance = total_distance;
                    total_best_channel = channel;
                }
            }

            solution[row][col] = total_best_channel;
            unused_channels.erase(total_best_channel);

            if (unused_channels.empty())
            {
                // Go back to picking from all channels
                reset_channels(unused_channels);
            }

            // Clear per-epoch unused channels
            auto &epoch_map = parameters.is_prologue ? unused_channels_in_epoch_prologue : unused_channels_in_epoch;
            for (auto &epoch : used_epochs)
            {
                const bool no_epoch = env_as<bool>("PYBUDA_CLOSEST_NO_EPOCH", false);
                if (!no_epoch)
                    epoch_map[epoch].erase(total_best_channel);
                if (epoch_map[epoch].empty())
                {
                    reset_channels(epoch_map[epoch]);
                }
            }
        }
    }
    current_node = node;
    return pick_channel(parameters, c, channel_allocators);
}

// Reset the ClosestPicker structures
void ClosestPicker::reset()
{
    solution.clear();
    unused_channels_in_epoch.clear();
    unused_channels_in_epoch_prologue.clear();
    current_node = nullptr;
}

}  // namespace tt::placer
