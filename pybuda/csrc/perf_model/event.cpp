// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "perf_model/event.hpp"

#include <list>
#include <queue>

#include "perf_model/simulator.hpp"

constexpr std::uint32_t PACKER_OPERAND = 16;

namespace tt::perf_model
{

// TODO: move this to somewhere and add better estimates
std::uint32_t get_noc_transfer_time(Buffer *, Buffer *, std::uint32_t)
{
    // noc latency is mostly hidden by pack latency... still, need to model this
    return 100; /* return 500 + int(count * 32);*/
}
std::uint32_t get_host_transfer_time(std::uint32_t)
{
    // Need to model this better...
    return 100; /*return 1000 + int(count * 320); */
}
std::uint32_t get_pack_time(std::uint32_t count) { return 100 + count * 16; }

OpDataEvent::OpDataEvent(
    std::uint32_t input_index,
    TimeData data,
    Buffer *output_buffer,
    std::uint32_t current_t,
    std::uint32_t current_ublock,
    std::uint32_t current_k) :
    DataEvent(input_index, data, output_buffer),
    current_t(current_t),
    current_ublock(current_ublock),
    current_k(current_k)
{
    op = output_buffer->get_node();
    total_t = op->get_perf_data()->op_perf_data.op_model.block_shape().t;
    total_ublocks = op->get_perf_data()->op_perf_data.op_model.block_shape().mblock_m *
                    op->get_perf_data()->op_perf_data.op_model.block_shape().mblock_n;
    total_k = 1;

    if ( (op->get_op_type() == "matmul") || (op->get_op_type() == "sparse_matmul") )
    {
        total_k = op->get_perf_data()->attr.m_k;
        total_ublocks = 1;
    }
    if (op->get_op_type() == "reduce")
    {
        total_k = op->get_perf_data()->attr.m_k * total_ublocks;
        total_ublocks = 1;
    }

    // We need to do some fused op input analysis when fusing, and then figure out their ublock consumption rate when setting input ublock shapes, and input buffers.
    // Unitl then, we can "cheat", and pretend that fused op consumes and produces full inputs/mblocks, and doesn't stream ublock by ublock.
    if (op->get_op_type() == "fused_op")
        total_ublocks = 1;

    TT_ASSERT(current_t < total_t);
    TT_ASSERT(current_k < total_k);
}

OutputDataEvent::OutputDataEvent(
    std::uint32_t input_index,
    TimeData data,
    Buffer *output_buffer,
    const std::vector<std::pair<Buffer *, std::uint32_t>> &consumers) :
    DataEvent(input_index, data, output_buffer), consumers(consumers)
{
    TT_ASSERT(
        consumers.size() > 0,
        "Node " + output_buffer->get_node()->get_name() + " has no consumers, but has output buffer");

    for (std::size_t i = 0; i < consumers.size(); i++)
    {
        remaining.push_back(consumers.at(i).first->get_broadcast_multiplier() * data.count);  // remaining to send
    }

    consumed = std::vector<std::vector<TimeData>>(consumers.size(), std::vector<TimeData>());
}

// Process this event. Return pointer to buffer on which we're stalled, if stalled...
// If any new events have been generated, populate them in new_events vector.
ProcessStatus OutputDataEvent::process(SimStateP &sim_state, SimCacheP &, std::string const&)
{
    ProcessStatus ret;
    if (unprocessed)
    {
        buffer->insert_data(data.count);
        unprocessed = false;
    }

    auto max_before_broadcast = [&]()
    {
        std::uint32_t max = 0;
        for (std::size_t i = 0; i < consumers.size(); i++)
        {
            std::uint32_t broadcast_multiplier = consumers[i].first->get_broadcast_multiplier();

            // round up
            std::uint32_t r = ((remaining[i] + broadcast_multiplier - 1) / broadcast_multiplier) * broadcast_multiplier;
            r /= broadcast_multiplier;
            if (r > max)
                max = r;
        }
        return max;
    };

    std::uint32_t max_remaing_before = max_before_broadcast();
    for (std::size_t i = 0; i < consumers.size(); i++)
    {
        if (remaining[i] == 0)
            continue;  // we're done with this consumer

        auto &[target_buffer, operand] = consumers[i];

        std::uint32_t to_transfer = std::min(target_buffer->available_space(), remaining[i]);
        if (to_transfer == 0)
        {
            ret.stall_reason.push_back(target_buffer);
            continue;  // no room
        }

        target_buffer->reserve_space(to_transfer);
        remaining[i] -= to_transfer;
        consumed[i].push_back(TimeData{.count = to_transfer, .timestamp = sim_state->timestamp});

        // Create an input buffer event
        std::uint32_t time_increment = get_noc_transfer_time(buffer, target_buffer, to_transfer);
        ret.new_events.push_back(new InputDataEvent(
            input_index,
            TimeData{.count = to_transfer, .timestamp = sim_state->timestamp + time_increment},
            target_buffer));

        ret.modified_buffers.push_back(buffer);

        if (remaining[i] > 0)
            ret.stall_reason.push_back(target_buffer);  // we're still stalled since we couldn't send it all
    }

    std::uint32_t max_remaing_after = max_before_broadcast();
    if (max_remaing_after < max_remaing_before)
        buffer->pop_data(max_remaing_before - max_remaing_after);

    if (max_remaing_before > max_remaing_after)
        SIMLOG << "  POPPED " << (max_remaing_before - max_remaing_after) << " from output buffer." << std::endl;

    return ret;
}

std::string OutputDataEvent::to_string() const
{
    std::stringstream ss;
    ss << "OutputDataEvent(input=" << input_index << ", org=@" << data.timestamp << ", count=" << data.count
       << ", buf=" << buffer->to_string();
    return ss.str();
}

// Process this event. Return pointer to buffer on which we're stalled, if stalled...
// If any new events have been generated, populate them in new_events vector.
ProcessStatus InputDataEvent::process(SimStateP &, SimCacheP &, std::string const&)
{
    if (unprocessed)
    {
        buffer->insert_data(data.count);
        unprocessed = false;
    }
    ProcessStatus ret;
    ret.modified_buffers.push_back(buffer);

    return ret;  // this event never stalls
}

// Process this event. Return pointer to buffer on which we're stalled, if stalled...
// If any new events have been generated, populate them in new_events vector.
ProcessStatus OpDataEvent::process(SimStateP &sim_state, SimCacheP &cache, std::string const& arch_name)
{
    unprocessed = false;

    // We'll check all input buffers for the op that this input data belongs to, and if everything is available,
    // and there's room in output buffer, we'll go ahead and produce data
    NodeP node = buffer->get_node();
    TT_ASSERT(node->is_op());

    std::vector<Buffer *> input_buffers = cache->node_input_buffers(node);
    Buffer *output_buffer = buffer;

    ProcessStatus ret;
    for (Buffer *b : input_buffers)
    {
        if (!b->above_threshold())
        {
            // stalled on this input buffer
            ret.stall_reason.push_back(b);
            SIMLOG << "   @" << sim_state->timestamp << " [op] Stalled " << node->get_name() << " on input buffer"
                   << b->to_string() << std::endl;

            if (sim_state->trace)
                sim_state->trace_op.at(node)->unpack_stall(
                    input_index, b->get_operand(), sim_state->timestamp, b->get_threshold());
        }
        else
        {
            if (sim_state->trace)
                sim_state->trace_op.at(node)->unpack_data_available(
                    input_index, b->get_operand(), sim_state->timestamp);
        }
    }

    // check that output has enough room
    std::uint32_t output_size = cache->node_output_size_in_tiles(node) / (total_t * total_ublocks);
    TT_LOG_ASSERT(output_size > 0, "Node {} has output size of 0", node->get_name());

    if (output_buffer->available_space() < output_size)
    {
        // stalled on output buffer having enough space
        ret.stall_reason.push_back(output_buffer);
        SIMLOG << "   @" << sim_state->timestamp << " [op] Stalled " << node->get_name() << " on output buffer"
               << std::endl;

        if (sim_state->trace)
            sim_state->trace_op.at(node)->pack_stall(input_index, PACKER_OPERAND, sim_state->timestamp, output_size);
    }

    if (ret.stall_reason.size() > 0)
        return ret;

    // Good to go
    SIMLOG << "   @" << sim_state->timestamp << " [op] Executed " << node->get_name() << ", in=" << input_index
           << ", t=" << current_t << "/" << total_t << ", k=" << current_k << "/" << total_k
           << ", ublock=" << current_ublock << "/" << total_ublocks << std::endl;

    std::uint32_t op_time =
        node->get_perf_data()->op_perf_data.cycle_count_ideal(arch_name) / (total_t * total_k * total_ublocks);
    std::uint32_t end_time = sim_state->timestamp;

    end_time += op_time;

    // Pop input buffers
    std::vector<Buffer *> pop_on_output, pop_on_end;
    for (std::size_t i = 0; i < input_buffers.size(); i++)
    {
        Buffer *b = input_buffers[i];
        if ((node->get_op_type() == "matmul") && (i == 2))
        {
            pop_on_output.push_back(b);  // Bias only gets popped on produced output
        }
        else if ((node->get_op_type() == "sparse_matmul") && ((i==0) || (i==2)))
        {
            pop_on_end.push_back(b);  // tiles/indices only get popped at the end
        }
        else
        {
            b->pop_threshold();
            ret.modified_buffers.push_back(b);
        }
    }

    // If this is not the last k/t, schedule the next one... otherwise produce output
    std::uint32_t next_t, next_k, next_ublock;
    bool produce_output = false;
    bool next_op = false;

    bool intermediate = false;
    if (current_k + 1 < total_k)
    {
        next_t = current_t;
        next_k = current_k + 1;
        next_ublock = current_ublock;
        TT_ASSERT(current_ublock == 0, "Matmul doesn't have partial ublock outputs");
        intermediate = true;
    }
    if (current_ublock + 1 < total_ublocks)
    {
        next_t = current_t;
        next_ublock = current_ublock + 1;
        next_k = current_k;
        TT_ASSERT(current_k == 0, "Non-matmul doesn't have 'k' counter");
        intermediate = true;
        produce_output = true;
    }

    if (!intermediate)
    {
        if (current_t + 1 < total_t)
        {
            next_t = current_t + 1;
            next_k = 0;
            next_ublock = 0;
            produce_output = true;
        }
        else
        {
            next_t = 0;
            next_k = 0;
            next_ublock = 0;
            produce_output = true;
            next_op = true;
        }
    }

    if (produce_output)
    {
        for (Buffer *b: pop_on_output)
        {
            b->pop_threshold();
            ret.modified_buffers.push_back(b);
        }

        // Produce output
        output_buffer->reserve_space(output_size);
        ret.modified_buffers.push_back(output_buffer);

        SIMLOG << "    Op " << node->get_name() << " produced output size=" << output_size << std::endl;

        std::uint32_t pack_time = get_pack_time(output_size);
        ret.new_events.push_back(new OutputDataEvent(
            input_index,
            TimeData{.count = output_size, .timestamp = end_time + pack_time},
            output_buffer,
            cache->node_outputs(node)));

        if (sim_state->trace)
            sim_state->trace_op.at(node)->pack_started(input_index, PACKER_OPERAND, end_time);
        if (sim_state->trace)
            sim_state->trace_op.at(node)->pack_ended(input_index, end_time + pack_time);
    }

    if (!next_op || (input_index + 1 < sim_state->total_input_count))
    {
        std::uint32_t next_input_index = next_op ? input_index + 1 : input_index;
        ret.new_events.push_back(new OpDataEvent(
            next_input_index,
            TimeData{.count = data.count, .timestamp = end_time},
            output_buffer,
            next_t,
            next_ublock,
            next_k));
    }
    
    if (next_op)
    {
        for (Buffer *b: pop_on_end)
        {
            b->pop_threshold();
            ret.modified_buffers.push_back(b);
        }

    }

    return ret;
}

// Process this event. Return pointer to buffer on which we're stalled, if stalled...
// If any new events have been generated, populate them in new_events vector.
ProcessStatus QueueDataEvent::process(SimStateP &sim_state, SimCacheP &cache, std::string const&)
{
    unprocessed = false;

    // We'll check all input buffers for the op that this input data belongs to, and if everything is available,
    // and there's room in output buffer, we'll go ahead and produce data
    NodeP node = buffer->get_node();
    TT_ASSERT(!node->is_op());

    Buffer *input_buffer = buffer;
    Buffer *output_buffer = cache->node_output_buffer(node);

    ProcessStatus ret;
    if (!input_buffer->above_threshold())
    {
        ret.stall_reason.push_back(input_buffer);
        return ret;  // wait for data
    }

    //
    // NOTE: Queue doesn't really have an "output buffer". However, the output buffer
    // concept and output buffer event that comes with it "just works", so we don't have
    // implement special behaviour for the queue. Setting latency of 0 from input to output buffer
    // should produce correct results.
    //
    // check that output has enough room
    std::uint32_t output_size = data.count;
    if (output_buffer->available_space() < output_size)
    {
        // stalled on output buffer having enough space
        ret.stall_reason.push_back(output_buffer);
        return ret;  // wait for room in output buffer
    }

    // Pop input buffer
    input_buffer->pop_threshold();
    ret.modified_buffers.push_back(input_buffer);

    // Produce output
    output_buffer->reserve_space(output_size);
    ret.modified_buffers.push_back(output_buffer);

    ret.new_events.push_back(new OutputDataEvent(
        input_index,
        TimeData{.count = output_size, .timestamp = sim_state->timestamp},  // 0 latency
        output_buffer,
        cache->node_outputs(node)));

    if (input_index + 1 < sim_state->total_input_count)
    {
        ret.new_events.push_back(new QueueDataEvent(
            input_index + 1, TimeData{.count = data.count, .timestamp = sim_state->timestamp + 1}, buffer));
    }

    return ret;
}

std::string InputDataEvent::to_string() const
{
    std::stringstream ss;
    ss << "InputDataEvent(input=" << input_index << ", org=@" << data.timestamp << ", count=" << data.count
       << ", buf=" << buffer->to_string();
    return ss.str();
}

std::string OpDataEvent::to_string() const
{
    std::stringstream ss;
    ss << "OpDataEvent(input=" << input_index << ", org=@" << data.timestamp << ", count=" << data.count
       << ", op=" << op->get_name();
    return ss.str();
}

std::string QueueDataEvent::to_string() const
{
    std::stringstream ss;
    ss << "QueueDataEvent(input=" << input_index << ", org=@" << data.timestamp << ", count=" << data.count
       << ", queue=" << buffer->get_node()->get_name();
    return ss.str();
}

std::string HostWriteDataEvent::to_string() const
{
    std::stringstream ss;
    ss << "HostWriteDataEvent(input=" << input_index << ", org=@" << data.timestamp << ", count=" << data.count
       << ", input=" << buffer->get_node()->get_name();
    return ss.str();
}

std::string HostReadDataEvent::to_string() const
{
    std::stringstream ss;
    ss << "HostReadDataEvent(input=" << input_index << ", org=@" << data.timestamp << ", count=" << data.count
       << ", output=" << buffer->get_node()->get_name();
    return ss.str();
}

// Process this event. Return pointer to buffer on which we're stalled, if stalled...
// If any new events have been generated, populate them in new_events vector.
ProcessStatus HostWriteDataEvent::process(SimStateP &sim_state, SimCacheP &cache, std::string const&)
{
    // Host only transfers if there's room for the whole input
    ProcessStatus ret;
    if (buffer->available_space() < data.count)
    {
        ret.stall_reason.push_back(buffer);
        return ret;
    }

    // Transfer
    buffer->reserve_space(data.count);
    ret.modified_buffers.push_back(buffer);
    unprocessed = false;

    ret.new_events.push_back(new OutputDataEvent(
        input_index,
        TimeData{.count = data.count, .timestamp = sim_state->timestamp + get_host_transfer_time(data.count)},
        buffer,
        cache->node_outputs(buffer->get_node())));
    return ret;
}

// Process this event. Return pointer to buffer on which we're stalled, if stalled...
// If any new events have been generated, populate them in new_events vector.
ProcessStatus HostReadDataEvent::process(SimStateP &, SimCacheP &, std::string const&)
{
    ProcessStatus ret;
    if (!buffer->above_threshold())
    {
        ret.stall_reason.push_back(buffer);
        return ret;
    }

    // Read data
    buffer->pop_data(data.count);
    ret.modified_buffers.push_back(buffer);
    unprocessed = false;
    return ret;
}

}  // namespace tt::perf_model
