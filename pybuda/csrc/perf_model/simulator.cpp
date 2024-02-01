// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "perf_model/simulator.hpp"
#include "utils/assert.hpp"

namespace tt::perf_model
{
std::uint32_t Buffer::s_id = 0;
std::ofstream Simulator::s_log;
bool Simulator::s_write_log = false;

Simulator::Simulator(Graph *graph, std::uint32_t input_count, bool trace, bool log) :
    graph(graph), input_count(input_count)
{
    sim_state = std::make_unique<SimState>(
        SimState{.timestamp = 0, .total_input_count = input_count, .trace = trace, .trace_op = {}});
    s_write_log = log;
}

SimCache::~SimCache()
{
    for (auto &[node, input_buffers] : node_input_buffer_map)
        for (auto input_buffer : input_buffers) delete input_buffer;

    for (auto &[node, output_buffer] : node_output_buffer_map) delete output_buffer;
}

std::uint32_t Buffer::available_space() const { return size - occupied - reserved; }
void Buffer::reserve_space(std::uint32_t count)
{
    TT_ASSERT(available_space() >= count);
    reserved += count;
}

void Buffer::insert_data(std::uint32_t count)
{
    TT_ASSERT(reserved >= count, "Trying to insert data without reserving space first");
    reserved -= count;
    occupied += count;
}

void Buffer::pop_data(std::uint32_t count)
{
    TT_ASSERT(
        occupied >= count,
        "Buffer underflow (have  {}, popping {}): {}", occupied, count, to_string());
    occupied -= count;
}

void Buffer::pop_threshold() { pop_data(threshold); }

bool Buffer::above_threshold() const { return occupied >= threshold; }

bool Buffer::empty() const { return (occupied == 0) && (reserved == 0); }

std::string Buffer::to_string(bool show_contents) const
{
    std::stringstream ss;
    ss << "#" << unique_id;
    if (input)
        ss << " InputBuffer";
    else
        ss << " OutputBuffer";

    ss << "(" << owner->get_name();
    if (input)
        ss << ", operand " << operand;
    ss << ")";

    if (show_contents)
        ss << " size: " << size << ", occupied: " << occupied << ", reserved: " << reserved
           << ", threshold: " << threshold << ", broadcast_x: " << broadcast_multiplier;
    return ss.str();
}

std::vector<std::pair<Buffer *, std::uint32_t>> SimCache::node_outputs(NodeP node)
{
    // Get consumer / op index
    auto consumers = [&](const NodeP node)
    {
        std::vector<std::pair<Buffer *, std::uint32_t>> ret;
        std::unordered_map<NodeP, std::uint32_t>
            last_operand_index;  // to detect the case where multiple outputs go to same node
        for (NodeP user : node->get_outputs())
        {
            auto operand_index = user->get_operand_index(node, last_operand_index[user]);
            last_operand_index[user] = operand_index + 1;  // skip this one next time it's looked up
            ret.push_back(std::make_pair(node_input_buffer(user, operand_index), operand_index));
        }
        return ret;
    };

    auto it = node_output_map.find(node);
    if (it == node_output_map.end())
    {
        auto ret = consumers(node);
        node_output_map.insert(std::make_pair(node, ret));
        return ret;
    }

    return it->second;
}

const std::vector<Buffer *> SimCache::node_input_buffers(NodeP node)
{
    auto it = node_input_buffer_map.find(node);
    if (it == node_input_buffer_map.end())
    {
        std::vector<Buffer *> ibs;
        for (std::size_t operand_index = 0; operand_index < node->get_operands().size(); operand_index++)
        {
            if (!node->is_op())
            {
                std::uint32_t output_size = node_output_size_in_tiles(node);
                ibs.push_back(new Buffer(node, operand_index, output_size * 2, output_size, 1));
                continue;
            }

            NodeP operand = node->get_operands()[operand_index];
            std::uint32_t input_size =
                node->get_perf_data()->op_perf_data.op_model.input_buffers.at(operand_index).l1_size_tiles;

            auto perf_data = node->get_perf_data()->op_perf_data;
            auto input_block_shape = perf_data.op_model.input_buffers.at(operand_index).block_shape;
            std::uint32_t threshold = input_block_shape.buffered_rt() * input_block_shape.buffered_ct();

            std::uint32_t grid_multiplier = 0;
            if ( (node->get_op_type() == "matmul") || (node->get_op_type() == "sparse_matmul") )
            {
                if (operand_index == 0)
                    grid_multiplier = perf_data.grid.size_r;  // same activations on each column
                else if (operand_index == 1)
                    grid_multiplier = perf_data.grid.size_c;  // same params on each row
                else if (operand_index == 2) {
                    if (node->get_op_type() == "matmul")
                    {
                        // Bias input shape is just ublock, but matmul even doesn't read this ublock by ublock...
                        // so let's override here for now
                        threshold = perf_data.op_model.input_buffers.at(1).block_shape.buffered_ct();
                        if (input_size < threshold) input_size = threshold; // not correct, but ok for bias... TODO
                        grid_multiplier = perf_data.grid.size_c * node->get_perf_data()->output.shape.rt();  // bias
                    }
                    else
                    {
                        // Special case, we read some variable amount per t... so we'll just read all of it once at the end
                        input_size = node->get_perf_data()->inputs.at(2).size_in_tiles();
                        threshold = input_size;
                        grid_multiplier = 1;
                    }
                }
                else 
                {
                    TT_THROW("Invalid operand for matmul");
                }

            }
            else
                grid_multiplier = perf_data.grid.size();

            input_size *= grid_multiplier;
            threshold *= grid_multiplier;
            std::uint32_t broadcast_multiplier = node->get_perf_data()->input_broadcast_multiplier.at(operand_index);

            if (node->get_op_type() == "fused_op")
            {
                // We need to do some fused op input analysis when fusing, and then figure out their ublock consumption rate when setting input ublock shapes, and input buffers.
                // Unitl then, we can "cheat", and pretend that fused op consumes and produces full inputs/mblocks, and doesn't stream ublock by ublock.
                auto input_tensor_shape = perf_data.op_model.op_shape.inputs[operand_index];
                threshold = input_tensor_shape.rt * input_tensor_shape.ct;
                input_size = threshold * 2;
            }


            //input_size *= 1024; // TEST

            ibs.push_back(new Buffer(
                node,
                operand_index,
                input_size,
                threshold,
                broadcast_multiplier));

            SIMLOG << "Node " << node->get_name() << ", operand " << operand_index << " input buffer: "
                   << " input_size: " << input_size << ", threshold: " << threshold << " ("
                   << input_block_shape.buffered_rt() << ", " << input_block_shape.buffered_ct()
                   << "), broadcast_x: " << broadcast_multiplier << " grid: " << perf_data.grid.size_r << "x"
                   << perf_data.grid.size_c << ": ";
            SIMLOG << ibs.back()->to_string() << std::endl;
        }

        node_input_buffer_map.insert(std::make_pair(node, ibs));
        return ibs;
    }

    return it->second;
}

Buffer *SimCache::node_input_buffer(NodeP node, std::uint32_t operand_index)
{
    auto input_buffers = node_input_buffers(node);
    TT_ASSERT(operand_index < input_buffers.size());
    return input_buffers.at(operand_index);
}

Buffer *SimCache::node_output_buffer(NodeP node)
{
    auto it = node_output_buffer_map.find(node);
    if (it == node_output_buffer_map.end())
    {
        return create_node_output_buffer(node);
    }

    return it->second;
}

// API to explicitly create an output buffer with given output multiplier
Buffer *SimCache::create_node_output_buffer(NodeP node, std::uint32_t output_mb)
{
    std::uint32_t output_size = node->get_perf_data()->output.size_in_tiles() * output_mb; // default for nodes (like input) that don't have output buffers
    if (node->get_perf_data()->op_perf_data.op_model.output_buffers.size() > 0) 
    {
        output_size = node->get_perf_data()->op_perf_data.op_model.output_buffers.at(0).l1_size_tiles;
        output_size *= node->get_perf_data()->op_perf_data.op_model.grid_shape.volume();
    }

    auto ret = new Buffer(node, output_size);
    node_output_buffer_map.insert(std::make_pair(node, ret));
    return ret;
}

std::uint32_t SimCache::node_output_size_in_tiles(NodeP node)
{
    auto it = node_output_size_map.find(node);
    if (it == node_output_size_map.end())
    {
        std::uint32_t output_size = node->get_perf_data()->output.size_in_tiles();
        node_output_size_map.insert(std::make_pair(node, output_size));
        return output_size;
    }
    return it->second;
}

void Simulator::unstall_dependencies(Buffer *b)
{
    auto it = stalled_events.find(b);
    if (it != stalled_events.end())
    {
        // Find the lowest input for which we have a stalled event. Don't unstall any after it, it's not necessary.
        std::uint32_t lowest_input = UINT32_MAX;
        for (DataEvent *e : it->second)
            if (e->get_input_index() < lowest_input)
                lowest_input = e->get_input_index();

        std::vector<DataEvent *> remaining_events;
        for (DataEvent *e : it->second)
        {
            // TODO
            if (e->get_input_index() > lowest_input + 1)
            {
                remaining_events.push_back(e);
                continue;
            }

            // Erase the event from any other buffer stalls it was on
            for (Buffer *other_b : stalled_events_reverse_map.at(e))
            {
                if (other_b == b)
                    continue;
                auto &v = stalled_events.at(other_b);
                auto it2 = std::find(v.begin(), v.end(), e);
                TT_ASSERT(it2 != v.end());
                v.erase(it2);
            }

            // Not stalled any more
            stalled_events_reverse_map.erase(e);

            add_data_event(e);
            SIMLOG << "  UNSTALL " << e->to_string() << std::endl;
        }

        if (remaining_events.size() > 0)
            stalled_events[b] = remaining_events;
        else
            stalled_events.erase(it);
    }
}

void Simulator::schedule_ops(SimCacheP &cache)
{
    for (NodeP node : graph->get_nodes())
    {
        if (node->is_op())
        {
            add_data_event(new OpDataEvent(
                0,
                TimeData{.count = cache->node_output_size_in_tiles(node), .timestamp = 0},
                cache->node_output_buffer(node),
                0,
                0,
                0));
        }
        else if ((node->get_operands().size() > 0) && (node->get_outputs().size() > 0))
        {
            // Intra-epoch queue
            add_data_event(new QueueDataEvent(
                0,
                TimeData{.count = cache->node_output_size_in_tiles(node), .timestamp = 0},
                cache->node_input_buffer(node, 0)));
        }
    }
}

bool Simulator::run(std::string const& arch_name, std::uint32_t epoch)
{
    if (Simulator::s_write_log)
        s_log.open("simulator.log");

    SIMLOG << "NODES:" << std::endl;
    for (NodeP node : graph->get_nodes())
    {
        SIMLOG << " - " << node->get_name() << std::endl;
        for (NodeP input : node->get_operands()) SIMLOG << "      *-> " << input->get_name() << std::endl;
        for (NodeP output : node->get_outputs()) SIMLOG << "    <-*   " << output->get_name() << std::endl;
    }

    // Cache various lookups that we'll be doing a lot
    auto cache = std::make_unique<SimCache>();

    // Populate host read/write events
    initialize_io(cache, sim_state);

    // Schedule first set of ops
    schedule_ops(cache);

    sim_state->timestamp = 0;

    if (env_as<bool>("PYBUDA_PERF_SIMULATOR_TRACE"))
    {
        for (NodeP node : graph->get_nodes())
        {
            if (!node->is_op())
                continue;

            if (sim_state->trace)
                sim_state->trace_op.emplace(std::pair(
                    node,
                    new TraceOp(
                        node->get_name(),
                        node->get_op_type(),
                        node->get_perf_data()->op_perf_data.grid,
                        DataFormat::Float16_b,
                        {DataFormat::Float16_b})));
        }
    }

    while (event_queue.size() > 0)
    {
        DataEvent *event = pop_data_event();
        if (event->timestamp() > sim_state->timestamp)
            sim_state->timestamp = event->timestamp();

        SIMLOG << "@" << sim_state->timestamp << " Processing: " << event->to_string() << std::endl;

        ProcessStatus ps = event->process(sim_state, cache, arch_name);

        for (Buffer *b : ps.modified_buffers) unstall_dependencies(b);

        for (Buffer *stall_buffer : ps.stall_reason)
        {
            // Record the stall so that we can re-queue this event later
            SIMLOG << "  STALLED on " << stall_buffer->to_string() << std::endl;
            stalled_events[stall_buffer].push_back(event);
            stalled_events_reverse_map[event].push_back(stall_buffer);
        }

        if (ps.stall_reason.size() == 0)
        {
            // Done
            delete event;
        }

        // Schedule new events
        for (DataEvent *new_event : ps.new_events)
        {
            SIMLOG << "  SCHEDULE " << new_event->to_string() << std::endl;
            add_data_event(new_event);
        }
    }

    for (auto &[stalled, bufs] : stalled_events_reverse_map)
    {
        SIMLOG << "** STALLED event: " << stalled->to_string() << std::endl;
        for (Buffer *b : bufs)
        {
            SIMLOG << "  - on buffer: " << b->to_string() << std::endl;
        }
        delete stalled;
    }
    bool ok = stalled_events.size() == 0;
    stalled_events.clear();
    stalled_events_reverse_map.clear();

    for (NodeP node : graph->get_nodes())
    {
        for (Buffer *input : cache->node_input_buffers(node))
            if (!input->empty()) {
                SIMLOG << "** NON-EMPTY BUFFER: " << input->to_string(true) << std::endl;
                ok = false;
            }

        Buffer *output = cache->node_output_buffer(node);
        if (!output->empty()) {
            SIMLOG << "** NON-EMPTY BUFFER: " << output->to_string(true) << std::endl;
            ok = false;
        }
    }
    if (Simulator::s_write_log)
        s_log.close();

    if (ok && sim_state->trace)
    {
        std::ofstream os("perf_postprocess_epoch_" + std::to_string(epoch) + ".json");
        std::vector<std::uint32_t> input_indices = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
        for (std::uint32_t i = input_count - 3; i < input_count; i++)
            if (i > 2)
                input_indices.push_back(i);
        // std::vector<std::uint32_t> input_indices;
        //  for (std::uint32_t input_index = 0; input_index < input_count; input_index++)
        //      input_indices.push_back(input_index);

        os << sim_state->trace_to_json(input_indices);

        os.close();
    }
    return ok;
}

// Populate input/output events
void Simulator::initialize_io(SimCacheP &cache, SimStateP &sim_state)
{
    for (std::uint32_t input_index = 0; input_index < input_count; input_index++)
    {
        for (NodeP input : graph->get_inputs())
        {
            // Input buffer holds the full microbatch
            Buffer *b = cache->create_node_output_buffer(input, sim_state->total_input_count);
            std::uint32_t count = cache->node_output_size_in_tiles(input);
            add_data_event(new HostWriteDataEvent(input_index, TimeData{.count = count, .timestamp = input_index}, b));
        }

        // TODO: for optimizer outputs, we'll read the weights out before they reach the optimizer ops... causing a hang
        for (NodeP output : graph->get_outputs())
        {
            Buffer *b = cache->node_input_buffer(output, 0);
            std::uint32_t count = cache->node_output_size_in_tiles(output);
            add_data_event(new HostReadDataEvent(input_index, TimeData{.count = count, .timestamp = input_index}, b));
        }
    }
}

void Simulator::add_data_event(DataEvent *event) { event_queue.push(event); }

DataEvent *Simulator::pop_data_event()
{
    DataEvent *ret = event_queue.top();
    event_queue.pop();
    return ret;
}

std::string SimState::trace_to_json(const std::vector<std::uint32_t> &input_indices) const
{
    json j;
    for (auto &[node, op] : trace_op)
    {
        op->add_to_json(j, input_indices);
    }

    auto &e = j["per-epoch-events"];

    e["AICLK"] = 1202;
    e["average-input-throughput"] = 230.54;
    e["device-id"] = 0;
    e["last-input-" + std::to_string(input_indices.back()) + "-execution-time"] = 25000;

    auto &p = e["last-pack"];
    for (auto input_index : input_indices)
    {
        p["input_" + std::to_string(input_index)] = {{"core-id", "(x=2, y=2)"}, {"end-timestamp", 25000}};
    }
    auto &u = e["unpack-first-block-available"];
    for (auto input_index : input_indices)
    {
        u["input_" + std::to_string(input_index)] = {{"core-id", "(x=2, y=2)"}, {"timestamp", 0}};
    }

    return j.dump(2);
}

}  // namespace tt::perf_model
