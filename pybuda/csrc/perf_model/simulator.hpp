// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <fstream>
#include <list>
#include <queue>
#include <unordered_map>

#include "perf_model/event.hpp"
#include "perf_model/graph.hpp"
#include "perf_model/trace.hpp"

#define SIMLOG                  \
    if (Simulator::s_write_log) \
    Simulator::s_log

namespace tt::perf_model
{

// Input buffer keeps track of received data
class Buffer
{
    // Set at creation
    static std::uint32_t s_id;
    std::string unique_id;
    NodeP owner;
    bool input;  // input or output
    std::uint32_t size;

    // Input buffer only
    std::uint32_t operand;
    std::uint32_t threshold;             // how much is needed by the node to process
    std::uint32_t broadcast_multiplier;  // incoming data is multiplied through broadcast

    // Simulation time
    std::uint32_t reserved = 0;
    std::uint32_t occupied = 0;

   public:
    // Input buffer
    Buffer(
        NodeP owner,
        std::uint32_t operand,
        std::uint32_t size,
        std::uint32_t threshold,
        std::uint32_t broadcast_multiplier) :
        owner(owner),
        input(true),
        size(size),
        operand(operand),
        threshold(threshold),
        broadcast_multiplier(broadcast_multiplier)
    {
        TT_LOG_ASSERT(threshold <= size, "Buffer {} threshold ({}) is larger than buffer size ({})", owner->get_name(), threshold, size);
        unique_id = "b" + std::to_string(s_id++);
    }

    // Output buffer
    Buffer(NodeP owner, std::uint32_t size) : owner(owner), input(false), size(size), threshold(0), broadcast_multiplier(0)
    {
        unique_id = "b" + std::to_string(s_id++);
    }

    NodeP get_node() const { return owner; }
    std::uint32_t get_operand() const { return operand; }
    std::uint32_t get_threshold() const { return threshold; }
    std::uint32_t get_broadcast_multiplier() const { return broadcast_multiplier; }
    bool is_input() const { return input; }

    std::uint32_t available_space() const;
    void reserve_space(std::uint32_t count);
    void insert_data(std::uint32_t count);
    void pop_data(std::uint32_t count);
    void pop_threshold();  // pop data, threshold amount
    bool above_threshold() const;
    bool empty() const;

    std::string to_string(bool show_contents = false) const;

    // Process data in the buffer, and return the amount consumed, if any
    std::uint32_t process();
};

// Cache regularly looked up data that requires a bit of calculation
class SimCache
{
   private:
    std::unordered_map<NodeP, std::vector<Buffer *>> node_input_buffer_map;
    std::unordered_map<NodeP, Buffer *> node_output_buffer_map;
    std::unordered_map<NodeP, std::uint32_t> node_output_size_map;
    using OutputMap = std::unordered_map<NodeP, std::vector<std::pair<Buffer *, std::uint32_t>>>;
    OutputMap node_output_map;

   public:
    ~SimCache();
    const std::vector<Buffer *> node_input_buffers(NodeP node);
    Buffer *node_input_buffer(NodeP node, std::uint32_t operand_index);
    Buffer *node_output_buffer(NodeP node);
    Buffer *create_node_output_buffer(NodeP node, std::uint32_t output_mb = 2);
    std::uint32_t node_output_size_in_tiles(NodeP node);
    std::vector<std::pair<Buffer *, std::uint32_t>> node_outputs(NodeP node);
};

// Simulator state
struct SimState
{
    std::uint32_t timestamp;
    std::uint32_t total_input_count;
    bool trace;  // set to generate trace for routeagui
    std::unordered_map<NodeP, TraceOp *> trace_op;

    std::string trace_to_json(const std::vector<std::uint32_t> &input_indices) const;
};

using SimCacheP = std::unique_ptr<SimCache>;
using SimStateP = std::unique_ptr<SimState>;

struct EventComp
{
    bool operator()(const DataEvent *a, const DataEvent *b) { return *b < *a; }
};

using EventQueue = std::priority_queue<DataEvent *, std::vector<DataEvent *>, EventComp>;

// Main simulator class
class Simulator
{
   private:
    // Graph we're modelling
    Graph *graph;
    std::uint32_t input_count;

    // Pending, non-stalled, events to be processed
    EventQueue event_queue;

    // Current state
    SimStateP sim_state;

    // Stalled events, keyed on buffer they are waiting on, as well as a reverse map
    std::unordered_map<Buffer *, std::vector<DataEvent *>> stalled_events;
    std::unordered_map<DataEvent *, std::vector<Buffer *>> stalled_events_reverse_map;

    // Input buffers
    std::unordered_map<NodeP, std::vector<Buffer *>> input_buffers;  // vector (of operands) per node

    // Populate input/output events
    void initialize_io(SimCacheP &cache, SimStateP &sim_state);

    // Schedule first set of ops on all cores
    void schedule_ops(SimCacheP &cache);

    // Add data event to the end of the queue
    void add_data_event(DataEvent *event);

    // Pop the left-most event
    DataEvent *pop_data_event();

    // Re-schedule events that were stalled on this buffer
    void unstall_dependencies(Buffer *b);

   public:
    Simulator(Graph *graph, std::uint32_t input_count, bool trace = false, bool log = false);

    // Temporary logging to a file
    static bool s_write_log;
    static std::ofstream s_log;

    // Run full simulation, return true if completed without a deadlock
    // Epoch number is only used to generated logs and traces
    bool run(std::string const& arch_name, std::uint32_t epoch = 0);

    // Get final timestamp
    std::uint32_t get_timestamp() const { return sim_state->timestamp; }
};

}  // namespace tt::perf_model
