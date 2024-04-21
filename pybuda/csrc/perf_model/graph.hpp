// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "backend_api/device_config.hpp"
#include "balancer/types.hpp"
#include "graph_lib/defines.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/shape.hpp"

namespace tt
{
namespace perf_model
{

class Node;
using NodeP = std::shared_ptr<Node>;

struct SystemSpec
{
    float clock_period;
    float noc_bw;
    std::vector<float> dram_bw;
    std::uint32_t grid_size_r, grid_size_c;
    std::string arch_name;

    static SystemSpec get_for_device(const DeviceConfig &device_config);
};

struct TensorData
{
    graphlib::Shape shape;
    std::uint32_t t;
    DataFormat df;

    std::uint32_t size_in_bytes() const;
    std::uint32_t size_in_tiles(bool include_z = true) const;
};

struct OpGrid
{
    std::uint32_t loc_r, loc_c;
    std::uint32_t size_r, size_c;
    std::uint32_t size() const { return size_r * size_c; }
};

struct OpPerfData
{
    // Cycle count to produce one output, if input bandwidth is 100% of needed
    OpGrid grid;
    balancer::OpModel op_model;

    // Epoch number and type
    std::uint32_t temporal_epoch;
    graphlib::NodeEpochType epoch_type;

    OpPerfData(
        OpGrid grid, balancer::OpModel op_model, std::uint32_t temporal_epoch, graphlib::NodeEpochType epoch_type) :
        grid(grid), op_model(op_model), temporal_epoch(temporal_epoch), epoch_type(epoch_type)
    {
    }

    OpPerfData() {}

   private:
    bool _has_execution_cycles = false;  // cache because calls are expensive
    bool _has_op_cycle_estimates = false;
    std::uint32_t _cycle_count_ideal;
    std::uint32_t _theoretical_cycles;
    balancer::OpCycleEstimates _op_cycle_estimates;

    void _get_execution_cycles(std::string const &arch_name)
    {
        if (_has_execution_cycles)
            return;
        _cycle_count_ideal = op_model.get_execution_cycles(arch_name);
        _theoretical_cycles = op_model.get_execution_cycles(arch_name, true);
        _has_execution_cycles = true;
    }

    void _get_op_cycle_estimates(
        const DeviceConfig &device_config,
        const graphlib::Graph *graph,
        bool input_queues_on_host,
        bool output_queues_on_host,
        const std::unordered_map<graphlib::Node const *, balancer::OpModel> &selected_op_models);

   public:
    std::uint32_t cycle_count_ideal(std::string const &arch_name)
    {
        _get_execution_cycles(arch_name);
        return _cycle_count_ideal;
    }
    std::uint32_t theoretical_cycles(std::string const &arch_name)
    {
        _get_execution_cycles(arch_name);
        return _theoretical_cycles;
    }
    const balancer::OpCycleEstimates& get_op_cycle_estimates(
        const DeviceConfig &device_config,
        const graphlib::Graph *graph,
        bool input_queues_on_host,
        bool output_queues_on_host,
        const std::unordered_map<graphlib::Node const *, balancer::OpModel> &selected_op_models)
    {
        _get_op_cycle_estimates(
            device_config, graph, input_queues_on_host, output_queues_on_host, selected_op_models);
        return _op_cycle_estimates;
    }
};

struct OpPerfCalculatedData
{
    // BWs - ideal/actual
    std::vector<float> input_bw_needed, input_bw_got;
    float output_bw_perc;  // the percentage of required bw we got (for worst case operand), which is also output bw%
    float output_bw_ideal, output_bw_produced;

    // Cycle counts, utilization
    float utilization;
    std::uint32_t cycle_count_actual;
};

struct QueuePerfData
{
    // Location - dram, host, etc.
    std::string location;
    std::vector<std::uint32_t> dram_channels;
};

struct QueuePerfCalculatedData
{
    float total_read_bw_ideal;  // ideal total BW requested by all consumers
    float write_bw_ideal;       // ideal write BW from the producer

    float total_bw_perc;           // the percentage of requested bw that we can get from dram
    float total_read_bw_produced;  // actual BW that can be given to the op
    float write_bw_received;       // actual write BW from the producer
};

struct Attr
{
    // number of inner-dim blocks (typically only for matmul)
    std::uint32_t m_k;
    std::uint32_t u_kt;
};

struct PerfData
{
    bool is_op;
    Attr attr;  // general attributes, only applicable to some ops/queues

    // Input/output shapes
    std::vector<TensorData> inputs;
    std::vector<std::uint32_t> input_broadcast_multiplier;
    TensorData output;

    OpPerfData op_perf_data;
    OpPerfCalculatedData op_perf_calculated_data;

    QueuePerfData queue_perf_data;
    QueuePerfCalculatedData queue_perf_calculated_data;

    PerfData(
        std::vector<TensorData> inputs,
        std::vector<std::uint32_t> input_broadcast_multiplier,
        TensorData output,
        const OpPerfData &op_perf_data) :
        is_op(true),
        inputs(inputs),
        input_broadcast_multiplier(input_broadcast_multiplier),
        output(output),
        op_perf_data(op_perf_data)
    {
    }
    PerfData(std::vector<TensorData> inputs, TensorData output, const QueuePerfData &queue_perf_data) :
        is_op(false), inputs(inputs), output(output), queue_perf_data(queue_perf_data)
    {
    }
};

enum NodeType
{
    OP,
    QUEUE
};

using PerfDataP = std::shared_ptr<PerfData>;
class Node
{
   private:
    std::string name;
    NodeType type;

    std::vector<NodeP> operands;
    std::vector<NodeP> outputs;

    std::string op_type;
    graphlib::QueueNodeType queue_type;

    PerfDataP perf_data;

   public:
    Node(const std::string &name, const std::string &op_type, const std::vector<NodeP> &operands, PerfDataP perf_data);
    Node(const std::string &name, graphlib::QueueNodeType queue_type, NodeP operand, PerfDataP perf_data);

    void add_output(NodeP node) { outputs.push_back(node); }

    const std::vector<NodeP> &get_operands() const { return operands; }
    const std::vector<NodeP> &get_outputs() const { return outputs; }
    std::string get_name() const { return name; }

    bool is_op() const { return type == NodeType::OP; }
    std::string get_op_type() const
    {
        TT_ASSERT(is_op());
        return op_type;
    }

    bool is_queue() const { return type == NodeType::QUEUE; }
    graphlib::QueueNodeType get_queue_type() const
    {
        TT_ASSERT(is_queue());
        return queue_type;
    }

    PerfDataP get_perf_data() const { return perf_data; }

    // Find which input is fed by node
    std::size_t get_operand_index(const NodeP node, std::uint32_t start = 0) const;
};

class Graph
{
   private:
    std::vector<NodeP> nodes;
    std::vector<NodeP> inputs;

   public:
    Graph() {}
    NodeP add_op(
        const std::string &name,
        const std::string &op_type,
        const std::vector<NodeP> &operands,
        PerfDataP perf_data,
        bool is_input);
    NodeP add_queue(
        const std::string &name, graphlib::QueueNodeType queue_type, NodeP operand, PerfDataP perf_data, bool is_input);
    const std::vector<NodeP> &get_nodes() const { return nodes; }
    const std::vector<NodeP> &get_inputs() const { return inputs; }
    std::vector<NodeP> get_outputs() const;

   private:
    // Add a created node
    NodeP add_node(NodeP node, bool input);

    // Find the longest op in the graph, and its length
    std::pair<NodeP, std::uint32_t> get_longest_op() const;
};

}  // namespace perf_model
}  // namespace tt
