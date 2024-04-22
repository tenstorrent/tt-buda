// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "perf_model/graph.hpp"
#include "balancer/policies/policy_utils.hpp"

namespace tt::perf_model
{

SystemSpec SystemSpec::get_for_device(const DeviceConfig &device_config)
{
    // Placeholder until DeviceConfig has it
    if (device_config.is_grayskull())
    {
        return SystemSpec{
            .clock_period = 1 / (1.2 * 1000000000),
            .noc_bw = 1,                                  // TODO
            .dram_bw = {10, 10, 10, 10, 10, 10, 10, 10},  // bytes/s
            .grid_size_r = 10,
            .grid_size_c = 12,
            .arch_name = device_config.arch_name,
        };
    }

    // wormhole and blackhole flavours
    return SystemSpec{
        .clock_period = 1 / (1.2 * 1000000000),
        .noc_bw = 1,                          // TODO
        .dram_bw = {60, 60, 60, 60, 60, 60},  // bytes/s
        .grid_size_r = 10,
        .grid_size_c = 8,
        .arch_name = device_config.arch_name,
    };
}

std::uint32_t TensorData::size_in_bytes() const
{
    std::uint32_t size = shape.volume();
    switch (df)
    {
        case DataFormat::Bfp2:
        case DataFormat::Bfp2_b: return size / 4;

        case DataFormat::Bfp4:
        case DataFormat::Bfp4_b: return size / 2;

        case DataFormat::Int8:
        case DataFormat::Bfp8:
        case DataFormat::Bfp8_b:
        case DataFormat::Lf8: return size;

        case DataFormat::UInt16:
        case DataFormat::Float16:
        case DataFormat::Float16_b: return size * 2;

        case DataFormat::Float32: return size * 4;
        case DataFormat::Int32: return size * 4;

        default: return size * 4;  // anything else?
    }
}

std::uint32_t TensorData::size_in_tiles(bool include_z) const
{
    auto out = shape.volume() / (32 * 32);
    if (!include_z)
        out /= shape.z();
    return out;
}

Node::Node(
    const std::string &name, const std::string &op_type, const std::vector<NodeP> &operands, PerfDataP perf_data) :
    name(name), type(NodeType::OP), operands(operands), op_type(op_type), perf_data(perf_data)
{
}

Node::Node(const std::string &name, graphlib::QueueNodeType queue_type, NodeP operand, PerfDataP perf_data) :
    name(name), type(NodeType::QUEUE), queue_type(queue_type), perf_data(perf_data)
{
    if (operand == nullptr)
        operands = {};
    else
        operands = {operand};
}

std::size_t Node::get_operand_index(const NodeP node, std::uint32_t start) const
{
    for (std::size_t i = start; i < operands.size(); i++)
    {
        if (operands[i] == node)
            return i;
    }
    TT_THROW("Not an operand");
    return 0;  // avoid warning
}

NodeP Graph::add_node(NodeP node, bool is_input)
{
    nodes.push_back(node);
    if (is_input)
        inputs.push_back(node);

    for (NodeP operand : node->get_operands()) operand->add_output(node);

    return node;
}

NodeP Graph::add_op(
    const std::string &name,
    const std::string &op_type,
    const std::vector<NodeP> &operands,
    PerfDataP perf_data,
    bool is_input)
{
    NodeP node = std::make_shared<Node>(name, op_type, operands, perf_data);
    return add_node(node, is_input);
}

NodeP Graph::add_queue(
    const std::string &name, graphlib::QueueNodeType queue_type, NodeP operand, PerfDataP perf_data, bool is_input)
{
    NodeP node = std::make_shared<Node>(name, queue_type, operand, perf_data);
    return add_node(node, is_input);
}

// Find the longest op in the graph, and its length
std::pair<NodeP, std::uint32_t> Graph::get_longest_op() const
{
    std::uint32_t max_len = 0;
    NodeP max_op = nullptr;

    for (NodeP node : nodes)
    {
        std::uint32_t cycle_count = node->get_perf_data()->op_perf_calculated_data.cycle_count_actual;
        if ((max_op == nullptr) || (cycle_count > max_len))
        {
            max_op = node;
            max_len = cycle_count;
        }
    }

    return std::make_pair(max_op, max_len);
}

// Find outputs
std::vector<NodeP> Graph::get_outputs() const
{
    std::vector<NodeP> ret;
    for (NodeP node : get_nodes())
        if (node->get_outputs().size() == 0)
            ret.push_back(node);
    return ret;
}

void OpPerfData::_get_op_cycle_estimates(
    const DeviceConfig &device_config,
    const graphlib::Graph *graph,
    bool input_queues_on_host,
    bool output_queues_on_host,
    const std::unordered_map<graphlib::Node const *, balancer::OpModel> &selected_op_models)
{
    if (_has_op_cycle_estimates)
        return;

    _op_cycle_estimates = get_op_cycles_estimates(
        op_model, 
        graph, 
        device_config, 
        input_queues_on_host, 
        output_queues_on_host, 
        0 /* dram_access_core_count */,
        0 /* pcie_access_core_count */,
        nullptr /* current_epoch_nodes */,
        false /* invalidate_cached */,
        &selected_op_models);
    _has_op_cycle_estimates = true;
}

}  // namespace tt::perf_model
