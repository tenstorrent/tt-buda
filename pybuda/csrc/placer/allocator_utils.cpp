// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "backend_api/device_config.hpp"
#include "balancer/types.hpp"
#include "graph_lib/edge.hpp"
#include "graph_lib/graph.hpp"
#include "graph_lib/node.hpp"
#include "graph_lib/node_types.hpp"
#include "placer/host_memory.hpp"
#include "placer/placer.hpp"

namespace tt
{
namespace backend
{
extern int get_io_size_in_bytes(
    const DataFormat data_formati,
    const bool is_untilized,
    const int ublock_ct,
    const int ublock_rt,
    const int mblock_m,
    const int mblock_n,
    const int t,
    const int entries,
    const int tile_height = 32,
    const int tile_width = 32);
extern uint32_t get_next_aligned_address(const uint32_t address);
}  // namespace backend
namespace placer
{

int get_queue_size(const graphlib::QueueNode* node, balancer::BlockShape const& block_shape, bool untilized)
{
    const graphlib::Shape shape = node->shape();

    std::uint32_t queue_size = tt::backend::get_io_size_in_bytes(
        node->output_df(),
        untilized,
        block_shape.ublock.ct,
        block_shape.ublock.rt,
        block_shape.mblock_m,
        block_shape.mblock_n,
        block_shape.t,
        node->get_num_entries());

    return queue_size;
}

graphlib::Node* get_reference_node(const graphlib::Graph* graph, const graphlib::Node* node)
{
    graphlib::Node* ref_node = nullptr;  // Node from which we'll get the placement information

    if (node->node_type() == graphlib::NodeType::kInput and graph->num_users(node))
    {
        ref_node = graph->data_users(node).at(0);
    }

    if ((node->node_type() == graphlib::NodeType::kQueue) || (node->node_type() == graphlib::NodeType::kOutput))
    {
        std::vector<graphlib::Node*> operands = graph->data_operands(node);
        TT_ASSERT(operands.size() == 1, "There can only be exactly one queue writer, not true for " + node->name());
        ref_node = operands[0];
    }
    return ref_node;
}

bool is_queue_already_placed(const PlacerSolution& placer_solution, const graphlib::Node* node)
{
    bool already_placed =
        placer_solution.name_to_queue_placement.find(node->name()) != placer_solution.name_to_queue_placement.end();
    return already_placed;
}

bool is_queue_already_allocated(const PlacerSolution& placer_solution, const graphlib::Node* node)
{
    bool already_allocated = is_queue_already_placed(placer_solution, node) and
                             placer_solution.name_to_queue_placement.at(node->name()).dram_buffers.size() != 0;
    return already_allocated;
}

bool is_input_host_queue(const HostMemoryPlacerConfig& config, const graphlib::Graph* graph, const graphlib::Node* node)
{
    return is_input_host_queue(config.place_input_queues_on_host(), graph, node);
}

bool is_output_host_queue(
    const HostMemoryPlacerConfig& config, const graphlib::Graph* graph, const graphlib::Node* node)
{
    return is_output_host_queue(config.place_output_queues_on_host(), graph, node);
}

bool is_host_queue(
    const HostMemoryPlacerConfig& host_memory_config, const graphlib::Graph* graph, const graphlib::Node* node)
{
    return is_input_host_queue(host_memory_config, graph, node) or
           is_output_host_queue(host_memory_config, graph, node);
}

}  // namespace placer
}  // namespace tt