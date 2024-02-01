// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "forked_dram_inputs.hpp"

namespace std
{
template <>
struct hash<std::pair<uint32_t, tt::balancer::BlockShape>>
{
    std::size_t operator()(const std::pair<uint32_t, tt::balancer::BlockShape> &block_shape) const
    {
        std::size_t seed = 0;
        tt::hash_combine(seed, static_cast<size_t>(block_shape.first));
        tt::hash_combine(seed, hash<tt::balancer::BlockShape>{}(block_shape.second));
        return seed;
    }
};
}  // namespace std

namespace tt::passes
{
std::unordered_map<Edge, Edge> get_forked_dram_inputs(
    bool enable_forked_dram_inputs,
    Graph *graph,
    unordered_map<string, placer::OpPlacement> *name_to_op_placement,
    balancer::OpModelMap *op_model)
{
    if (!enable_forked_dram_inputs)
        return {};

    std::unordered_map<Edge, Edge> forked_dram_input_edges;
    std::vector<Node *> nodes = graphlib::topological_sort(*graph);
    for (Node *node : nodes)
    {
        // Only applies to nodes that are inputs or queues
        if (node->node_type() != graphlib::NodeType::kInput && node->node_type() != graphlib::NodeType::kQueue){
            continue;
        }
        // If it's an input, only apply to inputs where prologue=false
        auto input = dynamic_cast<graphlib::InputNode *>(node);
        if (input && input->is_prologue())
            continue;

        std::vector<Edge> consumer_edges = graph->user_data_edges(node);

        std::unordered_map<std::uint32_t, std::vector<Edge>> per_epoch_edge_map;
        std::unordered_map<std::pair<std::uint32_t, balancer::BlockShape>, std::vector<Edge>> per_block_shape_edge_map;

        // Group consumer edges based on epoch_id
        for (auto &edge : consumer_edges)
        {
            Node *consumer = graph->node_by_id(edge.consumer_node_id);
            auto buda_op = consumer->as<graphlib::BudaOpNode>();
            // If the op is using Sparse MM or Tilize optimization, disallow forked_dram optimization
            if (buda_op->is_tilize() || buda_op->is_sparse_matmul() || buda_op->is_splice())
                continue;
            auto consumer_epoch_id = name_to_op_placement->at(consumer->as<graphlib::BudaOpNode>()->name()).epoch_id();
            per_epoch_edge_map[consumer_epoch_id].push_back(edge);
        }

        // Group consumer edges based identical block_shapes per epoch
        for (const auto &[epoch_id, edges] : per_epoch_edge_map)
        {
            for (auto edge : edges)
            {
                Node *consumer = graph->node_by_id(edge.consumer_node_id);
                balancer::BlockShape consumer_block_shape = op_model->at(consumer->as<graphlib::BudaOpNode>()->name())
                                                                .input_buffers[edge.consumer_input_port_id]
                                                                .block_shape;
                // Check this once again
                per_block_shape_edge_map[std::make_pair(epoch_id, consumer_block_shape)].push_back(edge);
            }
        }
        // Find edges that can reuse DRAM read from other edge
        for (auto &[epoch_id_block_shape, edges] : per_block_shape_edge_map)
        {
            if (edges.size() > 1)
            {
                for (uint idx = 1; idx < edges.size(); idx++)
                {
                    uint32_t epoch_id = epoch_id_block_shape.first;

                    auto is_reachable_epoch = [&epoch_id, &name_to_op_placement](graphlib::Node *n)
                    {
                        if (dynamic_cast<graphlib::BudaOpNode *>(n) == nullptr)
                        {
                            return false;
                        }
                        auto node_epoch_id = name_to_op_placement->at(n->as<graphlib::BudaOpNode>()->name()).epoch_id();
                        return (node_epoch_id == epoch_id);
                    };

                    // check if any data dependency exists between two nodes
                    if (check_producer_consumer(graph, graph->node_by_id(edges[idx].consumer_node_id), graph->node_by_id(edges[0].consumer_node_id), is_reachable_epoch) ||
                        check_producer_consumer(graph, graph->node_by_id(edges[0].consumer_node_id), graph->node_by_id(edges[idx].consumer_node_id), is_reachable_epoch))
                    {
                        continue;
                    }
                    forked_dram_input_edges.insert({edges[idx], edges[0]});
                }
            }
        }
    }
    return forked_dram_input_edges;
}
}  // namespace tt::passes