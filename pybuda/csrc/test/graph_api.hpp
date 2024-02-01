// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "graph_lib/defines.hpp"
#include "graph_lib/graph.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/shape.hpp"
#include "graph_lib/utils.hpp"
#include "passes/passes_utils.hpp"

namespace tt
{
inline tt::graphlib::InputNode* create_input(
    tt::graphlib::Graph& g,
    const std::string& name,
    const tt::graphlib::Shape& shape,
    tt::graphlib::InputNodeType input_type = tt::graphlib::InputNodeType::Activation,
    unsigned int subgraph_id = 0)
{
    auto* input = (input_type == graphlib::InputNodeType::Constant)
                      ? g.add_node(graphlib::create_node<graphlib::ConstantInputNode>(name, 0.0), subgraph_id)
                      : g.add_node(graphlib::create_node<graphlib::InputNode>(name, input_type, false), subgraph_id);
    input->set_shape(shape);
    return input;
}

inline tt::graphlib::OutputNode* create_output(
    tt::graphlib::Graph& g, const std::string& name, const tt::graphlib::Node* input, unsigned int subgraph_id = 0)
{
    // Calculate and set shapes for the graph
    tt::recalculate_shapes(&g);

    // Create output node and connecting edge
    auto* output = g.add_node(graphlib::create_node<graphlib::OutputNode>(name), subgraph_id);
    output->set_shape(input->shape());
    g.add_edge(graphlib::Edge(input->id(), 0, output->id(), 0, graphlib::EdgeType::kData));

    return output;
}

inline void add_operand_edges(
    tt::graphlib::Graph& graph,
    tt::graphlib::Node* node,
    std::vector<tt::graphlib::Node*> ops,
    const std::vector<int>& user_edge_op_id_edge_id = {})
{
    // Connect with user edge if specified
    std::pair<int, int> inserted_node_producer_edges_id;
    if (!user_edge_op_id_edge_id.empty())
    {
        std::vector<tt::graphlib::Edge> edges = graph.user_data_edges(ops[user_edge_op_id_edge_id[0]]);
        auto inserted_node_edges = tt::graphlib::insert_node_on_edge(&graph, edges[user_edge_op_id_edge_id[1]], node);
        inserted_node_producer_edges_id.first = inserted_node_edges.first.producer_node_id;
        inserted_node_producer_edges_id.second = inserted_node_edges.first.consumer_node_id;
    }

    // Create producer edges
    bool insert_prod_edge_skipped = false;
    for (std::uint32_t i = 0; i < ops.size(); ++i)
    {
        if (ops[i] != nullptr)
        {
            if (i > 0)
                TT_ASSERT(ops[i - 1] != nullptr);

            if (!insert_prod_edge_skipped and inserted_node_producer_edges_id.first == ops[i]->id() and
                inserted_node_producer_edges_id.second == node->id())
            {
                insert_prod_edge_skipped = true;
                continue;
            }
            graph.add_edge(tt::graphlib::Edge(ops[i]->id(), 0, node->id(), i, tt::graphlib::EdgeType::kData));
        }
    }
}

template <typename T>
T* add_node(
    tt::graphlib::Graph& graph,
    const std::string& name,
    const tt::graphlib::OpType& op_type,
    std::vector<tt::graphlib::Node*> ops,
    std::vector<int> user_edge_op_id_edge_id = {},
    unsigned int subgraph_id = 0)
{
    // Create node
    auto* node = graph.add_node(tt::graphlib::create_node<T>(name, op_type), subgraph_id);

    add_operand_edges(graph, node, ops, user_edge_op_id_edge_id);

    // Calculate and set node shape if inserted into middle of the predefined graph
    if (!user_edge_op_id_edge_id.empty())
    {
        tt::graphlib::calculate_and_set_node_shape(&graph, node);
    }

    return node;
}

template <typename T>
T* add_node(
    tt::graphlib::Graph& graph,
    const std::string& name,
    const std::string& type,
    std::vector<tt::graphlib::OpType::Attr> op_attrs,
    std::vector<tt::graphlib::Node*> ops,
    std::vector<int> user_edge_op_id_edge_id = {},
    const tt::BudaOpAttrs& buda_op_attrs = {},
    const tt::graphlib::OpType::Attrs& named_attrs = {},
    unsigned int subgraph_id = 0)
{
    return add_node<T>(
        graph, name, graphlib::OpType(type, op_attrs, buda_op_attrs, named_attrs), ops, user_edge_op_id_edge_id, subgraph_id);
}
}  // namespace tt
