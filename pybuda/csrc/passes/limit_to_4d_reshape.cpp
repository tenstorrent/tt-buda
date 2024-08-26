// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "passes/erase_consecutive_reshape.hpp"

#include <pybind11/pybind11.h>

#include "graph_lib/node_types.hpp"
#include "graph_lib/utils.hpp"
#include "passes/passes_utils.hpp"
#include "utils/logger.hpp"

namespace tt::passes
{


static bool is_reshape(graphlib::Node const *node)
{
    graphlib::OpNode const *op = dynamic_cast<graphlib::OpNode const *>(node);
    return op and op->op_name() == "reshape";
}

void limit_to_4d_reshape(graphlib::Graph *graph)
{
    std::vector<std::vector<std::uint32_t>> nd_graph_output_shapes = graph->get_ordered_output_shapes();
    for (auto node : graph->nodes())
    {
        if (not is_reshape(node))
            continue;

        auto op_node = dynamic_cast<graphlib::OpNode *>(node);
        auto attr = op_node->op_attrs();
        if (attr.size() <= 4)
            continue;

        auto users = graph->users(node);
        bool feeds_graph_output = false;
        for (auto const &user : users)
        {
            if (user->node_type() == graphlib::NodeType::kOutput)
            {
                feeds_graph_output = true;
                break;
            }
        }

        // Don't change target shape if it feeds a graph output
        if (feeds_graph_output)
            continue;

        bool dims_before_last_4d_are_singleton = true;
        for (long unsigned int i = 0; i < attr.size() - 4; ++i)
        {
            if (std::get<int>(attr[i]) != 1)
            {
                dims_before_last_4d_are_singleton = false;
                break;
            }
        }

        if (dims_before_last_4d_are_singleton) {
            auto new_shape = attr;
            new_shape.erase(new_shape.begin(), new_shape.begin() + attr.size() - 4);
            op_node->overwrite_op_attrs(new_shape);
        } else {
            TT_ASSERT(false, "Don't support reshape with more than 4 non-singleton dimensions");
        }

    }

    // Update node shapes in graph
    recalculate_shapes(graph);

    std::vector<std::vector<std::uint32_t>> _graph_output_shapes = graph->get_ordered_output_shapes();
    TT_ASSERT(nd_graph_output_shapes.size() == _graph_output_shapes.size());

    for (long unsigned int i = 0; i < nd_graph_output_shapes.size(); ++i)
    {
        if (nd_graph_output_shapes[i] == _graph_output_shapes[i])
            continue;

        // Found output shape change, insert reshape before graph output
        auto node = graph->ordered_module_outputs()[i];
        auto operand_edges = graph->operand_edges(node);
        TT_ASSERT(operand_edges.size() == 1);
        auto target_edge = operand_edges[0];
        auto tags = node->as<graphlib::TaggedNode>()->get_tags();

        // Get producer edge TMs
        auto attrs = graph->get_edge_attributes(target_edge);
        std::vector<graphlib::OpType> tms = attrs->get_tms();
    
        std::string name = node->name() + "_reshape";
        graphlib::OpType op_type("reshape");

        // Create reshape node
        auto target_shape = nd_graph_output_shapes[i];
        std::vector<graphlib::OpType::Attr> target_attr;
        for (auto dim : target_shape)
            target_attr.push_back((int)dim);
        op_type.attr = target_attr;
        auto _reshape = graph->add_node(
            std::make_unique<graphlib::PyOpNode>(name, op_type), graph->get_subgraph_id_for_node(node->id()));

        // Set reshape node properties
        _reshape->set_shape(graphlib::Shape::create(nd_graph_output_shapes[i]));
        _reshape->set_output_df(node->output_df());
        _reshape->as<graphlib::TaggedNode>()->add_tags(tags);
        auto [new_in_edge, new_out_edge] =
            graphlib::insert_node_on_edge(graph, target_edge, _reshape);

        // All TMs should always go to input edge
        graph->get_edge_attributes(new_in_edge)->set_tms(tms);

        // Set output shape
        node->set_shape(graphlib::Shape::create(nd_graph_output_shapes[i]));
    }


}



}  // namespace tt::passes
