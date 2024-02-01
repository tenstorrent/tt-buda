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

using Attr = BudaOpAttr;

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




template<typename T>
bool all_have_same_dim_and_shape_stride1(std::vector<T> const &v) {
    if (v.size() == 0) {
        return false;
    }

    return std::all_of(v.begin(), v.end(), [&] (T const &e) {
        auto attrs = dynamic_cast<graphlib::OpNode const *>(e)->op_attrs();
        int dim = std::get<int>(attrs[0]);
        return dim == std::get<int>(dynamic_cast<graphlib::OpNode const *>(v.front())->op_attrs()[0])
                and e->shape() == v.front()->shape()
                and std::get<int>(attrs[3]) == 1;
    });
}

void decompose_nd_reshape_split(graphlib::Graph *graph) {

    for (auto node : graph->nodes())
    {
        // Only consider reshape nodes with last dimension tile_dim aligned
        if (not is_reshape(node) or node->shape()[-1] % graphlib::Shape::BUDA_TILE_DIM != 0)
            continue;

        auto consumers = graph->users(node);
        bool all_consumers_are_index = all_of(consumers.begin(), consumers.end(), [](auto const &consumer) {
            auto op = dynamic_cast<graphlib::OpNode const *>(consumer);
            return op and op->op_name() == "index";
        });

        // All consumers must be index
        if (not all_consumers_are_index)
            continue;

        bool all_index_have_same_dim_and_shape = all_have_same_dim_and_shape_stride1(consumers);

        // All index must have same dim and shape
        if (not all_index_have_same_dim_and_shape)
            continue;

        uint32_t total_index_size = 0;
        int dim = std::get<int>(dynamic_cast<graphlib::OpNode const *>(consumers[0])->op_attrs()[0]);

        for (auto const &consumer : consumers) {
            total_index_size += consumer->shape()[dim];
        }
        bool index_all_channels = total_index_size == node->shape()[dim];
        bool all_index_have_length1 = all_of(consumers.begin(), consumers.end(), [](auto const &consumer) {
            auto op = dynamic_cast<graphlib::OpNode const *>(consumer);
            return std::get<int>(op->op_attrs()[2]) - std::get<int>(op->op_attrs()[1]) == 1;
        });

        // All index must have length 1 and total indexed size must be equal to node dim
        if (not (index_all_channels and all_index_have_length1))
            continue;


        bool all_sequence_consumers_are_squeeze = all_of(consumers.begin(), consumers.end(), [graph] (auto const &consumer) {
            auto users = graph->users(consumer);
            auto shape_before = consumer->shape();
            auto shape_after = users[0]->shape();
            auto op = dynamic_cast<graphlib::OpNode const *>(users[0]);
            return users.size() == 1 and op->op_name() == "reshape" and shape_after.volume() == shape_before.volume()
                    and shape_after.size() + 1 == shape_before.size();
        });

        // All consumers of Index must be reshape nodes that are equivalent to squeeze
        if (not all_sequence_consumers_are_squeeze)
            continue;

        auto reshape_producer = graph->operands(node)[0];

        // Remove reshape node and update index nodes to select
        int new_dim = dim + 1;
        int producer_dim_size = reshape_producer->shape()[new_dim];
        TT_ASSERT(producer_dim_size % total_index_size == 0);
        int new_dim_size = producer_dim_size / total_index_size;

        for (uint32_t i = 0; i < consumers.size(); i++) {
            auto op = dynamic_cast<graphlib::OpNode *>(consumers[i]);

            auto op_type_ = op->op_type();
            TT_ASSERT(op_type_.op == "index");
            auto op_attrs = op->op_attrs();
            op_type_.op = "select";

            std::vector<Attr> new_op_attrs(4);
            new_op_attrs[0] = new_dim;
            new_op_attrs[1] = (int) (std::get<int>(op_attrs[1]) * node->shape()[-1]);
            new_op_attrs[2] = (int) node->shape()[-1];
            new_op_attrs[3] = (int) (consumers.size() * node->shape()[-1]);
            op_type_.attr = new_op_attrs;
            op->change_op_type(op_type_);

            auto producer_shape = reshape_producer->shape().as_vector();
            auto target_shape = graphlib::Shape::create(producer_shape);
            target_shape[new_dim] = new_dim_size;
            op->set_shape(target_shape);
        }

        graphlib::bypass_node(graph, node, true);

        // Update node shapes in graph
        recalculate_shapes(graph);
    }
}

}  // namespace tt::passes
