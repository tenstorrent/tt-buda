// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "passes/insert_inverse_outside_quantized_region.hpp"
#include "passes/erase_inverse_ops.hpp"

#include <pybind11/pybind11.h>

#include <vector>

#include "graph_lib/node_types.hpp"
#include "graph_lib/utils.hpp"
#include "passes/commute_utils.hpp"
#include "passes/passes_utils.hpp"
#include "reportify/reportify.hpp"
#include "utils/logger.hpp"

namespace tt::passes
{

bool is_op_in_quantized_region(graphlib::OpNode *op)
{
    std::vector<DataFormat> int_types{
        DataFormat::Int32,
        DataFormat::Int8,
        DataFormat::UInt16,
        DataFormat::RawUInt8,
        DataFormat::RawUInt32,
        DataFormat::RawUInt16};
    return std::find(int_types.begin(), int_types.end(), op->output_df()) != int_types.end();
}

static std::vector<graphlib::Node *> find_downward_path_out(graphlib::Graph *graph, graphlib::OpNode *initial_op) {
    std::vector<graphlib::Node *> path;

    graphlib::OpNode *iter = initial_op;

    auto clone_shape = initial_op->shape();
    auto commute_shape = shape_of_only_operand(graph, initial_op);

    bool found_dequantize = false;
    while (not found_dequantize) {
        graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(iter);
        TT_ASSERT(op);

        // For now if there are multiple children then dont commute
        std::vector<graphlib::Edge> user_edges = graph->user_data_edges(op);
        if (user_edges.size() > 1)
            break;

        graphlib::Edge user_edge = user_edges[0];
        
        
        // For now, if there are any edge tms just dont commute
        if (op != initial_op) {
            std::vector<graphlib::OpType> tms = graph->get_edge_attributes(user_edge)->get_tms();
            if (tms.size() > 0) {
                break;
            }
        }


        bool can_commute = can_commute_past_op(op, initial_op, graph, &commute_shape, &clone_shape, false);
        if (not can_commute and op != initial_op) {
            break;
        }
        path.push_back(op);
        if (is_quantization_ops(op)) 
            found_dequantize = true;

        iter = dynamic_cast<graphlib::OpNode *>(graph->node_by_id(user_edge.consumer_node_id));
        if (not iter)
            break;
    }

    if (not found_dequantize)
        path.clear();

    return path;
}

void insert_inverse_pair_below(graphlib::Graph *graph, graphlib::OpNode *transpose_op, std::vector<graphlib::Edge> edges) {

    const graphlib::OpType orig_op_type = transpose_op->op_type();

    for (graphlib::Edge edge : edges) {

        graphlib::Node *operand = graph->node_by_id(edge.producer_node_id);

        const std::string inverse_name = transpose_op->name() + "_quant_remove_clone" + std::to_string(edge.edge_creation_id);
        auto *clone_inverse = graph->add_node(transpose_op->clone(inverse_name), graph->get_subgraph_id_for_node(edge.consumer_node_id));
        graphlib::OpNode *clone_inverse_op = dynamic_cast<graphlib::OpNode *>(clone_inverse);

        clone_inverse_op->op_type().set_attr("dim0", orig_op_type.get_attr("dim1"));
        clone_inverse_op->op_type().set_attr("dim1", orig_op_type.get_attr("dim0"));
        clone_inverse_op->op_type().set_attr("z_dim_slice", orig_op_type.get_attr("z_dim_slice"));
        auto [incoming_edge, outgoing_edge] = insert_node_on_edge(graph, edge, clone_inverse_op);
        clone_inverse_op->set_output_df_from_operands(graph);
        graphlib::Shape clone_inverse_shape = operand->shape();
        clone_inverse_shape[orig_op_type.get_attr_as<int>("dim0")] = operand->shape()[orig_op_type.get_attr_as<int>("dim1")];
        clone_inverse_shape[orig_op_type.get_attr_as<int>("dim1")] = operand->shape()[orig_op_type.get_attr_as<int>("dim0")];
        clone_inverse_op->set_shape(clone_inverse_shape);

        const std::string clone_name = transpose_op->name() + "_quant_remove_clone" + std::to_string(outgoing_edge.edge_creation_id);
        graphlib::Node* clone = graph->add_node(
            transpose_op->clone(clone_name), 
            graph->get_subgraph_id_for_node(edge.consumer_node_id)
        );
        graphlib::OpNode *clone_op = dynamic_cast<graphlib::OpNode *>(clone);
        insert_node_on_edge(graph, outgoing_edge, clone_op);
        clone_op->set_output_df_from_operands(graph);
        graphlib::Shape clone_shape = operand->shape();
        clone_op->set_shape(clone_shape);
    }
    
}

bool insert_inverse_outside_quantized_region(graphlib::Graph *graph)
{
    bool updated_anything = false;
    bool attempt_update = true;

    std::vector<graphlib::Node *> ops_already_checked;

    while (attempt_update)
    {
        attempt_update = false;
        for (auto *node : graphlib::topological_sort(*graph))
        {
            graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(node);

            if (not op)
                continue;

            if (op->op_name() != "transpose")
                continue;

            if (not is_op_in_quantized_region(op))
                continue;

            if (std::find(ops_already_checked.begin(), ops_already_checked.end(), op) != ops_already_checked.end())
                continue;

            std::vector<graphlib::Node*> downward_path = find_downward_path_out(graph, op);

            if (not downward_path.empty()) {
                // Insert inverse pair on all outgoing edges of last node in downward path
                graphlib::Node *last_node = downward_path.back();
                insert_inverse_pair_below(graph, op, graph->user_data_edges(last_node));
                ops_already_checked.push_back(op);
                updated_anything = true;
                attempt_update = true;
                break;
            }

        }
    }
    reportify::dump_graph(graph->name(), "move_transpose", graph);
    return updated_anything;
}
}  // namespace tt::passes