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

static std::tuple<std::vector<graphlib::Edge>, graphlib::Shape, graphlib::Shape> find_downward_path_out(graphlib::Graph *graph, graphlib::OpNode *initial_op) {
    std::vector<graphlib::Edge> users_outside;

    graphlib::OpNode *iter = initial_op;

    auto clone_shape = initial_op->shape();
    auto commute_shape = shape_of_only_operand(graph, initial_op);

    bool found_dequantize = false;
    while (not found_dequantize) {
        graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(iter);
        TT_ASSERT(op);

        // For now if there are multiple children then dont commute
        std::vector<graphlib::Edge> user_edges = graph->user_data_edges(op);
        if (user_edges.size() > 1 and op->op_name() != "buda_dequantize")
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

        if (op->op_name() == "buda_dequantize") {
            found_dequantize = true;
            for (graphlib::Edge user_edge : user_edges)
                users_outside.push_back(user_edge);
        }

        iter = dynamic_cast<graphlib::OpNode *>(graph->node_by_id(user_edge.consumer_node_id));
        if (not iter)
            break;
    }

    if (not found_dequantize)
        users_outside.clear();

    return std::make_tuple(users_outside, commute_shape, clone_shape);
}

static std::tuple<std::vector<graphlib::Edge>, graphlib::Shape, graphlib::Shape> find_upward_path_out(graphlib::Graph *graph, graphlib::OpNode *initial_op) {
    std::vector<graphlib::Edge> operands_outside;

    graphlib::OpNode *iter = initial_op;

    auto commute_shape = initial_op->shape();
    auto clone_shape = shape_of_only_operand(graph, initial_op);

    bool found_quantize = false;
    while (not found_quantize) {
        graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(iter);
        TT_ASSERT(op);

        // For now if there are multiple children then dont commute
        std::vector<graphlib::Edge> operand_edges = graph->operand_data_edges(op);
        if (operand_edges.size() > 1 and op->op_name() != "buda_quantize")
            break;

        graphlib::Edge operand_edge = operand_edges[0];
        
        // For now, if there are any edge tms just dont commute
        if (op != initial_op) {
            std::vector<graphlib::OpType> tms = graph->get_edge_attributes(operand_edge)->get_tms();
            if (tms.size() > 0) {
                break;
            }
        }

        bool can_commute = can_commute_past_op(op, initial_op, graph, &commute_shape, &clone_shape, true);
        if (not can_commute and op != initial_op) {
            break;
        }

        if (op->op_name() == "buda_quantize") {
            found_quantize = true;
            for (graphlib::Edge operand_edge : operand_edges) {
                // If the operand of this edge is already an inverse to this op, dont bother returning the edge
                graphlib::OpNode *operand = dynamic_cast<graphlib::OpNode *>(graph->node_by_id(operand_edge.producer_node_id));
                if (not (operand and are_compatible_ops(graph, initial_op, operand, &commute_shape)))
                    operands_outside.push_back(operand_edge);
            }
        }
        iter = dynamic_cast<graphlib::OpNode *>(graph->node_by_id(operand_edge.producer_node_id));
        if (not iter)
            break;
    }

    if (not found_quantize)
        operands_outside.clear();

    return std::make_tuple(operands_outside, commute_shape, clone_shape);
}

void insert_inverse_transpose_pair(graphlib::Graph *graph, graphlib::OpNode *transpose_op, std::vector<graphlib::Edge> edges, bool below) {

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
        if (not below)
            clone_inverse_op->tag("dont_erase", true);
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
        if (below)
            clone_op->tag("dont_erase", true);
    }
    
}

void insert_inverse_reshape_pair(graphlib::Graph *graph, graphlib::OpNode *reshape_op, std::vector<graphlib::Edge> edges, graphlib::Shape commute_shape, graphlib::Shape clone_shape, bool below) {
    const graphlib::OpType orig_op_type = reshape_op->op_type();

    for (graphlib::Edge edge : edges) {

        const std::string inverse_name = reshape_op->name() + "_quant_remove_clone" + std::to_string(edge.edge_creation_id);
        auto *clone_inverse = graph->add_node(reshape_op->clone(inverse_name), graph->get_subgraph_id_for_node(edge.consumer_node_id));
        graphlib::OpNode *clone_inverse_op = dynamic_cast<graphlib::OpNode *>(clone_inverse);
        clone_inverse_op->set_shape(commute_shape);
        update_reshape_attr(clone_inverse_op, commute_shape);
        if (not below) {
            clone_inverse_op->tag("dont_erase", true);
        }
        
        auto [incoming_edge, outgoing_edge] = insert_node_on_edge(graph, edge, clone_inverse_op);
        clone_inverse_op->set_output_df_from_operands(graph);


        const std::string clone_name = reshape_op->name() + "_quant_remove_clone" + std::to_string(outgoing_edge.edge_creation_id);
        graphlib::Node* clone = graph->add_node(reshape_op->clone(clone_name), graph->get_subgraph_id_for_node(edge.consumer_node_id));
        graphlib::OpNode *clone_op = dynamic_cast<graphlib::OpNode *>(clone);
        clone_op->set_shape(clone_shape);
        update_reshape_attr(clone_op, clone_shape);
        if (below) {
            clone_op->tag("dont_erase", true);
        }
        insert_node_on_edge(graph, outgoing_edge, clone_op);
        handle_change_rank(graph, clone_op);
        clone_op->set_output_df_from_operands(graph);
        auto *input = dynamic_cast<graphlib::InputNode *>(graph->node_by_id(edge.producer_node_id));
        if (input)
        {
            try_consteval_op(graph, clone_inverse_op, true);
        } else {
            handle_change_rank(graph, clone_inverse_op);
        }
    }
}

void insert_inverse_pair(graphlib::Graph *graph, graphlib::OpNode *op, std::vector<graphlib::Edge> edges, graphlib::Shape commute_shape, graphlib::Shape clone_shape, bool below) {
    if (op->op_name() == "transpose")
        insert_inverse_transpose_pair(graph, op, edges, below);
    else if (op->op_name() == "reshape")
        insert_inverse_reshape_pair(graph, op, edges, commute_shape, clone_shape, below);
    else {
        TT_ASSERT(false, "Invalid Op passed");
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

            if (op->op_name() != "transpose" and op->op_name() != "reshape")
                continue;

            if (not is_op_in_quantized_region(op))
                continue;

            if (std::find(ops_already_checked.begin(), ops_already_checked.end(), op) != ops_already_checked.end())
                continue;
            
            auto user_out_data = find_downward_path_out(graph, op);
            std::vector<graphlib::Edge> user_edges = std::get<0>(user_out_data);
            graphlib::Shape commute_shape = std::get<1>(user_out_data);
            graphlib::Shape clone_shape = std::get<2>(user_out_data);

            if (not user_edges.empty()) {
                // Insert inverse pair on all outgoing edges of last node in downward path
                op->tag("dont_erase", false);

                insert_inverse_pair(graph, op, user_edges, commute_shape, clone_shape, true);
                ops_already_checked.push_back(op);
                updated_anything = true;
                attempt_update = true;
                break;
            }
            else {
                auto operand_out_data = find_upward_path_out(graph, op);
                std::vector<graphlib::Edge> operand_edges = std::get<0>(operand_out_data);
                graphlib::Shape commute_shape = std::get<1>(operand_out_data);
                graphlib::Shape clone_shape = std::get<2>(operand_out_data);

                if (not operand_edges.empty()) {
                    // Insert inverse pair on all outgoing edges of last node in downward path
                    op->tag("dont_erase", false);

                    insert_inverse_pair(graph, op, operand_edges, commute_shape, clone_shape, false);
                    ops_already_checked.push_back(op);
                    updated_anything = true;
                    attempt_update = true;
                    break;
                }
            }

        }
    }
    reportify::dump_graph(graph->name(), "move_transpose", graph);
    return updated_anything;
}
}  // namespace tt::passes