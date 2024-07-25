// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>

#include "graph_lib/node_types.hpp"
#include "graph_lib/utils.hpp"
#include "utils/logger.hpp"
#include "python_bindings_common.hpp"
#include "graph_lib/node.hpp"
#include "graph_lib/graph.hpp"
#include "passes/make_quantized_ops.hpp"
#include "reportify/reportify.hpp"
#include <iostream>

namespace tt::passes
{

bool is_quantizeable_matmul(graphlib::Graph *graph, graphlib::Node *matmul) {
    
    graphlib::OpNode *matmul_op = dynamic_cast<graphlib::OpNode *>(matmul);
    if (not matmul_op)
        return false;

    if (matmul_op->op_type().op != "matmul")
        return false;

    // Both inputs must be dequantize nodes
    for (graphlib::Node *operand : graph->data_operands(matmul)) {
        graphlib::OpNode *operand_op = dynamic_cast<graphlib::OpNode *>(operand);
        if (not operand_op)
            return false;

        if (operand_op->op_type().op != "dequantize")
            return false;
    }

    return true;
}

bool is_quantizeable_add(graphlib::Graph *graph, graphlib::Node *add) {
    
    graphlib::OpNode *add_op = dynamic_cast<graphlib::OpNode *>(add);
    if (not add_op)
        return false;

    if (add_op->op_type().op != "add")
        return false;

    // Both inputs must be dequantize nodes
    std::vector<graphlib::Node *> scales;
    for (graphlib::Node *operand : graph->data_operands(add_op)) {
        graphlib::OpNode *operand_op = dynamic_cast<graphlib::OpNode *>(operand);
        if (not operand_op)
            return false;

        if (operand_op->op_type().op != "dequantize")
            return false;

        scales.push_back(graph->data_operands(operand_op)[1]);
    }

    // Scales to dequant must be identical
    return scales[0] == scales[1];
}

bool is_quantizeable_conv2d(graphlib::Graph *graph, graphlib::Node *conv2d) {
    graphlib::OpNode *conv_op = dynamic_cast<graphlib::OpNode *>(conv2d);
    if (not conv_op)
        return false;

    if (conv_op->op_type().op != "conv2d")
        return false;

    // All inputs must be dequantize nodes
    for (graphlib::Node *operand : graph->data_operands(conv2d)) {
        graphlib::OpNode *operand_op = dynamic_cast<graphlib::OpNode *>(operand);
        if (not operand_op)
            return false;

        if (operand_op->op_type().op != "dequantize")
            return false;
    }
    
    // If three is no bias then it is quantizeable, since we already know the act and weight are dequantize ops
    if (graph->data_operands(conv2d).size() == 2)
        return true;

    // The scale of the bias dequant must be equal to the product of the scales of the act and weight
    graphlib::OpNode *deq_act = dynamic_cast<graphlib::OpNode *>(graph->data_operands(conv2d)[0]);
    graphlib::OpNode *deq_weight = dynamic_cast<graphlib::OpNode *>(graph->data_operands(conv2d)[1]);
    graphlib::OpNode *deq_bias = dynamic_cast<graphlib::OpNode *>(graph->data_operands(conv2d)[2]);

    graphlib::Node *deq_act_scale = graph->data_operands(deq_act)[1];
    graphlib::Node *deq_weight_scale = graph->data_operands(deq_weight)[1];
    graphlib::Node *deq_bias_scale = graph->data_operands(deq_bias)[1];
    graphlib::OpNode *deq_bias_scale_op = dynamic_cast<graphlib::OpNode *>(deq_bias_scale);
    
    if (not deq_bias_scale_op or deq_bias_scale_op->op_type().op != "multiply")
        return false;

    std::vector<graphlib::Node *> bias_scale_multiply_operands = graph->data_operands(deq_bias_scale_op);

    bool bias_scale_valid = (bias_scale_multiply_operands[0] == deq_act_scale and bias_scale_multiply_operands[1] == deq_weight_scale)
                            or (bias_scale_multiply_operands[1] == deq_act_scale and bias_scale_multiply_operands[0] == deq_weight_scale);

    return bias_scale_valid;
}

void make_quantized_matmul(graphlib::Graph *graph, graphlib::OpNode *matmul) {
    TT_ASSERT(matmul, "Null OpNode pointer given.");
    TT_ASSERT(matmul->op_type().op == "matmul", "OpNode is not matmul");
    TT_ASSERT(is_quantizeable_matmul(graph, matmul), "Matmul is not quantizeable.");

    graphlib::OpNode *deq0 = dynamic_cast<graphlib::OpNode *>(graph->data_operands(matmul)[0]);
    graphlib::OpNode *deq1 = dynamic_cast<graphlib::OpNode *>(graph->data_operands(matmul)[1]);

    graphlib::Node *deq0_scale = graph->data_operands(deq0)[1];
    graphlib::Node *deq1_scale = graph->data_operands(deq1)[1];

    // We convert the dequant axis to to a negative index because the matmul
    // shape size might be larger than the shape of deq1 
    // i.e deq1 - [32, 32], matmul - [1, 1, 32, 32]
    int new_deq_axis = std::get<int>(deq1->op_attrs()[1]);
    if (new_deq_axis >= 0)
        new_deq_axis -= deq1->shape().size();

    // Must multiply the scales of both inputs to create new scale
    std::string scale_multiply_name = matmul->name() + "_multiply_scales_" + deq0_scale->name() + "_" + deq1_scale->name();
    graphlib::OpNode *scale_multiply = graph->add_node<graphlib::OpNode>(graphlib::create_node<graphlib::PyOpNode>(scale_multiply_name, "multiply"), 
                                           graph->get_subgraph_id_for_node(matmul->id()));


    uint32_t max_scale_shape = std::max<uint32_t>(deq0_scale->shape()[0], deq1_scale->shape()[0]);
    graphlib::Shape scale_miltiply_shape = graphlib::Shape::create(std::vector<uint32_t>{max_scale_shape});
    scale_multiply->set_shape(scale_miltiply_shape);

    graph->add_edge(deq0_scale, scale_multiply);
    graph->add_edge(deq1_scale, scale_multiply);
    scale_multiply->set_output_df_from_operands(graph);

    // Potentially add broadcast on scale edge if one of the scales is not shaped [1]
    if (deq0_scale->shape()[0] != deq1_scale->shape()[0]) {
        TT_ASSERT(deq0_scale->shape()[0] == 1 or deq1_scale->shape()[0] == 1, "Cannot multiply differently shaped tensors if the dim of one of them is not 1");

        if (deq0_scale->shape()[0] > deq1_scale->shape()[0]) {
            graphlib::Edge edge = retrieve_between_edge(graph, deq0_scale, scale_multiply);
            graph->get_edge_attributes(edge)->set_broadcast_dim(-1, max_scale_shape);
        }
        else {
            graphlib::Edge edge = retrieve_between_edge(graph, deq1_scale, scale_multiply);
            graph->get_edge_attributes(edge)->set_broadcast_dim(-1, max_scale_shape);
        }
    }

    // Make dequant axis positive again, this time using matmul shape
    // as that is the new input to dequant.
    if (new_deq_axis < 0)
        new_deq_axis += matmul->shape().size();

    // Add dequantize node after matmul for all consumer edges
    std::vector<graphlib::OpType::Attr> dequant_attrs{0.0f, new_deq_axis};

    for (graphlib::Edge consumer_edge : graph->user_data_edges(matmul)) {
        std::string dequant_name = "dequantize_post_matmul_" + std::to_string(consumer_edge.edge_creation_id);
        graphlib::OpNode *dequant = graph->add_node<graphlib::OpNode>(graphlib::create_node<graphlib::PyOpNode>(dequant_name, "dequantize"),
                                        graph->get_subgraph_id_for_node(matmul->id()));
        dequant->overwrite_op_attrs(dequant_attrs);
        dequant->set_shape(matmul->shape());
        insert_node_on_edge(graph, consumer_edge, dequant);
        graph->add_edge(scale_multiply, dequant);
    }

    // Remove scale edges so that bypass node works (it requires that the node has one operand)
    graphlib::Edge old_deq0_scale_edge = retrieve_between_edge(graph, deq0_scale, deq0);
    graphlib::Edge old_deq1_scale_edge = retrieve_between_edge(graph, deq1_scale, deq1);
    graph->remove_edge(old_deq0_scale_edge);
    graph->remove_edge(old_deq1_scale_edge);

    bypass_node(graph, deq0, true);
    bypass_node(graph, deq1, true);
    matmul->set_output_df(DataFormat::Int32);
}

void make_quantized_add(graphlib::Graph *graph, graphlib::OpNode *add) {
    TT_ASSERT(add, "Null OpNode pointer given.");
    TT_ASSERT(add->op_type().op == "add", "OpNode is not add");
    TT_ASSERT(is_quantizeable_add(graph, add), "add is not quantizeable.");

    graphlib::OpNode *deq0 = dynamic_cast<graphlib::OpNode *>(graph->data_operands(add)[0]);
    graphlib::OpNode *deq1 = dynamic_cast<graphlib::OpNode *>(graph->data_operands(add)[1]);

    // We already know from is_quantizeable_add that both dequant nodes share the same scale
    graphlib::Node *scale = graph->data_operands(deq0)[1];

    int new_deq_axis = std::get<int>(deq1->op_attrs()[1]);
    if (new_deq_axis >= 0)
        new_deq_axis = new_deq_axis - deq1->shape().size() + add->shape().size();

    
    std::vector<graphlib::OpType::Attr> dequant_attrs{0.0f, new_deq_axis};
    for (graphlib::Edge consumer_edge : graph->user_data_edges(add)) {
        std::string dequant_name = "dequantize_post_add_" + std::to_string(consumer_edge.edge_creation_id);
        graphlib::OpNode *dequant = graph->add_node<graphlib::OpNode>(graphlib::create_node<graphlib::PyOpNode>(dequant_name, "dequantize"),
                                        graph->get_subgraph_id_for_node(add->id()));
        dequant->overwrite_op_attrs(dequant_attrs);
        dequant->set_shape(add->shape());
        insert_node_on_edge(graph, consumer_edge, dequant);
        graph->add_edge(scale, dequant);
    }

    // Remove scale edges so that bypass node works (it requires that the node has one operand)
    graphlib::Node *deq0_scale = graph->data_operands(deq0)[1];
    graphlib::Node *deq1_scale = graph->data_operands(deq1)[1];
    graphlib::Edge old_deq0_scale_edge = retrieve_between_edge(graph, deq0_scale, deq0);
    graphlib::Edge old_deq1_scale_edge = retrieve_between_edge(graph, deq1_scale, deq1);
    graph->remove_edge(old_deq0_scale_edge);
    graph->remove_edge(old_deq1_scale_edge);

    bypass_node(graph, deq0, true);
    bypass_node(graph, deq1, true);
    add->set_output_df(DataFormat::Int32);
    
}

void make_quantized_conv2d(graphlib::Graph *graph, graphlib::OpNode *conv2d) {
    TT_ASSERT(conv2d, "Null OpNode pointer given.");
    TT_ASSERT(conv2d->op_type().op == "conv2d", "OpNode is not conv2d");
    TT_ASSERT(is_quantizeable_conv2d(graph, conv2d), "conv2d is not quantizeable.");

    graphlib::OpNode *deq_act = dynamic_cast<graphlib::OpNode *>(graph->data_operands(conv2d)[0]);
    graphlib::OpNode *deq_weight = dynamic_cast<graphlib::OpNode *>(graph->data_operands(conv2d)[1]);
    graphlib::OpNode *deq_bias = nullptr;
    if (graph->data_operands(conv2d).size() == 3)
        deq_bias = dynamic_cast<graphlib::OpNode *>(graph->data_operands(conv2d)[2]);

    graphlib::Node *deq_act_scale = graph->data_operands(deq_act)[1];
    graphlib::Node *deq_weight_scale = graph->data_operands(deq_weight)[1];
    graphlib::Node *deq_bias_scale = nullptr;
    if (graph->data_operands(conv2d).size() == 3)
        deq_bias_scale = graph->data_operands(deq_bias)[1];

    // We convert the dequant axis to to a negative index because the conv
    // shape size might be larger than the shape of deq1 
    // i.e deq1 - [32, 32], matmul - [1, 1, 32, 32]
    int new_deq_axis = std::get<int>(deq_weight->op_attrs()[1]);
    if (new_deq_axis >= 0)
        new_deq_axis -= deq_weight->shape().size();

    // Must multiply the scales of both inputs to create new scale
    std::string scale_multiply_name = conv2d->name() + "multiply_scales_" + deq_act_scale->name() + "_" + deq_weight_scale->name();
    graphlib::OpNode *scale_multiply = graph->add_node<graphlib::OpNode>(graphlib::create_node<graphlib::PyOpNode>(scale_multiply_name, "multiply"), 
                                           graph->get_subgraph_id_for_node(conv2d->id()));


    uint32_t max_scale_shape = std::max<uint32_t>(deq_act_scale->shape()[0], deq_weight_scale->shape()[0]);
    graphlib::Shape scale_miltiply_shape = graphlib::Shape::create(std::vector<uint32_t>{max_scale_shape});
    scale_multiply->set_shape(scale_miltiply_shape);

    graph->add_edge(deq_act_scale, scale_multiply);
    graph->add_edge(deq_weight_scale, scale_multiply);
    scale_multiply->set_output_df_from_operands(graph);

    // Potentially add broadcast on scale edge if one of the scales is not shaped [1]
    if (deq_act_scale->shape()[0] != deq_weight_scale->shape()[0]) {
        TT_ASSERT(deq_act_scale->shape()[0] == 1 or deq_weight_scale->shape()[0] == 1, "Cannot multiply differently shaped tensors if the dim of one of them is not 1");

        if (deq_act_scale->shape()[0] > deq_weight_scale->shape()[0]) {
            graphlib::Edge edge = retrieve_between_edge(graph, deq_act_scale, scale_multiply);
            graph->get_edge_attributes(edge)->set_broadcast_dim(-1, max_scale_shape);
        }
        else {
            graphlib::Edge edge = retrieve_between_edge(graph, deq_weight_scale, scale_multiply);
            graph->get_edge_attributes(edge)->set_broadcast_dim(-1, max_scale_shape);
        }
    }

    // Make dequant axis positive again, this time using matmul shape
    // as that is the new input to dequant.
    if (new_deq_axis < 0)
        new_deq_axis += conv2d->shape().size();

    // The dequant axis may be 0 since conv weights may have a w dim
    if (new_deq_axis == 0)
        new_deq_axis = 1;

    // Add dequantize node after matmul for all consumer edges
    std::vector<graphlib::OpType::Attr> dequant_attrs{0.0f, new_deq_axis};

    for (graphlib::Edge consumer_edge : graph->user_data_edges(conv2d)) {
        std::string dequant_name = "dequantize_post_conv2d_" + std::to_string(consumer_edge.edge_creation_id);
        graphlib::OpNode *dequant = graph->add_node<graphlib::OpNode>(graphlib::create_node<graphlib::PyOpNode>(dequant_name, "dequantize"),
                                        graph->get_subgraph_id_for_node(conv2d->id()));
        dequant->overwrite_op_attrs(dequant_attrs);
        dequant->set_shape(conv2d->shape());
        insert_node_on_edge(graph, consumer_edge, dequant);
        graph->add_edge(scale_multiply, dequant);
    }

    // Remove scale edges so that bypass node works (it requires that the node has one operand)
    graphlib::Edge old_deq_act_scale_edge = retrieve_between_edge(graph, deq_act_scale, deq_act);
    graphlib::Edge old_deq_weight_scale_edge = retrieve_between_edge(graph, deq_weight_scale, deq_weight);
    graph->remove_edge(old_deq_act_scale_edge);
    graph->remove_edge(old_deq_weight_scale_edge);
    if (deq_bias) {
        graphlib::Edge old_deq_bias_scale_edge = retrieve_between_edge(graph, deq_bias_scale, deq_bias);
        graph->remove_edge(old_deq_bias_scale_edge);
    }
    bypass_node(graph, deq_act, true);
    bypass_node(graph, deq_weight, true);
    if (deq_bias)
        bypass_node(graph, deq_bias, true);
    conv2d->set_output_df(DataFormat::Int32);
}

const std::array<std::string, 3> quantizeable_ops{
    "matmul",
    "conv2d",
    "add"
};
bool make_quantized_ops(graphlib::Graph *graph) {
    /* 
    * This pass converts the following pattern (also works for conv2d):
    *
    *   dequantize   dequantize               ...          ...
    *       |            |                     |            |
    *        \          /         =====>        \          /
    *         \        /                         \        /
    *           matmul                        matmul (quantized)
    *                                                |
    *                                                |
    *                                            dequantize
    */
    
    bool attempt_update = true;
    bool graph_changed = false;
    while (attempt_update) {
        attempt_update = false;
        for (tt::graphlib::Node *node : graphlib::topological_sort(*graph)) {
            graphlib::OpNode *op_node = dynamic_cast<graphlib::OpNode *>(node);
            if (not op_node)
                continue;
            
            if (std::find(quantizeable_ops.begin(), quantizeable_ops.end(), op_node->op_type().op) == quantizeable_ops.end())
                continue;

            if (is_quantizeable_matmul(graph, op_node)) {
                log_debug(LogGraphCompiler, "Making quantized matmul {}", op_node->name());
                make_quantized_matmul(graph, op_node);
                attempt_update = true;
                graph_changed = true;
                break;
            }
            else if (is_quantizeable_conv2d(graph, op_node)) {
                log_debug(LogGraphCompiler, "Making quantized conv2d {}", op_node->name());
                make_quantized_conv2d(graph, op_node);
                attempt_update = true;
                graph_changed = true;
                break;
            } else if (is_quantizeable_add(graph, op_node)) {
                log_debug(LogGraphCompiler, "Making quantized add {}", op_node->name());
                make_quantized_add(graph, op_node);
                attempt_update = true;
                graph_changed = true;
                break;
            }

        }
    }

    return graph_changed;
}

}