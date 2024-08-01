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
#include "passes/insert_qdq_on_biases.hpp"
#include "reportify/reportify.hpp"


namespace tt::passes
{

bool can_insert_on_conv2d_bias(graphlib::Graph *graph, graphlib::OpNode *conv2d) {
    if (conv2d->op_type().op != "conv2d" and conv2d->op_type().op != "conv2d_transpose")
        return false;

    if (graph->data_operands(conv2d).size() != 3)
        return false;

    // Both act and weight must have a dequant node as input and the bias cannot
    graphlib::OpNode *act = dynamic_cast<graphlib::OpNode *>(graph->data_operands(conv2d)[0]);
    graphlib::OpNode *weight = dynamic_cast<graphlib::OpNode *>(graph->data_operands(conv2d)[1]);
    graphlib::Node *bias = graph->data_operands(conv2d)[2];
    graphlib::OpNode *bias_op = dynamic_cast<graphlib::OpNode *>(bias);

    if ((not act) or (not weight))
        return false;
    //                                                                                       if bias is nullptr then its just a parameter/constant node which is fine.
    bool can_insert = (act->op_type().op == "dequantize") and (weight->op_type().op == "dequantize") and ((not bias_op) or bias_op->op_type().op != "dequantize");
    // bias must be single dim as well
    can_insert = can_insert and bias->shape().size() == 1;
    return can_insert;
}

bool can_insert_on_matmul_bias(graphlib::Graph *graph, graphlib::OpNode *add) {
    if (add->op_type().op != "add")
        return false;

    // One of these must be dequant, other will then be bias
    graphlib::OpNode *lhs = dynamic_cast<graphlib::OpNode *>(graph->data_operands(add)[0]);
    graphlib::OpNode *rhs = dynamic_cast<graphlib::OpNode *>(graph->data_operands(add)[1]);

    graphlib::OpNode *deq;
    graphlib::Node *bias;

    if (lhs and lhs->op_type().op == "dequantize") {
        deq = lhs;
        bias = graph->data_operands(add)[1];
    } 
    else if (rhs and rhs->op_type().op == "dequantize") {
        deq = rhs;
        bias = graph->data_operands(add)[0];
    }
    else {
        // Neither input is dequantize
        return false;
    }

    // The first non-TM op above the dequantize must be a matmul, or else this isnt a matmul bias-add
    py::object eval_module = py::module_::import("pybuda.op.eval.pybuda");
    py::function is_tm = eval_module.attr("is_tm");
    graphlib::OpNode *operand = dynamic_cast<graphlib::OpNode *>(graph->data_operands(deq)[0]);
    while (operand and is_tm(operand->op_type()).cast<bool>() and graph->data_operands(operand).size() == 1) {
        operand = dynamic_cast<graphlib::OpNode *>(graph->data_operands(operand)[0]);
    }
    if (not operand or operand->op_name() != "matmul")
        return false;
    
    // For now, the way we know this is a bias-add is if the dequantize nodes input has output_df Int32 
    // This is because the quantized matmul above returns an Int32.
    graphlib::Node *deq_input = graph->data_operands(deq)[0];
    bool can_insert = deq_input->output_df() == tt::DataFormat::Int32;

    // bias must be single dim as well
    can_insert = can_insert and bias->shape().size() == 1;
    return can_insert;
}

bool insert_qdq_on_matmul_bias(graphlib::Graph *graph, graphlib::OpNode *add) {
    TT_ASSERT(can_insert_on_matmul_bias(graph, add), "Cannot insert qdq on add bias");

    // One of these must be dequant, other will then be bias
    graphlib::OpNode *lhs = dynamic_cast<graphlib::OpNode *>(graph->data_operands(add)[0]);
    graphlib::OpNode *rhs = dynamic_cast<graphlib::OpNode *>(graph->data_operands(add)[1]);

    graphlib::OpNode *deq;
    graphlib::Node *bias;

    
    // Due to the TT_ASSERT at the top of the function, we know one of lhs or rhs must be a dequantize
    bool bias_is_rhs = false;
    if (lhs and lhs->op_type().op == "dequantize") {
        deq = lhs;
        bias = graph->data_operands(add)[1];
        bias_is_rhs = true;
    } 
    else {
        deq = rhs;
        bias = graph->data_operands(add)[0];
        bias_is_rhs = false;
    }

    int axis = std::get<int>(deq->op_attrs()[1]);
    // Insert unsqueezes to to match the rank of add
    handle_change_rank(graph, add);
    if (bias_is_rhs) {
        bias = graph->data_operands(add)[1];
    }
    else {
        bias = graph->data_operands(add)[0];
    }

    graphlib::Node *scale = graph->data_operands(deq)[1];
    // Find matching dim for axis
    for (uint32_t i = 0; i < bias->shape().size(); i++) {
        if (bias->shape()[i] == scale->shape()[0])
            axis = (int)i;
    }

    graphlib::Edge add_bias_edge = retrieve_between_edge(graph, bias, add);
    std::vector<graphlib::OpType::Attr> quant_attrs{0.0f, axis, std::string("torch.int32")};
    std::vector<graphlib::OpType::Attr> dequant_attrs{0.0f, axis};

    std::string quantize_name = "bias_quantize_insert_" + std::to_string(add_bias_edge.edge_creation_id);
    graphlib::OpNode *quantize = graph->add_node<graphlib::OpNode>(graphlib::create_node<graphlib::PyOpNode>(quantize_name, "quantize"), 
                                           graph->get_subgraph_id_for_node(add->id()));

    quantize->set_shape(bias->shape()); // Use bias shape because we place quantize before tms
    quantize->overwrite_op_attrs(quant_attrs);
    quantize->set_output_df(tt::DataFormat::Int32);

    std::string dequantize_name = "bias_dequantize_insert_" + std::to_string(add_bias_edge.edge_creation_id);
    graphlib::OpNode *dequantize = graph->add_node<graphlib::OpNode>(graphlib::create_node<graphlib::PyOpNode>(dequantize_name, "dequantize"), 
                                           graph->get_subgraph_id_for_node(add->id()));

    dequantize->overwrite_op_attrs(dequant_attrs);
    dequantize->set_output_df(tt::DataFormat::Float32);

    auto edge_tms1 = graph->get_edge_attributes(add_bias_edge)->get_tms();
    auto [_, out_edge] = insert_node_on_edge(graph, add_bias_edge, quantize, true, true, 0U, true);
    
    // Raise broadcast tms to op nodes so that the broadcasts can be consteval'ed into the bias
    // We do this because pre-placer may insert a matmul to perform a tile broadcast. But then
    // both inputs would be int32, and we cannot integer matmuls with inputs that are not either
    // int8 or Uint8
    std::vector<graphlib::OpType> tms = graph->get_edge_attributes(out_edge)->get_tms();
    graph->get_edge_attributes(out_edge)->set_tms({});
    auto current_shape = quantize->shape();
    for (uint32_t i = 0; i < tms.size(); i++) {
        auto tm = tms[i];
        TT_ASSERT(tm.op == "broadcast", "TM must be broadcast");
        std::string name = "quantized_bias_insertion_raised_" + tm.op + std::to_string(out_edge.edge_creation_id) + "_"+ std::to_string(i);
        graphlib::OpNode *tm_op = graph->add_node<graphlib::OpNode>(graphlib::create_node<graphlib::PyOpNode>(name, tm.op), graph->get_subgraph_id_for_node(add->id()));
        tm_op->overwrite_op_attrs(tm.attr);
        current_shape[std::get<int>(tm.attr[0])] = std::get<int>(tm.attr[1]);
        tm_op->set_shape(current_shape);
        out_edge = insert_node_on_edge(graph, out_edge, tm_op).second;
        tm_op->set_output_df(tt::DataFormat::Int32);
    }

    insert_node_on_edge(graph, out_edge, dequantize);
    dequantize->set_shape(current_shape); // Use shape_of_operand because we place dequantize after tms


    graph->add_edge(scale, quantize);
    graph->add_edge(scale, dequantize);
    return true;
}


bool insert_qdq_on_conv2d_bias(graphlib::Graph *graph, graphlib::OpNode *conv2d) {
    TT_ASSERT(can_insert_on_conv2d_bias(graph, conv2d), "Cannot insert qdq on conv2d bias");

    // Both act and weight must have a dequant node as input and the bias cannot
    graphlib::OpNode *deq_act = dynamic_cast<graphlib::OpNode *>(graph->data_operands(conv2d)[0]);
    graphlib::OpNode *deq_weight = dynamic_cast<graphlib::OpNode *>(graph->data_operands(conv2d)[1]);
    graphlib::Node *bias = graph->data_operands(conv2d)[2];

    graphlib::Node *deq_act_scale = graph->data_operands(deq_act)[1];
    graphlib::Node *deq_weight_scale = graph->data_operands(deq_weight)[1];

    std::string scale_multiply_name = conv2d->name() + "_multiply_scales_" + deq_act_scale->name() + "_" + deq_weight_scale->name();
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

    std::vector<graphlib::OpType::Attr> quant_attrs{0.0f, (int)0, std::string("torch.int32")};
    std::vector<graphlib::OpType::Attr> dequant_attrs{0.0f, (int)0};
    graphlib::Edge conv_bias_edge = retrieve_between_edge(graph, bias, conv2d);
    std::string quantize_name = "bias_quantize_insert_" + std::to_string(conv_bias_edge.edge_creation_id);
    graphlib::OpNode *quantize = graph->add_node<graphlib::OpNode>(graphlib::create_node<graphlib::PyOpNode>(quantize_name, "quantize"), 
                                           graph->get_subgraph_id_for_node(conv2d->id()));

    quantize->set_shape(bias->shape()); // Use bias shape because we place quantize before tms
    quantize->overwrite_op_attrs(quant_attrs);
    quantize->set_output_df(tt::DataFormat::Int32);

    std::string dequantize_name = "bias_dequantize_insert_" + std::to_string(conv_bias_edge.edge_creation_id);
    graphlib::OpNode *dequantize = graph->add_node<graphlib::OpNode>(graphlib::create_node<graphlib::PyOpNode>(dequantize_name, "dequantize"), 
                                           graph->get_subgraph_id_for_node(conv2d->id()));

    dequantize->set_shape(bias->shape()); // Use shape_of_operand because we place dequantize after tms
    dequantize->overwrite_op_attrs(dequant_attrs);

    auto [_, out_edge] = insert_node_on_edge(graph, conv_bias_edge, quantize);
    insert_node_on_edge(graph, out_edge, dequantize);

    graph->add_edge(scale_multiply, quantize);
    graph->add_edge(scale_multiply, dequantize);
    quantize->set_output_df_from_operands(graph);
    dequantize->set_output_df_from_operands(graph);

    return true;
}

const std::array<std::string, 3> quantizeable_ops{
    "add",
    "conv2d",
    "conv2d_transpose"
};
bool insert_qdq_on_biases(graphlib::Graph *graph) {
    
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

            if (can_insert_on_conv2d_bias(graph, op_node)) {
                log_debug(LogGraphCompiler, "Inserting qdq pair on conv2d {}", op_node->name());
                insert_qdq_on_conv2d_bias(graph, op_node);
                attempt_update = true;
                graph_changed = true;
                break;
            }
            else if (can_insert_on_matmul_bias(graph, op_node)) {
                log_debug(LogGraphCompiler, "Inserting qdq pair on add {}", op_node->name());
                insert_qdq_on_matmul_bias(graph, op_node);
                attempt_update = true;
                graph_changed = true;
                break;
            }

        }
    }

    return graph_changed;
}
}