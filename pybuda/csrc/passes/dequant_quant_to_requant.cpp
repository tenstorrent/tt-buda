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
#include "passes/dequant_quant_to_requant.hpp"
#include "reportify/reportify.hpp"

namespace tt::passes 
{

void replace_dq_q_with_req(graphlib::Graph *graph, graphlib::OpNode *dequantize, graphlib::OpNode *quantize) {
    TT_ASSERT(dequantize->op_type().op == "dequantize" and quantize->op_type().op == "quantize", "Improper ops passed.");
    TT_ASSERT(graph->data_users(dequantize).size() == 1, "Only support dequant with one child, quantize");

    // The requantize axis should be the axis which contains the size equal to the max of the scale sizes between quantize and dequant
    graphlib::Node *deq_scale = graph->data_operands(dequantize)[1];
    graphlib::Node *q_scale = graph->data_operands(quantize)[1];

    int max_size = deq_scale->shape()[0] > q_scale->shape()[0] ? deq_scale->shape()[0] : q_scale->shape()[0];

    int requant_axis = -1;
    for (int i = (int)quantize->shape().size()-1; i >= 0; i--) {
        if ((int)quantize->shape()[i] == max_size) {
            requant_axis = i;
            break;
        }
    }
    TT_ASSERT(requant_axis >= 0, "Requant axis should have been set");

    std::vector<graphlib::OpType::Attr> requant_attrs{0.0f, 0.0f, requant_axis, true, std::string("torch.int8")};

    for (graphlib::Edge consumer_edge : graph->user_data_edges(quantize)) {
        std::string name = dequantize->name() + "_" + quantize->name() + "_combined_requantize_" + std::to_string(consumer_edge.edge_creation_id);
        graphlib::OpNode *requant = graph->add_node<graphlib::OpNode>(graphlib::create_node<graphlib::PyOpNode>(name, "requantize"),
                                        graph->get_subgraph_id_for_node(quantize->id()));

        requant->overwrite_op_attrs(requant_attrs);
        
        requant->set_shape(quantize->shape());
        insert_node_on_edge(graph, consumer_edge, requant);
        graph->add_edge(deq_scale, requant);
        graph->add_edge(q_scale, requant);
        requant->set_output_df(tt::DataFormat::Int8);
    }

    // Remove scale edges so that bypass node works (it requires that the node has one operand)
    graphlib::Edge old_deq_scale_edge = retrieve_between_edge(graph, deq_scale, dequantize);
    graphlib::Edge old_q_scale_edge = retrieve_between_edge(graph, q_scale, quantize);
    graph->remove_edge(old_deq_scale_edge);
    graph->remove_edge(old_q_scale_edge);

    bypass_node(graph, dequantize, true);
    bypass_node(graph, quantize, true);

}

bool dequant_quant_to_requant(graphlib::Graph *graph) {
    
    bool attempt_update = true;
    bool graph_changed = false;
    while (attempt_update) {
        attempt_update = false;
        for (tt::graphlib::Node *node : graphlib::topological_sort(*graph)) {
            graphlib::OpNode *op_node = dynamic_cast<graphlib::OpNode *>(node);
            if (not op_node)
                continue;
            
            if (graph->data_users(op_node).size() != 1)
                continue;

            graphlib::OpNode *op_child = dynamic_cast<graphlib::OpNode *>(graph->data_users(op_node)[0]);
            if (not op_child)
                continue;
            
            // Dequantize should only have one user edge going into the dequantize
            graphlib::Edge user_edge = graph->user_data_edges(op_node)[0];
            if (graph->get_edge_attributes(user_edge)->get_tms().size() > 0)
                continue;

            // Must be a dequantize followed by a quantize
            if (op_node->op_type().op != "dequantize" or op_child->op_type().op != "quantize")
                continue;


            // Quantize should be producing an int8
            // if (std::get<std::string>(op_child->op_attrs()[4]) != std::string("torch.int8"))
            //     continue;

            replace_dq_q_with_req(graph, op_node, op_child);
            graph_changed = true;
            attempt_update = true;
            break;

        }
    }

    return graph_changed;
}
}