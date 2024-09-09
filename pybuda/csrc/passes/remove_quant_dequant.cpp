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

void bypass_qdq_pair(graphlib::Graph *graph, graphlib::OpNode *quantize, graphlib::OpNode *dequantize) {
    TT_ASSERT(quantize->op_type().op == "quantize" and dequantize->op_type().op == "dequantize", "Improper ops passed.");
    TT_ASSERT(graph->data_users(dequantize).size() == 1, "Only support dequant with one child, quantize");

    // Purge the graph of all nodes that solely feed the scales of the quantize or dequantize
    auto purge_scale_graph = [graph](graphlib::Node *scale) {
        std::vector<graphlib::Node *> nodes_to_check{scale};
        std::vector<graphlib::Node *> nodes_to_remove;
        while (nodes_to_check.size() > 0) {
            graphlib::Node* to_check = nodes_to_check.back();
            nodes_to_check.pop_back();

            if (graph->data_users(to_check).size() > 1) {
                continue;
            } else {
                for (graphlib::Node *operand : graph->data_operands(to_check))
                    nodes_to_check.push_back(operand);
                nodes_to_remove.push_back(to_check);
            }
        }
        for (graphlib::Node *node : nodes_to_remove) {
            graph->remove_node(node);
        }
    };

    graphlib::TaggedNode *quant_scale = graph->data_operands(quantize)[1]->as<graphlib::TaggedNode>();
    graphlib::TaggedNode *dequant_scale = graph->data_operands(dequantize)[1]->as<graphlib::TaggedNode>();

    // If we can be certain that the scales have the same value then we can just drop them from the graph
    bool scales_are_same_node = quant_scale == dequant_scale;
    bool can_drop_scales = scales_are_same_node;
    if (not can_drop_scales) {
        if (quant_scale->has_tag("forked_from") and dequant_scale->has_tag("forked_from")) {
            can_drop_scales = quant_scale->tag_value("forked_from") == dequant_scale->tag_value("forked_from");
        }
        else {
            can_drop_scales = false;
        }
    }

    if (can_drop_scales) {
        // Purge the scale of one before the other. This way if both quant and dequant point to the same scale (directly or indirectly),
        // the first call to purge_scale_graph will do nothing as the scale has multiple users. After the quantize
        // is bypassed, when wecall purge_scale_graph again the sale will be erased as the edge that was once
        // pointing to the quantize is gone (thanks to bypass_node).
        purge_scale_graph(graph->data_operands(quantize)[1]);
        bypass_node(graph, quantize, true);
        purge_scale_graph(graph->data_operands(dequantize)[1]);
        bypass_node(graph, dequantize, true);
    } else {
        // If we cannot be certain that the scales are equal. Then we must divide the dequant scale by the quant scale and multiply
        // the activations.
        graphlib::Edge dequant_scale_edge = retrieve_between_edge(graph, dequant_scale, dequantize);
        graphlib::Edge quant_scale_edge = retrieve_between_edge(graph, quant_scale, quantize);

        std::string quant_scale_recip_name = "quantize_scale_reciprocal_" + quant_scale->name();
        graphlib::OpNode *quant_scale_recip = graph->add_node<graphlib::OpNode>(graphlib::create_node<graphlib::PyOpNode>(quant_scale_recip_name, "reciprocal"), 
                                            graph->get_subgraph_id_for_node(quantize->id()));

        graph->add_edge(quant_scale, quant_scale_recip);
        quant_scale_recip->set_shape(quant_scale->shape());
        quant_scale_recip->set_output_df_from_operands(graph);

        std::string scale_multiply_name = "multiply_scales_" + quant_scale->name() + "_" + dequant_scale->name();
        graphlib::OpNode *scale_multiply = graph->add_node<graphlib::OpNode>(graphlib::create_node<graphlib::PyOpNode>(scale_multiply_name, "multiply"), 
                                            graph->get_subgraph_id_for_node(quantize->id()));

        uint32_t max_scale_shape = std::max<uint32_t>(dequant_scale->shape()[0], quant_scale->shape()[0]);
        graphlib::Shape scale_miltiply_shape = graphlib::Shape::create(std::vector<uint32_t>{max_scale_shape});

        graph->add_edge(dequant_scale, scale_multiply);
        graph->add_edge(quant_scale_recip, scale_multiply);
        scale_multiply->set_output_df_from_operands(graph);
        scale_multiply->set_shape(dequant_scale->shape());

        // Potentially add broadcast on scale edge if one of the scales is not shaped [1]
        if (dequant_scale->shape()[0] != quant_scale->shape()[0]) {
            TT_ASSERT(dequant_scale->shape()[0] == 1 or quant_scale->shape()[0] == 1, "Cannot multiply differently shaped tensors if the dim of one of them is not 1");

            if (dequant_scale->shape()[0] > quant_scale->shape()[0]) {
                graphlib::Edge edge = retrieve_between_edge(graph, dequant_scale, scale_multiply);
                graph->get_edge_attributes(edge)->set_broadcast_dim(-1, max_scale_shape);
            }
            else {
                graphlib::Edge edge = retrieve_between_edge(graph, quant_scale_recip, scale_multiply);
                graph->get_edge_attributes(edge)->set_broadcast_dim(-1, max_scale_shape);
            }
        }

        // Now create the op which multiplies the scales with the bias
        std::string bias_multiply_name = "bias_qdq_bypass_scale_multiply_" + quantize->name() + "_" + dequantize->name();
        graphlib::OpNode *bias_multiply = graph->add_node<graphlib::OpNode>(graphlib::create_node<graphlib::PyOpNode>(bias_multiply_name, "multiply"), 
                                            graph->get_subgraph_id_for_node(quantize->id()));

        graphlib::Node *bias = graph->data_operands(quantize)[0];
        graph->add_edge(scale_multiply, bias_multiply);
        bias_multiply->set_shape(bias->shape());

        graphlib::Edge bias_quant_edge = retrieve_between_edge(graph, bias, quantize);
        insert_node_on_edge(graph, bias_quant_edge, bias_multiply);
        bias_multiply->set_output_df_from_operands(graph);
        
        graph->remove_edge(dequant_scale_edge);
        graph->remove_edge(quant_scale_edge);
        bypass_node(graph, dequantize, true);
        bypass_node(graph, quantize, true);
    }
}

bool remove_quant_dequant(graphlib::Graph *graph) {
    
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
            if (op_node->op_type().op != "quantize" or op_child->op_type().op != "dequantize")
                continue;

            bypass_qdq_pair(graph, op_node, op_child);
            graph_changed = true;
            attempt_update = true;
            break;

        }
    }

    return graph_changed;
}
}