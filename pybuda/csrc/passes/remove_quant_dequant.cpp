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

    // Purge the scale of one before the other. This way if both quant and dequant point to the same scale (directly or indirectly),
    // the first call to purge_scale_graph will do nothing as the scale has multiple users. After the quantize
    // is bypassed, when wecall purge_scale_graph again the sale will be erased as the edge that was once
    // pointing to the quantize is gone (thanks to bypass_node).
    purge_scale_graph(graph->data_operands(quantize)[1]);
    bypass_node(graph, quantize, true);
    purge_scale_graph(graph->data_operands(dequantize)[1]);
    bypass_node(graph, dequantize, true);
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


            // Quantize should be producing an int8
            // if (std::get<std::string>(op_child->op_attrs()[4]) != std::string("torch.int8"))
            //     continue;

            bypass_qdq_pair(graph, op_node, op_child);
            graph_changed = true;
            attempt_update = true;
            break;

        }
    }

    return graph_changed;
}
}