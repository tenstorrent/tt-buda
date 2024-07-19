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
#include "passes/move_dequantize.hpp"
#include "reportify/reportify.hpp"
#include <iostream>
using namespace std;
namespace tt::passes
{

bool dequantize_can_commute_reshape(graphlib::Graph *graph, graphlib::Node *reshape) {
    graphlib::OpNode *deq_node = dynamic_cast<graphlib::OpNode *>(graph->data_operands(reshape)[0]);
    TT_ASSERT(deq_node->op_type().op == "dequantize", "Reshape operand is not dequantize!");

    // If the reshape is equivalent to a squeeze/unsqueeze we should be able to move the dequantize through
    uint32_t before_ones_count = 0;
    uint32_t after_ones_count = 0;
    for (uint32_t dim_size : deq_node->shape().as_vector()) {
        if (dim_size == 1)
            before_ones_count++;
    }
    for (uint32_t dim_size : reshape->shape().as_vector()) {
        if (dim_size == 1)
            after_ones_count++;
    }

    int32_t difference = before_ones_count - after_ones_count;
    return difference == 1 or difference == -1;
}

bool dequantize_can_commute_hslice(graphlib::Graph *graph, graphlib::Node *hslice) {
    graphlib::OpNode *deq_node = dynamic_cast<graphlib::OpNode *>(graph->data_operands(hslice)[0]);
    TT_ASSERT(deq_node->op_type().op == "dequantize", "HSlice operand is not dequantize!");

    int deq_axis = std::get<int>(deq_node->op_attrs()[1]);
    graphlib::OpNode *hslice_op = dynamic_cast<graphlib::OpNode *>(hslice);
    TT_ASSERT(hslice_op, "Expecting an OpNode");
    TT_ASSERT(hslice_op->op_type().op == "hslice", "Expecting an hslice op.");

    // We can swap the dequant and hslice so long as the dequant axis is not the z-dim or c-dim
    bool is_c_dim = deq_axis == (int)deq_node->shape().size()-1 or deq_axis == -1;
    bool is_z_dim = deq_node->shape().size() >= 3 and (deq_axis == (int)deq_node->shape().size()-3 and deq_axis == -3);

    // If the dequant shape has 3 dimensions and the dequant axis is the z-dim, then the z-dim must be zero and
    // the dequant axis can remain on the first dimension (w-dim) as hslice will yield a 4 dimension result
    bool can_commute = (not is_c_dim) and ((not is_z_dim) or deq_node->shape().size() == 3);

    return can_commute;
}

bool dequantize_can_commute_hstack(graphlib::Graph *graph, graphlib::Node *hstack) {
    graphlib::OpNode *deq_node = dynamic_cast<graphlib::OpNode *>(graph->data_operands(hstack)[0]);
    TT_ASSERT(deq_node->op_type().op == "dequantize", "HStack operand is not dequantize!");

    int deq_axis = std::get<int>(deq_node->op_attrs()[1]);
    graphlib::OpNode *hslice_op = dynamic_cast<graphlib::OpNode *>(hstack);
    TT_ASSERT(hslice_op, "Expecting an OpNode");
    TT_ASSERT(hslice_op->op_type().op == "hstack", "Expecting an hstack op.");

    // We can swap the dequant and hslice so long as the dequant axis is not the z-dim or c-dim
    bool is_c_dim = deq_axis == (int)deq_node->shape().size()-1 or deq_axis == -1;
    bool is_z_dim = deq_node->shape().size() >= 3 and (deq_axis == (int)deq_node->shape().size()-3 and deq_axis == -3);

    return (not is_z_dim) and (not is_c_dim);
}

bool op_commutes_dequantize(graphlib::Graph *graph, graphlib::Node *node) {
    /*
    Defines which ops commute dequantize node, meaning which ops can be done in int8 and produce the same output
    */
    graphlib::PyOpNode *op_node = node->as<graphlib::PyOpNode>();
    bool can_commute = op_node->op_type().op == "relu";
    can_commute = can_commute or (op_node->op_type().op == "reshape" and dequantize_can_commute_reshape(graph, op_node));
    can_commute = can_commute or op_node->op_type().op == "transpose";
    can_commute = can_commute or (op_node->op_type().op == "hslice"  and dequantize_can_commute_hslice(graph, op_node));
    can_commute = can_commute or (op_node->op_type().op == "hstack"  and dequantize_can_commute_hstack(graph, op_node));
    return can_commute;
}

tt::graphlib::Node * get_user(graphlib::Graph *graph, tt::graphlib::Node *node) {
    /*
    Gets the first user (consumer) of node
    In case that node has more consumers, returns only first one
    */
    auto users = graph->data_users(node);
    TT_ASSERT(users.size() > 0, "Node has no outputs");
    tt::graphlib::Node *user = users[0];

    return user;
}

tt::graphlib::Edge get_edge_from_parent_to_opnode(graphlib::Graph *graph, tt::graphlib::Node *op_node, tt::graphlib::Node *parent_node) {
    /*
    Returns the op_node_parent -> op_node edge
    */
    std::vector<tt::graphlib::Edge> edges  = graph->operand_data_edges(op_node);
    for (auto edge : edges) {
        if (edge.producer_node_id == parent_node->id())
            return edge;
    }
    return edges[0];
}

void insert_edge(graphlib::Graph *graph, tt::graphlib::NodeId input_node_id, tt::graphlib::PortId input_node_port_id, tt::graphlib::NodeId output_node_id, tt::graphlib::PortId output_node_port_id) {
    tt::graphlib::Edge skip_deq_edge = tt::graphlib::Edge(
        input_node_id, 
        input_node_port_id, 
        output_node_id,
        output_node_port_id, 
        graphlib::EdgeType::kData
    );
    graph->add_edge(skip_deq_edge);
} 

void move_dequant_through_hslice(graphlib::Graph *graph, graphlib::Node *deq_node, graphlib::Node *hslice) {
    TT_ASSERT(dequantize_can_commute_hslice(graph, hslice), "Dequantize cannot commute through hslice");
    graphlib::OpNode *deq_node_op = dynamic_cast<graphlib::OpNode *>(deq_node);
    graphlib::OpNode *hslice_op = dynamic_cast<graphlib::OpNode *>(hslice);
    graphlib::Node *scale = graph->data_operands(deq_node_op)[1];

    std::vector<graphlib::OpType::Attr> deq_attrs = deq_node_op->op_attrs();
    int orig_deq_axis = std::get<int>(deq_attrs[1]);

    int new_deq_axis = orig_deq_axis;
    // If the scale shape volume is 1 then there is no need to change dequant axis
    uint32_t scale_volume = scale->shape().volume();
    if (scale_volume > 1) {
        // If the scale volume is > 1 then the dequant axis must be that which has the same size as the scale
        // Since the scale should only ever be a 1-D vector by this point in compilation, the volume is the size we are looking for
        
        for (uint32_t i = 0; i < hslice_op->shape().size(); i++) {
            if (hslice_op->shape()[i] == scale_volume) {
                new_deq_axis = i;
                break;
            }
        }
    }

    deq_attrs[1] = new_deq_axis;
    deq_node_op->overwrite_op_attrs(deq_attrs);
    deq_node_op->set_shape(hslice_op->shape());

    // Clone hslice op and place on first operand of dequantize
    graphlib::Edge edge = retrieve_between_edge(graph, deq_node, hslice);
    std::string name = hslice->name() + "_dequant_commute_clone" + std::to_string(edge.producer_node_id);
    graphlib::Node *hslice_clone = graph->add_node(hslice->clone(name), graph->get_subgraph_id_for_node(edge.producer_node_id));
    graphlib::Edge input_edge = graph->operand_data_edges(deq_node)[0]; // Values to dequantize are the first input
    insert_node_on_edge(graph, input_edge, hslice_clone);

    bypass_node(graph, hslice, true);
}

void move_dequant_through_hstack(graphlib::Graph *graph, graphlib::Node *deq_node, graphlib::Node *hstack) {
    TT_ASSERT(dequantize_can_commute_hstack(graph, hstack), "Dequantize cannot commute through hstack");
    graphlib::OpNode *deq_node_op = dynamic_cast<graphlib::OpNode *>(deq_node);
    graphlib::OpNode *hstack_op = dynamic_cast<graphlib::OpNode *>(hstack);
    graphlib::Node *scale = graph->data_operands(deq_node_op)[1];

    std::vector<graphlib::OpType::Attr> deq_attrs = deq_node_op->op_attrs();
    int orig_deq_axis = std::get<int>(deq_attrs[1]);

    int new_deq_axis = orig_deq_axis;
    // If the scale shape volume is 1 then there is no need to change dequant axis
    uint32_t scale_volume = scale->shape().volume();
    for (int32_t i = hstack_op->shape().size()-1; i >= 0; i--) {
        if (hstack_op->shape()[i] == scale_volume) {
            new_deq_axis = i;
            break;
        }
    }

    deq_attrs[1] = new_deq_axis;
    deq_node_op->overwrite_op_attrs(deq_attrs);
    deq_node_op->set_shape(hstack_op->shape());

    // Clone hslice op and place on first operand of dequantize
    graphlib::Edge edge = retrieve_between_edge(graph, deq_node, hstack);
    std::string name = hstack->name() + "_dequant_commute_clone" + std::to_string(edge.producer_node_id);
    graphlib::Node *hstack_clone = graph->add_node(hstack->clone(name), graph->get_subgraph_id_for_node(edge.producer_node_id));
    graphlib::Edge input_edge = graph->operand_data_edges(deq_node)[0]; // Values to dequantize are the first input
    insert_node_on_edge(graph, input_edge, hstack_clone);

    bypass_node(graph, hstack, true);
}

void move_dequant_through_reshape(graphlib::Graph *graph, graphlib::Node *deq_node, graphlib::Node *reshape) {
    TT_ASSERT(dequantize_can_commute_reshape(graph, reshape), "Dequantize cannot commute through reshape");

    int32_t min_shape_size = deq_node->shape().size() > reshape->shape().size() ? reshape->shape().size() : deq_node->shape().size();
    graphlib::Edge edge = retrieve_between_edge(graph, deq_node, reshape);
    (void)edge;
    bool is_squeeze = reshape->shape().size() == (uint32_t)min_shape_size;

    // Find which dimension has been added/removed
    int32_t changed_dim = 0;
    bool found_changed_dim = false;
    for (int32_t dim = -1; dim >= -min_shape_size; dim--) {
        if (deq_node->shape()[dim] != reshape->shape()[dim]) {
            found_changed_dim = true;
            changed_dim = dim;
            break;
        }
    }

    if (not found_changed_dim) {
        changed_dim -= 1; // Dim was added/removed at the very front
    }

    graphlib::OpNode *deq_node_op = dynamic_cast<graphlib::OpNode *>(deq_node);
    std::vector<graphlib::OpType::Attr> op_attrs = deq_node_op->op_attrs();
    int32_t deq_axis = std::get<int>(op_attrs[1]);
    
    // Convert dequant axis to positive
    if (deq_axis < 0) {
        deq_axis += deq_node->shape().size();
    }

    if (is_squeeze) {
        if (changed_dim < deq_axis)
            deq_axis -= 1;
    }
    else {
        if (changed_dim <= deq_axis)
            deq_axis += 1;
    }
    op_attrs[1] = deq_axis;

    std::string name = reshape->name() + "_dequant_commute_clone" + std::to_string(edge.producer_node_id);
    graphlib::Node *clone = graph->add_node(reshape->clone(name), graph->get_subgraph_id_for_node(edge.producer_node_id));
    graphlib::OpNode *clone_op = dynamic_cast<graphlib::OpNode *>(clone);

    graphlib::Edge input_edge = graph->operand_data_edges(deq_node)[0]; // Values to dequantize are the first input
    insert_node_on_edge(graph, input_edge, clone);
    clone_op->set_output_df_from_operands(graph);
    bypass_node(graph, reshape, true);

    deq_node_op->set_shape(clone_op->shape());
    deq_node_op->overwrite_op_attrs(op_attrs);
}

void move_dequant_through_transpose(graphlib::Graph *graph, graphlib::Node *deq_node, graphlib::Node *transpose) {
    graphlib::OpNode *deq_node_op = dynamic_cast<graphlib::OpNode *>(deq_node);
    graphlib::Edge edge = retrieve_between_edge(graph, deq_node, transpose);


    std::vector<graphlib::OpType::Attr> deq_attrs = deq_node_op->op_attrs();
    int deq_axis = std::get<int>(deq_attrs[1]);
    if (deq_axis < 0)
        deq_axis += deq_node->shape().size();

    graphlib::OpNode *transpose_op = dynamic_cast<graphlib::OpNode *>(transpose);
    int dim0 = transpose_op->op_type().get_attr_as<int>("dim0");
    if (dim0 < 0)
        dim0 += transpose->shape().size();
    int dim1 = transpose_op->op_type().get_attr_as<int>("dim1");
    if (dim1 < 0)
        dim1 += transpose->shape().size();

    if (dim0 == deq_axis)
        deq_axis = dim1;
    else if (dim1 == deq_axis)
        deq_axis = dim0;

    deq_attrs[1] = deq_axis;

    std::string name = transpose->name() + "_dequant_commute_clone" + std::to_string(edge.producer_node_id);
    graphlib::Node *clone = graph->add_node(transpose->clone(name), graph->get_subgraph_id_for_node(edge.producer_node_id));
    graphlib::OpNode *clone_op = dynamic_cast<graphlib::OpNode *>(clone);

    graphlib::Edge input_edge = graph->operand_data_edges(deq_node)[0]; // Values to dequantize are the first input
    insert_node_on_edge(graph, input_edge, clone);
    clone_op->set_output_df_from_operands(graph);

    // Bypass the transpose node (without removing it) and sway the index of dequantization
    bypass_node(graph, transpose, true);
    deq_node_op->set_shape(clone_op->shape());
    deq_node_op->overwrite_op_attrs(deq_attrs);
}

void move_dequant_through_relu(graphlib::Graph *graph, graphlib::OpNode *deq_node, graphlib::OpNode *relu) {
    // Clone relu op onto the dequantize input and bypass the original relu
    tt::graphlib::Edge deq_data_input_edge  = graph->operand_data_edges(deq_node)[0];
    std::string name = relu->name() + "_dequant_commute_clone" + std::to_string(deq_data_input_edge.producer_node_id);
    graphlib::Node* relu_clone = graph->add_node(relu->clone(name), graph->get_subgraph_id_for_node(deq_data_input_edge.producer_node_id));
    graphlib::OpNode *relu_clone_op = dynamic_cast<graphlib::OpNode *>(relu_clone);

    insert_node_on_edge(graph, deq_data_input_edge, relu_clone);
    relu_clone_op->set_output_df_from_operands(graph);
    relu_clone_op->set_shape(deq_node->shape());
    bypass_node(graph, relu, true);
}

void swap_dequant_and_child(graphlib::Graph *graph, graphlib::OpNode *deq_node, graphlib::OpNode *child) {
    log_debug(LogGraphCompiler, "Swapping {} and {}", deq_node->name(), child->name());
    if (child->op_type().op == "relu")
        move_dequant_through_relu(graph, deq_node, child);
    else if (child->op_type().op == "reshape")
        move_dequant_through_reshape(graph, deq_node, child);
    else if (child->op_type().op == "transpose")
        move_dequant_through_transpose(graph, deq_node, child);
    else if (child->op_type().op == "hslice")
        move_dequant_through_hslice(graph, deq_node, child);
    else if (child->op_type().op == "hstack")
        move_dequant_through_hstack(graph, deq_node, child);

}

bool move_dequantize(graphlib::Graph *graph) {
/*
Moves dequantize op lower in the graph
All nodes that dequantize moves past become quantized meaning they get int32 as input
Currently works for graph with no forks between dequantize current and desired position
*/
    bool attempt_update = true;
    bool graph_changed = false;
    while (attempt_update) {
        attempt_update = false;
        for (tt::graphlib::Node *node : graphlib::topological_sort(*graph)) {
            graphlib::OpNode *deq_node = dynamic_cast<graphlib::OpNode *>(node);

            if (not deq_node)
                continue;

            if (deq_node->op_type().op != "dequantize") 
                continue;

            if (graph->data_users(deq_node).size() != 1)
                continue;

            log_debug(LogGraphCompiler, "Found dequant op: {}", deq_node->name());
            graphlib::OpNode *child = dynamic_cast<graphlib::OpNode *>(graph->data_users(deq_node)[0]);

            if(child and op_commutes_dequantize(graph, child)) {
                log_debug("Commuting {} through {}", deq_node->name(), child->name());
                swap_dequant_and_child(graph, deq_node, child);
                attempt_update = true;
                graph_changed = true;
                break;
            }

        }
    }

    return graph_changed;
}

}
