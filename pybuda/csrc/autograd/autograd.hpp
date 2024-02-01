// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <map>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>

#include "graph_lib/graph.hpp"
#include "graph_lib/node.hpp"
#include "graph_lib/node_types.hpp"

namespace py = pybind11;

namespace tt {

namespace autograd2 {

struct __attribute__ ((visibility ("hidden"))) autograd_config {
    bool recompute = false; // Add recompute
    py::object optimizer = py::none();
};

using grad_map = std::unordered_map<tt::graphlib::EdgeUniqueId, bool, EdgeUniqueIdHash>;

using Node = graphlib::Node;
using Graph = graphlib::Graph;
using NodeContext = graphlib::NodeContext;

class __attribute__ ((visibility ("hidden"))) autograd2_engine {

private:
    tt::graphlib::Graph *graph;
    autograd_config config;
    
    // fwd->output gradient producer map
    std::unordered_map<Node *, std::vector<Node *>> fwd_to_out_gradient_map;

public:
    autograd2_engine(Graph *graph, autograd_config config);
    ~autograd2_engine() = default;
    autograd2_engine(const autograd2_engine &other) = delete;

    // Run and return the modified graph
    Graph *run();

    // Create a backward op for the given fwd op's operand
    NodeContext create_op(
        graphlib::OpType type,
        std::vector<NodeContext> operands,
        Node *current_fwd_op,
        int operand_index,
        int created_op_index,
        std::string name_prefix = "",
        bool copy_golden_transforms = true);

    NodeContext create_optimizer_op(
        graphlib::OpType type,
        std::vector<NodeContext> operands,
        Node *current_fwd_op,
        int operand_index,
        int created_op_index,
        std::string name_prefix = "");

    // Create an integer constant used in backward calculations (typically a negative one)
    template <typename T>
    NodeContext create_constant(Node *current_fwd_op, int operand_index, T value, int created_op_index, graphlib::NodeEpochType epoch_type);

    NodeContext create_constant(
        Node *current_fwd_op,
        int operand_index,
        std::shared_ptr<void> tensor,
        graphlib::Shape shape,
        int created_op_index,
        graphlib::NodeEpochType epoch_type);

    NodeContext create_input(
        Node *current_fwd_op,
        int operand_index,
        int created_op_index,
        graphlib::NodeEpochType epoch_type,
        std::string& suffix_identifier,
        std::vector<std::uint32_t> tensor_shape,
        bool copy_consteval_operations,
        bool disable_consteval = false);

    bool contains_bwd_nodes() const;
    const std::map<int, std::vector<Node *>>& get_bwd_nodes(Node *fwd) const;

    // Get pointer to graph being worked on
    Graph *get_graph() const { return graph; }

private:
    // Propagate requires_grad from inputs to all edges of the graph, creating an edge->bool map
    grad_map propagate_requires_grad();
 
    // Create backward instructions, and hook them up accordingly
    void create_backward_graph(const grad_map &requires_grad_map);

    // Register fwd->bwd and bwd->fwd relationship
    void add_fwd_to_bwd_map(Node *fwd, Node *bwd, int operand_index, bool gradient = false);

    void add_fwd_to_optimizer_edge(Node *fwd, Node *opt, int operand_index);

    // Register fwd->out_gradient
    void add_fwd_to_out_gradient_map(Node *fwd, Node *out_gradient);

    // Combine incoming gradients by adding them, and return the new combined node
    Node *combine_incoming_gradients(Node *node);

    // Create optinstructions, and hook them up accordingly
    void create_optimizer_graph();
};

// Structure passed to python while generating backward ops. This allows us to register 
// backward ops in both the graph and autograd engine maps
struct  __attribute__ ((visibility ("hidden"))) autograd_context {

    autograd2_engine *autograd;
    Node *current_fwd_op;
    int operand;
    graphlib::NodeEpochType epoch_type = graphlib::NodeEpochType::Backward;
    int created_op_index = 0; // Incremented to ensure unique names when multiple ops are created

};

}
}
