// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "balancer/balancer.hpp"
#include "placer/placer.hpp"
#include "placer/dram.hpp"
#include "graph_lib/node.hpp"
#include "graph_lib/node_types.hpp"
#include "shared_utils/sparse_matmul_utils.hpp"

namespace tt {

// Lowering context provide an API for Python to create lowered Buda ops, given a PyBuda op and
// its operands.
using Graph = graphlib::Graph;
using Node = graphlib::Node;
using NodeContext = graphlib::NodeContext;
using NodeToNodeMap = std::unordered_map<Node *, Node *>;

class LoweringContext {

    // Old graph
    Graph *old_graph;

    // New graph
    Graph *new_graph;

    // Op being lowered
    graphlib::PyOpNode *node;

    // Mapping of old nodes to new ones
    NodeToNodeMap &old_to_new;

    // Running op index for how many ops were created
    std::uint32_t op_index = 0;
    std::uint32_t constant_index = 0;

    unsigned int subgraph_idx;

public:
    LoweringContext(
            Graph *old_graph, 
            Graph *new_graph, 
            graphlib::PyOpNode *node, 
            NodeToNodeMap &old_to_new) : 
        old_graph(old_graph), new_graph(new_graph), node(node), old_to_new(old_to_new) {

            subgraph_idx = old_graph->get_subgraph_id_for_node(node->id());
        }

    // Op / edge creation
        NodeContext op(
            graphlib::OpType const& op_type,
            std::vector<NodeContext> const& operands,
            std::string const& tag = "",
            int tile_height = graphlib::Shape::BUDA_TILE_DIM,
            int tile_width = graphlib::Shape::BUDA_TILE_DIM);
        NodeContext tm(graphlib::OpType const &op_type, NodeContext const &operand);
        NodeContext nary_tm(graphlib::OpType const &op_type, std::vector<NodeContext> const &operands);
        NodeContext constant(float value, std::pair<int, int> rc_dims);
        NodeContext constant_tile(std::vector<float> value);
        NodeContext tensor(std::shared_ptr<void> value, graphlib::Shape shape, DataFormat df = DataFormat::Invalid);
        NodeContext tensor_with_blob(
            std::shared_ptr<void> value,
            graphlib::Shape shape,
            sparse::SparseBUDA sparse_buda,
            DataFormat df = DataFormat::Invalid);

        void set_output_df(NodeContext node, DataFormat df);
        void set_broadcast_dim(NodeContext src, NodeContext dest, int dim, int factor, bool explicit_bcast = false);
        void set_runtime_tensor_transform(NodeContext node, graphlib::RuntimeTensorTransform t);
        std::vector<std::uint32_t> shape(NodeContext node, bool use_new_graph = false) const;
        std::vector<std::uint32_t> pybuda_shape() const;
        Graph *get_old_graph() const { return old_graph; }
        Graph *get_new_graph() const { return new_graph; }
        graphlib::PyOpNode *get_node() const { return node; }
        Node *get_or_insert_node(NodeContext old_node);

       private:
        template <typename NodeT>
        NodeT *lower_node(graphlib::OpType const &op_type, std::vector<NodeContext> const &operands);
};

bool requires_lowering_to_ram(Node *node);

Node *lower_queue(Graph *old_graph, Graph *new_graph, Node *old_node, NodeToNodeMap &old_to_new);

void lower_node(const LoweringContext &lc);

void copy_operand_edges_to_new_graph(
    Graph *old_graph,
    Graph *new_graph,
    Node *old_node,
    Node *new_node,
    const NodeToNodeMap &old_to_new,
    bool control_only = false,
    bool loopback_only = false
);

void lower_edge_tms(Graph *old_graph, Edge &old_edge, std::shared_ptr<graphlib::EdgeAttributes> new_attr);

}
