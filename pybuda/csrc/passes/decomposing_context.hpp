// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "balancer/balancer.hpp"
#include "graph_lib/node.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/utils.hpp"
#include "placer/dram.hpp"
#include "placer/placer.hpp"

namespace tt
{

using Graph = graphlib::Graph;
using NodeContext = graphlib::NodeContext;

class DecomposingContext
{
    // Graph that's being decomposed
    Graph* graph;

    // Current node being decomposed
    graphlib::PyOpNode* node_;

    // Running op index for how many ops were created
    std::uint32_t op_index = 0;

    graphlib::NodeId output_node_id = -1;

    std::vector<graphlib::PyOpNode*> inserted_nodes;

    std::shared_ptr<void> compiler_cfg;
    
    unsigned int subgraph_idx;

   public:
    DecomposingContext(Graph* graph, graphlib::PyOpNode* node, std::shared_ptr<void> compiler_cfg) :
        graph(graph), node_(node), compiler_cfg(compiler_cfg)
    {
        subgraph_idx = graph->get_subgraph_id_for_node(node->id());
    }

    // Available to op/eval/*
    NodeContext op(
        graphlib::OpType const& op_type,
        std::vector<NodeContext> const& operands,
        bool copy_tms = true,
        bool dont_decompose = false,
        bool optimize_hoist = false,
        DataFormat output_df = DataFormat::Invalid);
    void fuse(NodeContext operand, graphlib::PortId out_port);
    NodeContext tensor(
        std::shared_ptr<void> tensor_handle, graphlib::Shape tensor_shape, DataFormat df = DataFormat::Invalid);

    Graph* get_graph() { return graph; }

    inline int get_op_index() { return op_index; }

    inline graphlib::NodeId get_output_node_id() { return output_node_id; }

    inline bool is_training_enabled() { return graph->enable_training(); }

    inline std::string get_node_name() { return node_->name(); }

    inline std::shared_ptr<void> get_compiler_cfg() { return compiler_cfg; }
};

std::vector<std::pair<graphlib::NodeId, graphlib::NodeId>> decompose_pybuda_graph(
    Graph* graph, const char* dispatcher_name, std::shared_ptr<void> compiler_cfg);

}  // namespace tt
