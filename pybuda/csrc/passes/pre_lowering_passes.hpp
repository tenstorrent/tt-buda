// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once


#include "graph_lib/defines.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/node.hpp"
#include "graph_lib/utils.hpp"
#include "utils/logger.hpp"



namespace tt {

using Graph = graphlib::Graph;
using Node = graphlib::Node;

void convert_broadcast_ops_to_tms(Graph *graph);

void insert_tile_broadcast_ops(Graph *graph);

void fold_tile_broadcast_ops_into_inputs(Graph *graph);

// Fold explicit broadcast ops into reduce ops, where possible... Reduce can broadcast immediately
// through a different constatnt
void fold_tile_broadcast_ops_into_reduce(Graph *graph);

void bypass_embedding_input_nops(Graph *graph);
void duplicate_embedding_table_if_needed(Graph *graph);

bool safe_to_hoist_past(const Graph *graph, const Node *operand);

void fuse_bias(Graph *graph);
void fuse_gelu(Graph *graph);
void fuse_requantize(Graph *graph);
void place_inter_subgraph_queues(graphlib::Graph *graph);

void replace_with_broadcasted_const(
    Graph *graph,
    graphlib::ConstantInputNode* constant,
    std::shared_ptr<void> broadcasted_tensor,
    graphlib::Shape target_shape,
    graphlib::PyOpNode* original_tile_bcast);

}
