// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <pybind11/pybind11.h>

#include <iostream>
#include <string>
#include <vector>

#include "graph_lib/node_types.hpp"
#include "graph_lib/utils.hpp"
#include "utils/logger.hpp"

namespace tt::passes
{

enum StackSliceOpType
{
    None,
    HSlice,
    HStack,
    VSlice,
    VStack
};

void fuse_reshape_transpose_pairs_into_slice_or_stack_tm(graphlib::Graph *graph);

graphlib::OpNode *find_valid_candidate(
    StackSliceOpType ref_op_type, graphlib::Graph *graph, graphlib::OpNode *initial_op);

graphlib::OpNode *find_valid_candidate_for_hslice(
    graphlib::Graph *graph, graphlib::OpNode *initial_op, graphlib::OpNode *reference_op = nullptr);

bool is_hslice_compatible(graphlib::Graph *graph, graphlib::OpNode *a, graphlib::OpNode *b);

graphlib::OpNode *find_valid_candidate_for_hstack(
    graphlib::Graph *graph, graphlib::OpNode *initial_op, graphlib::OpNode *reference_op = nullptr);

bool is_hstack_compatible(graphlib::OpNode *a, graphlib::OpNode *b);

bool valid_commute_through_forks(
    graphlib::Graph *graph, graphlib::OpNode *firstOp, graphlib::OpNode *lastOp, graphlib::OpNode *commuteOp = nullptr);

bool can_commute_past_operand(graphlib::OpNode *op);

void commute_through_forks(
    StackSliceOpType ref_op_type, graphlib::Graph *graph, graphlib::OpNode *firstOp, graphlib::Node *lastOp);

void commute_through_hslice_forks(graphlib::Graph *graph, graphlib::OpNode *firstOp, graphlib::Node *lastOp);

void commute_through_hstack_forks(graphlib::Graph *graph, graphlib::OpNode *firstOp, graphlib::Node *lastOp);

void update_shape_during_commute(graphlib::Graph *graph, graphlib::Node *operand_node, graphlib::Node *lastOp);

void fuse_into_slice_or_stack(StackSliceOpType ref_op_type, graphlib::Graph *graph, graphlib::OpNode *firstOp);

void fuse_into_hslice(graphlib::Graph *graph, graphlib::OpNode *firstOp);

void fuse_into_hstack(graphlib::Graph *graph, graphlib::OpNode *firstOp);

void convert_reshape_into_vslice_or_vstack_if_possible(graphlib::Graph *graph, graphlib::OpNode *reference_op);

void convert_reshape_into_vslice(graphlib::Graph *graph, graphlib::OpNode *reference_op, std::vector<uint32_t> reshape_shape, std::vector<uint32_t> operand_shape);

void convert_reshape_into_vstack(graphlib::Graph *graph, graphlib::OpNode *reference_op, std::vector<uint32_t> reshape_shape, std::vector<uint32_t> operand_shape);

}  // namespace tt::passes
