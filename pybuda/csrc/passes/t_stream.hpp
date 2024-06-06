// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "balancer/balancer.hpp"
#include "graph_lib/graph.hpp"
#include "graph_lib/node.hpp"

namespace tt
{
//
// insert_t_stream_tms_for_eltwise
//
// Low level convenience function for inserting streaming TMs for the
// eltwise simple case. Only legal if you can guarantee that the
// consumer will accept eltwise tile ordering.
//
void insert_t_stream_tms_for_eltwise(
    std::vector<graphlib::OpType>& tms,
    balancer::TStreamFactor consumer_factor,
    balancer::TStreamFactor producer_factor,
    bool after_transpose = true);

//
// insert_t_stream_tms
//
// consumer:
//   The consumer op by which to apply incoming t-stream TMs.
//
// tms:
//   A vector of TMs, modified in place.  Results in a vector of TMs
//   with t-streaming TMs inserted.
//
// consumer_factor/producer_factor:
//   Respective consumer/producer t-streaming factors.
//
// operand_idx:
//   The edge input port that the provided tms belong to for this
//   consumer.
//
// through_queue:
//   If true, we have construction `producer -> e2e -> consumer`.
//   e2e inherits its producer's streaming amount.  This is a special
//   case because t-streaming constraints are relaxed when bouncing
//   through a queue so extra logic is needed to canonicalize the
//   form through the queue.
//
// group:
//   surrounds the stream TMs with a group factor, e.g.
//   Unstreaming R major you might have:
//
//       vstack(5)
//
//   If you wanted to preserve t-groupings of 9, than group=9 would
//   result:
//
//       hstack(9)
//       vstack(5)
//       hslice(9)
//
//   Note, the grouping is achieved by first stacking in the opposite
//   direction, applying the underlying streaming TM, and then slicing
//   back.
//
void insert_t_stream_tms(
    graphlib::OpNode const* consumer,
    std::vector<graphlib::OpType>& tms,
    balancer::TStreamFactor consumer_factor,
    balancer::TStreamFactor producer_factor,
    int operand_idx,
    bool through_queue = false,
    int group = 1,
    bool consumes_rz_major = false);

//
// insert_t_stream_tms
//
// Insert t-stream tms for all edges in the graph, given the selected set
// of op_models.
//
void insert_t_stream_tms(Graph* graph, balancer::OpModelMap const& op_models);

// Layout dataflow reorders the output buffer of sparse matmul in a way
// such that each row of cores between a sparse/consumer pair has a 1to1
// mapping of tiles and avoids inefficient gathers.  This function erases
// the existing TMs along this path and replaces them with "per row core"
// equivalent set of TMs. This often results in more complicated TMs, but
// much simpler pipes.
void insert_sparse_dataflow_tms(
    graphlib::Graph const* graph, graphlib::Node const* node, balancer::OpModel const& op_model);

void insert_sparse_dataflow_tms(std::vector<graphlib::OpType>& edge_tms, balancer::OpModel const& op_model);

}  // namespace tt
