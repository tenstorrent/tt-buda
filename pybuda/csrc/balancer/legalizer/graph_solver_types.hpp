// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "graph_lib/edge.hpp"

namespace tt::balancer::legalizer
{

// Simple struct describing how NOPs should be inserted in graph.
//
struct BufferInfo
{
    graphlib::Edge edge;
    int nop_count;
    bool hoist_tms;

    BufferInfo(const graphlib::Edge& edge, int nop_count, bool hoist_tms) :
        edge(edge), nop_count(nop_count), hoist_tms(hoist_tms)
    {
    }
};

enum GraphSolverOpModelInvalidationStrategy
{
    MatmulSparseDenseGridPairing = 1,
    DenseMatmulPrologue = 1 << 1,
    DenseMatmulBetterUkt = 1 << 2,
};

enum GraphSolverOpModelInvalidationStrategyTier
{
    FirstTier = 1,
    SecondTier
};

enum GraphSolverSelfCutType
{
    None = 0,
    ConsumerOperandDataEdgesFirst,
    ProducerUserDataEdgesFirst,
    FastCut
};

}  // namespace tt::balancer::legalizer
