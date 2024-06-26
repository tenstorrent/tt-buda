// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <unordered_map>
#include <vector>

#include "balancer/legalizer/graph_solver.hpp"
#include "balancer/types.hpp"
#include "graph_lib/graph.hpp"
#include "graph_lib/node.hpp"

namespace tt::balancer
{
struct BalancerConfig;
struct BalancerPolicySolution;

BalancerPolicySolution run_policy_single_op_per_epoch(
    graphlib::Graph const* graph,
    BalancerConfig const& config,
    legalizer::GraphSolver& graph_solver);

}  // namespace tt::balancer
