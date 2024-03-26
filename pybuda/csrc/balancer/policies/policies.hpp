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

BalancerPolicySolution run_policy(
    graphlib::Graph const* graph,
    BalancerConfig& config,
    legalizer::GraphSolver& graph_solver);
}  // namespace tt::balancer
