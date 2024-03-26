// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <unordered_map>
#include <vector>

#include "balancer/policies/policy_utils.hpp"
#include "utils/logger.hpp"

namespace tt::balancer
{
struct BalancerConfig;
struct BalancerPolicySolution;

BalancerPolicySolution run_policy_ribbon(
    graphlib::Graph const *graph,
    const BalancerConfig &,
    legalizer::GraphSolver &graph_solver);

BalancerPolicySolution run_policy_ribbon2(
    graphlib::Graph const *graph,
    const BalancerConfig &,
    legalizer::GraphSolver &graph_solver);

bool validate_sparse_matmul_model(
    const graphlib::BudaOpNode *op,
    const OpModel &op_model,
    const graphlib::Graph *graph,
    std::unordered_set<std::uint64_t> &validated_cache);
}  // namespace tt::balancer