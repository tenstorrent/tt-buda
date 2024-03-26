// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "balancer/policies/policy_random.hpp"

#include <algorithm>
#include <random>

#include "balancer/policies/policy_manager.hpp"
#include "balancer/policies/policy_utils.hpp"
#include "graph_lib/node_types.hpp"

using Graph = tt::graphlib::Graph;
using Node = tt::graphlib::Node;
using NodeType = tt::graphlib::NodeType;
using Edge = tt::graphlib::Edge;
using DataFormat = tt::DataFormat;

namespace tt::balancer
{

BalancerPolicySolution run_policy_random(
    graphlib::Graph const* graph,
    BalancerConfig const& config,
    legalizer::GraphSolver& graph_solver)
{
    (void)config;
    log_debug(LogBalancer, "Starting Random balancing.");

    PolicyManager policy_manager(graph, config, graph_solver);
    bool epoch_completed = false;
    
    std::mt19937 rand_gen(config.random_policy_seed);

    // Pick OpModel for each node.
    //
    while (const graphlib::Node* node = policy_manager.get_next_op())
    {
        auto op_models = policy_manager.at(node);

        std::uniform_int_distribution<int> d(0, op_models.size() - 1);
        int random = d(rand_gen);

        auto op_model = op_models.begin();
        for (int i = 0; i < random; ++i)
        {
            ++op_model;
        }
        std::tie(std::ignore, epoch_completed, std::ignore) = policy_manager.commit_op(*op_model);

        // If we're done with the epoch, finish it.
        //
        if (epoch_completed)
        {
            policy_manager.finish_current_epoch();
        }
    }

    return policy_manager.commit_solution();
}

}  // namespace tt::balancer
