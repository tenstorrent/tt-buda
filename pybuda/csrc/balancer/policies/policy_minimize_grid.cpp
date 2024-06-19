// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "balancer/policies/policy_minimize_grid.hpp"
#include "balancer/policies/policy_manager.hpp"
#include "balancer/policies/policy_utils.hpp"

using Graph = tt::graphlib::Graph;
using Node = tt::graphlib::Node;

namespace tt::balancer
{
BalancerPolicySolution run_policy_minimize_grid(
    Graph const* graph, BalancerConfig const& config, legalizer::GraphSolver& graph_solver)
{
    PolicyManager policy_manager(graph, config, graph_solver);
    bool epoch_completed = false;
    bool maximize_grid = env_as<bool>("PYBUDA_MAXIMIZE_GRID", false);
    if (maximize_grid)
    {
        policy_manager.invalidate_suboptimal_op_models(legalizer::MatmulSparseDenseGridPairing);
    }

    // Pick OpModel for each node.
    //
    while (const Node* node = policy_manager.get_next_op())
    {
        auto legal_op_models = policy_manager.at(node);
        const OpModel* target_grid_op_model = &(*legal_op_models.begin());

        for (const OpModel& op_model : legal_op_models)
        {
            if ((!maximize_grid and op_model.grid_shape.volume() < target_grid_op_model->grid_shape.volume()) or
                (maximize_grid and op_model.grid_shape.volume() > target_grid_op_model->grid_shape.volume()))
            {
                target_grid_op_model = &op_model;
            }
        }

        std::tie(std::ignore, epoch_completed, std::ignore) = policy_manager.commit_op(*target_grid_op_model);

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
