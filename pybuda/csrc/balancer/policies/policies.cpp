// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "balancer/policies/policies.hpp"

#include "balancer/balancer.hpp"
#include "balancer/policies/policy_cnn.hpp"
#include "balancer/policies/policy_maximize_t_minimize_grid.hpp"
#include "balancer/policies/policy_minimize_grid.hpp"
#include "balancer/policies/policy_nlp.hpp"
#include "balancer/policies/policy_random.hpp"
#include "balancer/policies/policy_ribbon.hpp"

using Graph = tt::graphlib::Graph;
using Node = tt::graphlib::Node;
using Edge = tt::graphlib::Edge;
using DataFormat = tt::DataFormat;
using Schedule = std::vector<std::string>;

namespace tt::balancer
{
BalancerPolicySolution run_policy(
    Graph const *graph,
    BalancerConfig &config,
    legalizer::GraphSolver &graph_solver)
{
    TT_ASSERT(
        !config.use_interactive_placer or can_use_interactive_placer(config.policy_type),
        "Interactive_placer is not currently supported by this policy!");

    BalancerPolicySolution balancer_policy_solution;

    switch (config.policy_type)
    {
        case PolicyType::MaximizeTMinimizeGrid:
        {
            balancer_policy_solution = run_policy_maximize_t_minimize_grid(graph, config, graph_solver);
            break;
        }
        case PolicyType::MinimizeGrid:
        {
            balancer_policy_solution = run_policy_minimize_grid(graph, config, graph_solver);
            break;
        }
        case PolicyType::Random:
        {
            TT_ASSERT(config.use_interactive_placer);
            balancer_policy_solution = run_policy_random(graph, config, graph_solver);
            break;
        }
        case PolicyType::NLP:
        {
            // Use newest policy version if using interactive placer.
            //
            if (config.use_interactive_placer)
            {
                balancer_policy_solution = run_policy_nlp_v2(graph, config, graph_solver);
            }
            // Fallback to legacy policy version if not using interactive placer.
            //
            else
            {
                balancer_policy_solution = run_policy_nlp(graph, config, graph_solver);
            }
            break;
        }
        case PolicyType::CNN:
        {
            balancer_policy_solution = run_policy_cnn(graph, config, graph_solver);
            break;
        }
        case PolicyType::Ribbon:
        {
            // There is no implementation without interactive placer.
            //
            if (!config.use_interactive_placer)
            {
                TT_THROW(
                    "Ribbon policy has to use interactive placer! Enable interactive placer or switch to other "
                    "balancing policy.");
            }

            // Ribbon2 is not default yet until it's been tested across all models are large blobs are handled.
            bool use_ribbon2 = env_as<bool>("PYBUDA_RIBBON2", false);
            if (use_ribbon2)
            {
                balancer_policy_solution = run_policy_ribbon2(graph, config, graph_solver);
            }
            else
            {
                balancer_policy_solution = run_policy_ribbon(graph, config, graph_solver);
            }
            break;
        }
        default:
        {
            log_fatal("Unsupported policy_type {}", config.policy_type);
            return {};
        }
    }

    // If we used interactive placer, we should have a placer solution and vice versa.
    //
    TT_ASSERT(
        balancer_policy_solution.placer_solution.has_value() == config.use_interactive_placer,
        "Interactive placer usage not properly defined for chosen policy type(can_use_interactive_placer)!");

    return balancer_policy_solution;
}

// Does policy support using interactive placer or not.
//
bool can_use_interactive_placer(PolicyType policy_type)
{
    switch (policy_type)
    {
        case PolicyType::MaximizeTMinimizeGrid:
        case PolicyType::MinimizeGrid:
        case PolicyType::CNN: return false;

        case PolicyType::Random:
        case PolicyType::NLP:
        case PolicyType::Ribbon: return true;

        default: TT_ASSERT("Undefined interactive placer usage for policy!");
    }

    return false;
}

}  // namespace tt::balancer

// Include this hpp to include all policies
