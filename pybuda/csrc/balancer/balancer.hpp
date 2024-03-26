// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "backend_api/device_config.hpp"
#include "balancer/balancer_cache_collection.hpp"
#include "balancer/balancer_config.hpp"
#include "balancer/balancer_utils.hpp"
#include "balancer/legalizer/graph_solver.hpp"
#include "balancer/policies/policy_types.hpp"
#include "balancer/types.hpp"
#include "graph_lib/graph.hpp"
#include "placer/chip_id_assignment.hpp"
#include "placer/placer.hpp"
#include "scheduler/scheduler.hpp"
#include "utils/env.hpp"
#include "utils/logger.hpp"
#include "utils/result.hpp"

using Graph = tt::graphlib::Graph;

namespace tt::balancer
{

struct BalancerScore
{
    float solution_score = 0;
    std::vector<float> epoch_scores;
};

struct BalancerSolution
{
    placer::PlacerSolution placer_solution;
    OpModelMap op_models;
    BlockShapeMap block_shapes;
    OutputHostTMMap output_host_tms;
    CutEdges graph_solver_cut_edges;
    BalancerScore balancer_score;

    BalancerSolution(
        placer::PlacerSolution const& placer_solution,
        OpModelMap const& op_models,
        BlockShapeMap const& block_shapes,
        OutputHostTMMap const& output_host_tms,
        CutEdges const& graph_solver_cut_edges,
        BalancerScore const& balancer_score = BalancerScore()) :
        placer_solution(placer_solution),
        op_models(op_models),
        block_shapes(block_shapes),
        output_host_tms(output_host_tms),
        graph_solver_cut_edges(graph_solver_cut_edges),
        balancer_score(balancer_score)
    {
    }
};

struct BalancerPolicySolution
{
    std::optional<placer::PlacerSolution> placer_solution = std::nullopt;
    legalizer::GraphSolverSolution graph_solver_solution;
    BalancerScore balancer_score;

    BalancerPolicySolution() = default;

    BalancerPolicySolution(legalizer::GraphSolverSolution const& graph_solver_solution) :
        graph_solver_solution(graph_solver_solution)
    {
    }

    BalancerPolicySolution(
        placer::PlacerSolution const& placer_solution, legalizer::GraphSolverSolution const& graph_solver_solution, BalancerScore const& balancer_score) :
        placer_solution(placer_solution), graph_solver_solution(graph_solver_solution), balancer_score(balancer_score)
    {
    }
};

std::shared_ptr<BalancerSolution> run_balancer_and_placer(
    Graph* graph, BalancerConfig& config, std::shared_ptr<BalancerCacheCollection> cache_collection);

legalizer::GraphSolver get_graph_solver(
    BalancerConfig const& config,
    std::shared_ptr<BalancerCacheCollection> cache_collection,
    graphlib::Graph* graph,
    LegalOpModels const& legal_op_models,
    bool use_op_model_recalculation = true);

void update_ops_on_selected_op_models(graphlib::Graph const* graph, OpModels const& op_models);

};  // namespace tt::balancer
