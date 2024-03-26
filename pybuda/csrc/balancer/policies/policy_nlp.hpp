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

BalancerPolicySolution run_policy_nlp(
    graphlib::Graph const* graph,
    BalancerConfig const&,
    legalizer::GraphSolver& graph_solver,
    std::uint32_t target_cycles = 0);
 
BalancerPolicySolution run_policy_nlp_v2(
    graphlib::Graph const* graph,
    BalancerConfig const& config,
    legalizer::GraphSolver& graph_solver,
    std::uint32_t target_cycles = 0);

std::uint32_t calculate_target_cycles(
    graphlib::Graph const* graph, legalizer::GraphSolver& graph_solver, std::string const& arch_name);

template <typename T>
std::unordered_map<graphlib::Node*, std::uint32_t> find_min_prologue_volumes(
    graphlib::Graph const* graph, const std::vector<graphlib::Node*>& topo_sort, T& graph_solver)
{
    // Until back-end can reblock on prologue, we have to ensure that shared parameters (i.e. parameters read by
    // multiple matmuls) are reads by ops with the same grid sizes. Otherwise, the queue will have to be blocked for the
    // smaller grid, and then the op that needs a bigger grid will no longer fit in L1

    std::unordered_map<graphlib::Node*, std::uint32_t> min_param_grid_volume;

    // Search for matmul parameters
    for (graphlib::Node* node : topo_sort)
    {
        if ((node->node_type() != graphlib::NodeType::kBudaOp) ||
            (node->as<graphlib::BudaOpNode>()->op_type().op != "matmul"))
            continue;

        // Find minimum valid for prologue
        auto grids = graph_solver.at(node);
        bool found_prologue = false;
        std::uint32_t min_volume = 100000;
        for (auto grid : grids)
        {
            bool has_prologue = grid.parameter_buffers[1];
            std::uint32_t volume = grid.grid_shape.volume();
            if (has_prologue && (!found_prologue || (min_volume > volume)))
            {
                min_volume = volume;
                found_prologue = true;
            }
        }
        graphlib::Node* param_node = graph->data_operands(node)[1];
        auto it = min_param_grid_volume.find(param_node);
        // Record max of all the min volumes
        if (found_prologue && ((it == min_param_grid_volume.end()) || (it->second < min_volume)))
        {
            min_param_grid_volume[param_node] = min_volume;
            log_debug(
                LogBalancer,
                "Setting minimum prologue volume on {} to {} due to {}",
                param_node->name(),
                min_volume,
                node->name());
            found_prologue = true;
        }
    }

    return min_param_grid_volume;
}

template <typename T>
std::uint32_t get_matmul_target_cycles(
    graphlib::Graph const* graph,
    const std::vector<graphlib::Node*>& topo_sort,
    T& graph_solver,
    const std::unordered_map<graphlib::Node*, std::uint32_t>& min_param_grid_volume,
    std::string const& arch_name)
{
    // Aim for the biggest block while fitting all parameters in L1... if possible.
    // To start, find the slowest cycle count for each matmul in which parameters fit.
    std::uint32_t slowest_matmul_cycles = UINT32_MAX;

    std::vector<graphlib::Node*> topo_matmuls;
    for (graphlib::Node* node : topo_sort)
    {
        if (node->node_type() != graphlib::NodeType::kBudaOp)
            continue;

        std::string op_type = node->as<graphlib::BudaOpNode>()->op_type().op;
        if (op_type != "matmul")
            continue;

        if (node->as<graphlib::BudaOpNode>()->is_sparse_matmul())
            continue;  // for now, ignore sparse matmuls

        topo_matmuls.push_back(node);
    }

    std::uint32_t min_cycles_filter = 0;
    for (graphlib::Node* node : topo_matmuls)
    {
        // Find the largest cycle count of the fastest we can do in the target rows. This is the minimum
        // cycle count we can allow.

        std::uint32_t min_prologue_volume = 0;  // min volume needed to remain prologue after other checks
        auto it = min_param_grid_volume.find(graph->data_operands(node)[1]);
        if (it != min_param_grid_volume.end())
            min_prologue_volume = it->second;

        const OpShape& op_shape = (*graph_solver.at(node).begin()).op_shape;
        std::uint32_t target_rows = std::uint32_t(op_shape.outputs.at(0).rt / 6);
        if (target_rows == 0)
            target_rows = 1;

        std::uint32_t fastest_cycles = UINT32_MAX;

        auto grids = graph_solver.at(node);
        for (auto grid : grids)
        {
            if ((std::uint32_t)grid.grid_shape.volume() < min_prologue_volume)
                continue;

            if ((std::uint32_t)grid.grid_shape.r != target_rows)
                continue;

            // Skip the extrmely small shapes, as they are very inefficient at the moment
            // TODO: add these as config options
            std::string op_type = node->as<graphlib::BudaOpNode>()->op_type().op;
            if (op_type == "matmul")
            {
                if (op_shape.outputs.at(0).ct / grid.grid_shape.c < 3)
                    continue;

                if (op_shape.outputs.at(0).rt / grid.grid_shape.r < 4)
                    continue;
            }

            std::uint32_t cycles = grid.get_execution_cycles(arch_name);
            if (cycles < fastest_cycles)
                fastest_cycles = cycles;
        }

        if ((fastest_cycles != UINT32_MAX) && (fastest_cycles > min_cycles_filter))
        {
            min_cycles_filter = fastest_cycles;
            log_debug(LogBalancer, "Setting min cycle filter to {} due to {}", min_cycles_filter, node->name());
        }
    }

    float min_cycles_margin = 0.85;
    min_cycles_filter = 1.0 * min_cycles_filter * min_cycles_margin;
    log_debug(LogBalancer, "Final min cycle filter is {} after margin {}", min_cycles_filter, min_cycles_filter);

    for (graphlib::Node* node : topo_matmuls)
    {
        std::uint32_t min_prologue_volume = 0;  // min volume needed to remain prologue after other checks
        auto it = min_param_grid_volume.find(graph->data_operands(node)[1]);
        if (it != min_param_grid_volume.end())
            min_prologue_volume = it->second;

        // Find the actual smallest grid, with matching target rows, if possible
        const OpShape& op_shape = (*graph_solver.at(node).begin()).op_shape;
        std::uint32_t target_rows = std::uint32_t(op_shape.outputs.at(0).rt / 6);
        if (target_rows == 0)
            target_rows = 1;

        std::uint32_t smallest_grid_volume = UINT32_MAX;
        std::uint32_t smallest_grid_cycles;
        std::uint32_t smallest_grid_volume_bad_rows = UINT32_MAX;  // backup in case we can't find the right target rows
        std::uint32_t smallest_grid_cycles_bad_rows = 0;

        auto grids = graph_solver.at(node);
        for (auto grid : grids)
        {
            if ((std::uint32_t)grid.grid_shape.volume() < min_prologue_volume)
                continue;

            std::uint32_t cycles = grid.get_execution_cycles(arch_name);

            if ((std::uint32_t)grid.grid_shape.r == target_rows)
            {
                if ((std::uint32_t)grid.grid_shape.volume() < smallest_grid_volume)
                {
                    smallest_grid_volume = grid.grid_shape.volume();
                    smallest_grid_cycles = cycles;
                }
            }
            else if ((std::uint32_t)grid.grid_shape.volume() < smallest_grid_volume_bad_rows)
            {
                smallest_grid_volume_bad_rows = grid.grid_shape.volume();
                smallest_grid_cycles_bad_rows = cycles;
            }
        }

        if (smallest_grid_volume == UINT32_MAX && smallest_grid_volume_bad_rows == UINT32_MAX)
        {
            log_warning(
                LogBalancer,
                "Matmul {} has no grid for which we can fit parameters in L1. Performance might suffer.",
                node->name());
        }
        else
        {
            std::uint32_t cycles =
                (smallest_grid_volume < UINT32_MAX) ? smallest_grid_cycles : smallest_grid_cycles_bad_rows;
            // std::cout << "Node " << node->name() << " target cycles: " << smallest_grid_volume << ": "
            //           << smallest_grid_cycles << ", bad rows: " << smallest_grid_volume_bad_rows << ": "
            //           << smallest_grid_cycles_bad_rows << std::endl;
            if ((cycles >= min_cycles_filter) && (cycles < slowest_matmul_cycles))
            {
                slowest_matmul_cycles = cycles;
                // std::cout << "Setting slowest matmul cycles to " << cycles << " because of " << node->name()
                //           << std::endl;
            }
        }
    }

    float margin = 1.2;
    std::uint32_t target_cycles = 1.0 * slowest_matmul_cycles * margin;

    // Set a reasonable range until this is more robust
    if (target_cycles < 45000)
        target_cycles = 45000;
    if (target_cycles > 125000)
        target_cycles = 125000;

    log_info(LogBalancer, "Based on NLP matmul analysis, target cycle count is set to {}", target_cycles);
    return target_cycles;
}

}  // namespace tt::balancer