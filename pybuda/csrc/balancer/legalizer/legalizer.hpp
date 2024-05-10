// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#include "balancer/balancer.hpp"
#include "balancer/balancer_cache_collection.hpp"
#include "balancer/balancer_utils.hpp"
#include "balancer/types.hpp"
#include "graph_lib/graph.hpp"
#include "graph_lib/node.hpp"

namespace tt::balancer
{
struct BalancerConfig;
}  // namespace tt::balancer

namespace tt::balancer::legalizer
{
using UBlockOrder = tt::graphlib::UBlockOrder;
using LegalSparseUKts = std::unordered_map<int, std::vector<int>>;

LegalOpModels get_legal_op_models(
    Graph const* graph,
    BalancerConfig const& config,
    std::shared_ptr<BalancerCacheCollection> cache_collection,
    std::unordered_set<graphlib::Node*>* nodes_to_legalize = nullptr);

OpModels resolve_fork_grids(Graph const* graph, BalancerConfig const& config, OpModels selected_op_models);

std::tuple<OpModelMap, BlockShapeMap, OutputHostTMMap, CutEdges> resolve_block_shapes(
    Graph const* graph, BalancerConfig const& config, GraphSolverSolution const& graph_solver_solution);

std::pair<OpModel, OpModelFailureReason> calculate_op_model(
    Graph const* graph,
    std::shared_ptr<BalancerCacheCollection> cache_collection,
    graphlib::BudaOpNode const* op_node,
    GridShape selected_grid,
    TStreamFactor t_stream_factor,
    UBlockOrder ublock_order,
    bool force_dram_parameters,
    std::size_t dst_size,
    std::size_t l1_usable_size,
    std::size_t dram_channel_capacity,
    std::string& customFailureMessage,
    int fracture_factor = 1,
    LegalSparseUKts const& = {},
    int u_kt_override = 0,
    std::map<std::uint32_t, std::uint32_t> const& min_input_buffer_factor_overrides = {},
    std::optional<int> output_buffer_factor_override = {},
    bool fallback_single_buffer = false);

}  // namespace tt::balancer::legalizer
