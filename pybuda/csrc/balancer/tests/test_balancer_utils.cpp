// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "test_balancer_utils.hpp"

#include "passes/passes_utils.hpp"
#include "passes/pre_placer_buda_passes.hpp"

namespace tt::test
{
std::unique_ptr<Graph> prepare_graph_for_legalizer(Graph *graph)
{
    std::unique_ptr<Graph> lowered_graph = lower_to_buda_ops(graph);
    recalculate_shapes(lowered_graph.get());
    calculate_ublock_order(lowered_graph.get());

    return lowered_graph;
}

balancer::BalancerConfig create_balancer_config(
    ARCH arch,
    std::optional< std::vector<std::uint32_t> > device_chip_ids,
    balancer::PolicyType policy_type,
    std::string cluster_config_yaml,
    std::string runtime_params_yaml,
    placer::ChipPlacementPolicy chip_placement_policy)
{
    return balancer::BalancerConfig(create_device_config(
        arch,
        device_chip_ids,
        cluster_config_yaml,
        runtime_params_yaml),
        policy_type,
        chip_placement_policy);
}

std::shared_ptr<tt::balancer::BalancerCacheCollection> create_balancer_cache_collection()
{
    return std::make_shared<tt::balancer::BalancerCacheCollection>();
}

}  // namespace tt::test
