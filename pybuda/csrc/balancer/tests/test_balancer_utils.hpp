// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <memory>
#include <optional>
#include <vector>
#include "balancer/balancer.hpp"
#include "balancer/balancer_cache_collection.hpp"
#include "test/common.hpp"

namespace tt::graphlib
{
class Graph;
}

namespace tt::balancer
{
struct BalancerConfig;
}

namespace tt::test
{
std::unique_ptr<tt::graphlib::Graph> prepare_graph_for_legalizer(tt::graphlib::Graph *graph);
balancer::BalancerConfig create_balancer_config(
    tt::ARCH arch = tt::ARCH::GRAYSKULL,
    std::optional< std::vector<std::uint32_t> > device_chip_ids = std::nullopt,
    balancer::PolicyType policy_type = balancer::PolicyType::Ribbon,
    std::string cluster_config_yaml = "",
    std::string runtime_params_yaml = "",
    placer::ChipPlacementPolicy chip_placement_policy = placer::ChipPlacementPolicy::MMIO_LAST);

std::shared_ptr<tt::balancer::BalancerCacheCollection> create_balancer_cache_collection();
}  // namespace tt::test
