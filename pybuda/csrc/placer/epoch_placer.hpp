// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "balancer/balancer.hpp"
#include "balancer/balancer_cache_collection.hpp"

namespace tt
{
namespace graphlib
{
class Graph;
}

namespace placer
{

// Return balancer solution for the full graps.
// It could modify the graph, and update the pointer
std::shared_ptr<balancer::BalancerSolution> run_epoch_placer(
    Graph** graph,
    balancer::BalancerConfig const& config,
    std::shared_ptr<balancer::BalancerCacheCollection> cache_collection);

}  // namespace placer
}  // namespace tt
