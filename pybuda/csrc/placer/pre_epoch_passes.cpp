// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "placer/pre_epoch_passes.hpp"

#include "graph_lib/graph.hpp"

namespace tt::placer
{

// Return modified graph if modifications are made
std::unique_ptr<Graph> run_pre_epoch_passes(graphlib::Graph *, const balancer::BalancerConfig &, PlacerHistory &)
{
    // TODO
    return nullptr;
}

}  // namespace tt::placer
