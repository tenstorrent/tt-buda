// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "balancer/types.hpp"
#include "balancer/balancer_utils.hpp"
#include "graph_lib/node_types.hpp"
#include "passes_utils.hpp"
#include "placer/placer.hpp"
#include "balancer/balancer.hpp"

namespace tt::passes
{
std::unordered_map<graphlib::Edge, graphlib::Edge> get_forked_dram_inputs(
    bool enable_forked_dram_inputs,
    Graph* graph,
    std::unordered_map<std::string, placer::OpPlacement> *name_to_op_placement,
    balancer::OpModelMap* op_model);
};  // namespace tt::passess