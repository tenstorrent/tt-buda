// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <memory>
#include <utility>

#include "balancer/balancer.hpp"
#include "passes/fracture.hpp"

namespace tt::passes
{
void insert_queues(
    graphlib::Graph* graph,
    std::unordered_map<graphlib::Node*, const balancer::BudaOpNodeLegalizerFailureInfo>& nodes_without_legal_op_model);

std::pair<std::shared_ptr<balancer::BalancerSolution>, bool> run_placer_buda_passes(
    graphlib::Graph* graph,
    balancer::BalancerConfig balancer_config,
    FractureChipIdAssignments const& fracture_chip_id_assignments, 
    const py::dict &paddings_dict);
}  // namespace tt::passes
