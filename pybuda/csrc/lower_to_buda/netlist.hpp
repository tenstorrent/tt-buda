// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <memory>
#include <vector>

#include "balancer/balancer.hpp"
#include "lower_to_buda/comment.hpp"
#include "lower_to_buda/fused_op.hpp"
#include "lower_to_buda/graph.hpp"
#include "lower_to_buda/program.hpp"
#include "lower_to_buda/queue.hpp"
#include "placer/placer.hpp"

namespace tt {

struct BudaNetlistConfig {
};

struct BudaNetlist {
    Comment comments;
    Comment debug_info;
    std::vector<program::Program> programs;
    std::vector<BudaQueue> queues;
    std::vector<BudaGraph> graphs;
    std::vector<BudaFusedOp> fused_ops;

    std::vector<std::uint32_t> chip_ids;
    std::string arch_string;

    std::string dump_to_yaml() const;
    inline void append_comment(std::string const &comment)
    {
        if (comments)
            comments.str += "\n";
        comments.str += comment;
    }
};

BudaNetlist merge_netlists(std::vector<BudaNetlist> subgraphs);
// Create Buda queues, program, and graphs
BudaNetlist lower_to_buda_netlist(
    graphlib::Graph *graph,
    std::string &graph_name,
    placer::PlacerSolution &placer_solution,
    std::shared_ptr<balancer::BalancerSolution> balancer_solution,
    const std::vector<std::uint32_t> &chip_ids,
    const DeviceConfig &device_config,
    bool disable_forked_dram_inputs);

BudaFusedOp create_fused_op(graphlib::BudaOpNode *op, const balancer::OpModel &op_model);

} // namespace tt
