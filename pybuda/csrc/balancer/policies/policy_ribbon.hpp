// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <unordered_map>
#include <vector>

#include "balancer/policies/policy_utils.hpp"
#include "utils/logger.hpp"

namespace tt::balancer
{
struct BalancerConfig;
}  // namespace tt::balancer

namespace tt::balancer
{
legalizer::GraphSolverSolution run_policy_ribbon(
    graphlib::Graph const *graph,
    const BalancerConfig &,
    legalizer::GraphSolver &graph_solver,
    std::optional<placer::PlacerSolution> &placer_solution);

legalizer::GraphSolverSolution run_policy_ribbon2(
    graphlib::Graph const *graph,
    const BalancerConfig &,
    legalizer::GraphSolver &graph_solver,
    std::optional<placer::PlacerSolution> &placer_solution);

class RibbonSolution
{
   public:
    struct OpModelPair
    {
        OpModel model;
        const graphlib::BudaOpNode *op;
    };

    std::unordered_set<const tt::graphlib::Node *> current_epoch_nodes;
    std::unordered_set<const tt::graphlib::Node *> current_epoch_ops;

   private:
    std::uint32_t ribbon_size;
    std::vector<OpModelPair> ops;
    float utilization;
    const DeviceConfig *device_config;
    const Graph *graph;
    int dram_readers_core_count;
    int dram_writers_core_count;

    float evaluate() const;
    void recalc_nodes();

   public:
    RibbonSolution(
        std::uint32_t ribbon_size,
        const DeviceConfig *device_config,
        std::vector<OpModelPair> &ops,
        const Graph *graph) :
        ribbon_size(ribbon_size), ops(ops), utilization(0.0f), device_config(device_config), graph(graph)
    {
        recalc_nodes();
        utilization = evaluate();
    }

    void update_model(std::uint32_t index, const OpModel &model)
    {
        ops[index].model = model;
        recalc_nodes();
        utilization = evaluate();
    }

    void set_op_count(std::size_t op_count)
    {
        ops.resize(op_count);
        recalc_nodes();
        utilization = evaluate();
    }

    void print() const;
    float get_score() const { return utilization; }
    const DeviceConfig *get_device_config() const { return device_config; }
    const std::vector<OpModelPair> &get_ops() const { return ops; }
    std::uint32_t get_ribbon_size() const { return ribbon_size; }
    const std::unordered_set<const tt::graphlib::Node *>& get_current_epoch_ops() { return current_epoch_ops; }
    const std::unordered_set<const tt::graphlib::Node *>& get_current_epoch_nodes() { return current_epoch_nodes; }
};

bool validate_sparse_matmul_model(
    const graphlib::BudaOpNode *op,
    const OpModel &op_model,
    const graphlib::Graph *graph,
    std::unordered_set<std::uint64_t> &validated_cache);
}  // namespace tt::balancer