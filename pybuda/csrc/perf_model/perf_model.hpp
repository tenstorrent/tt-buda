// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <memory>

#include "balancer/balancer.hpp"
#include "perf_model/graph.hpp"

namespace tt
{
namespace graphlib
{
class Graph;
}
namespace perf_model
{

class PerfModel
{
   private:
    std::unique_ptr<perf_model::Graph> graph;
    std::vector<std::unique_ptr<perf_model::Graph>> temporal_epoch_graphs;
    std::string graph_name;
    const DeviceConfig &device_config;

    std::unordered_map<std::string, float> results;

   public:
    PerfModel(
        graphlib::Graph *g,
        const std::string &graph_name,
        const DeviceConfig &device_config,
        const std::shared_ptr<balancer::BalancerSolution> balancer_solution,
        bool input_queues_on_host,
        bool output_queues_on_host);

    using NodeMap = std::unordered_map<graphlib::Node *, NodeP>;

    std::unordered_map<std::string, float> get_results() const { return results; }

   private:
    void create_graphs(
        graphlib::Graph *g,
        const std::shared_ptr<balancer::BalancerSolution> balancer_solution,
        bool input_queues_on_host,
        bool output_queues_on_host);
    void calculate_ideal_bws(const SystemSpec &system);

    void create_op(
        graphlib::Graph *g,
        graphlib::BudaOpNode *op,
        const std::shared_ptr<balancer::BalancerSolution> balancer_solution,
        NodeMap &node_map,
        std::vector<NodeMap> &epoch_node_map);

    void create_tm(
        graphlib::Graph *g,
        graphlib::BudaNaryTMNode *tm,
        const std::shared_ptr<balancer::BalancerSolution> balancer_solution,
        NodeMap &node_map,
        std::vector<NodeMap> &epoch_node_map);

    void create_queue(
        graphlib::Graph *g,
        graphlib::QueueNode *q,
        const std::shared_ptr<balancer::BalancerSolution> balancer_solution,
        NodeMap &node_map,
        std::vector<NodeMap> &epoch_node_map);

    void calculate_utilization(const SystemSpec &system);
};

std::unordered_map<std::string, float> run_performance_model(
    graphlib::Graph *g,
    const std::string &graph_name,
    const DeviceConfig &device_config,
    const std::shared_ptr<balancer::BalancerSolution> balancer_solution,
    bool input_queues_on_host,
    bool output_queues_on_host);

}  // namespace perf_model
}  // namespace tt
