// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include <optional>

#include "balancer/balancer.hpp"
#include "balancer/legalizer/graph_solver.hpp"
#include "balancer/legalizer/legalizer.hpp"
#include "balancer/policies/policies.hpp"
#include "balancer/tests/test_balancer_utils.hpp"
#include "balancer/types.hpp"
#include "graph_lib/defines.hpp"
#include "graph_lib/utils.hpp"
#include "gtest/gtest.h"
#include "passes/post_placer_buda_passes.hpp"
#include "test/common.hpp"

using namespace tt;
namespace tt::test
{

// check if node with node_name has queue (of specified QueueNodeType) as producer on specified consumer_input_port_id.
bool check_if_node_has_queue_as_prod(
    graphlib::Graph* graph, const std::string& node_name, graphlib::PortId port_id, graphlib::QueueNodeType queue_type)
{
    graphlib::Node* node = graph->get_node_by_name(node_name);
    std::vector<graphlib::Edge> operand_edges = graph->operand_data_edges(node);
    bool has_operand_on_port = false;
    for (graphlib::Edge operand_edge : operand_edges)
    {
        if (operand_edge.consumer_input_port_id == port_id)
        {
            has_operand_on_port = true;
            graphlib::NodeId producer_node_id = operand_edge.producer_node_id;
            graphlib::Node* producer_node = graph->node_by_id(producer_node_id);
            if (producer_node->node_type() != graphlib::NodeType::kQueue)
            {
                // According to the test producer_node should be queue type
                return false;
            }
            else
            {
                graphlib::QueueNode* queue = static_cast<graphlib::QueueNode*>(producer_node);
                return queue->queue_type() == queue_type ? true : false;
            }
        }
    }
    TT_ASSERT(has_operand_on_port, "node should have operand connected on input port_id");
    return false;
}
struct BypassBuffQueueMultipleConsumers : public BudaGraphTest
{
   protected:
    virtual std::vector<OpType*> create_graph() override
    {
        auto act_1 = create_activation(1, 1, 60 * 32, 32 * 32);
        auto in0_mm_0 = create_parameter(1, 1, 60 * 32, 60 * 32);
        auto in1_mm_1 = create_parameter(1, 1, 32 * 32, 1 * 32);
        auto in1_mm_2 = create_parameter(1, 1, 32 * 32, 1 * 32);
        auto in1_mm_3 = create_parameter(1, 1, 32 * 32, 1 * 32);

        auto matmul_0 = create_op("matmul", {in0_mm_0, act_1});

        auto buff_queue = create_buffering_queue(matmul_0, 2 /*num_entries*/);

        auto matmul_1 = create_op("matmul", {buff_queue, in1_mm_1});
        auto matmul_2 = create_op("matmul", {buff_queue, in1_mm_2});
        auto matmul_3 = create_op("matmul", {buff_queue, in1_mm_3});

        auto add_12 = create_op("add", {matmul_1, matmul_2});
        auto out = create_op("add", {add_12, matmul_3});

        return {out};
    }
};

TEST_F(BypassBuffQueueMultipleConsumers, bypass_buff_queue_with_multiple_consumers)
{
    // Buda graph
    graphlib::Graph* graph = get_graph();
    balancer::BalancerConfig balancer_config =
        create_balancer_config(ARCH::GRAYSKULL, std::nullopt, balancer::PolicyType::Ribbon);
    std::shared_ptr<balancer::BalancerCacheCollection> cache_collection = create_balancer_cache_collection();
    balancer_config.op_names_to_epoch_break.push_back({"matmul3"});

    balancer::LegalOpModels valid_op_models =
        balancer::legalizer::get_legal_op_models(graph, balancer_config, cache_collection);

    auto graph_solver = get_graph_solver(balancer_config, cache_collection, graph, valid_op_models);
    balancer::BalancerPolicySolution balancer_policy_solution = balancer::run_policy(graph, balancer_config, graph_solver);

    validate_subgraph_placement(graph, balancer_policy_solution.placer_solution.value());

    remove_buffering_queues_from_cross_epoch_edges(graph, balancer_policy_solution.placer_solution.value());

    insert_epoch_to_epoch_queues(
        graph,
        balancer_policy_solution.placer_solution.value(),
        {graphlib::NodeEpochType::Forward, graphlib::NodeEpochType::Backward, graphlib::Optimizer},
        balancer_policy_solution.graph_solver_solution.cut_edges);

    bool mm_1_check = check_if_node_has_queue_as_prod(graph, "matmul1", 0, graphlib::QueueNodeType::Buffering);
    EXPECT_TRUE(mm_1_check);
    bool mm_2_check = check_if_node_has_queue_as_prod(graph, "matmul2", 0, graphlib::QueueNodeType::Buffering);
    EXPECT_TRUE(mm_2_check);
    bool mm_3_check = check_if_node_has_queue_as_prod(graph, "matmul3", 0, graphlib::QueueNodeType::EpochToEpoch);
    EXPECT_TRUE(mm_3_check);
}

}  // namespace tt::test