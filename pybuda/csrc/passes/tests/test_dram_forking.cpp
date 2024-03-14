// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "balancer/balancer.hpp"
#include "balancer/balancer_cache_collection.hpp"
#include "balancer/legalizer/graph_solver.hpp"
#include "balancer/legalizer/legalizer.hpp"
#include "balancer/tests/test_balancer_utils.hpp"
#include "passes/forked_dram_inputs.hpp"
#include "placer/best_fit_allocator.hpp"
#include "placer/chip_id_assignment.hpp"
#include "placer/lowering_utils.hpp"
#include "placer/placer.hpp"
#include "test/common.hpp"

namespace tt::test
{

// This Graph has dram forking inputs
struct ForkedDramGraph : public BudaGraphTest
{
   protected:
    virtual std::vector<OpType *> create_graph() override
    {
        std::uint32_t seq_len = 32;
        std::uint32_t embed = 32;

        auto in0 = create_input("act0", shape(1, 1, seq_len, embed), tt::graphlib::InputNodeType::Activation);
        auto in1 = create_input("act1", shape(1, 1, seq_len, embed), tt::graphlib::InputNodeType::Activation);

        auto add0 = create_op("add", {in0, in1});
        (void)add0;
        auto add1 = create_op("add", {in0, in1});
        auto e2e1 = create_buffering_queue(add1, 1);
        auto add2 = create_op("add", {e2e1, in1});
        (void)add2;
        auto add3 = create_op("add", {in1, e2e1});

        auto gelu = create_op("gelu", {add3});
        return {gelu};
    }
};

// This test evaluates a regular dram forking inputs with scheduled ops on the same epoch , all ops with same
// block shape
TEST_F(ForkedDramGraph, forked_dram_test)
{
    graphlib::Graph *graph = get_graph();

    // get balancer solution
    balancer::BalancerConfig balancer_config = create_balancer_config();
    
    std::shared_ptr<balancer::BalancerCacheCollection> cache_collection = create_balancer_cache_collection();
    // run balancer and placer
    auto balancer_solution = run_balancer_and_placer(graph, balancer_config, cache_collection);
    // run dram forking pass
    auto forked_dram_map = tt::passes::get_forked_dram_inputs(
        true, graph, &balancer_solution->placer_solution.name_to_op_placement, &balancer_solution->op_models);

    // Can store main node and forked nodes info(similar to what netlist.cpp will do)
    std::unordered_map<std::string, std::vector<std::pair<std::string, std::string>>> forked_dram_node_map;
    for (Node *node : graphlib::topological_sort(*graph))
    {
        if (node->node_type() == graphlib::NodeType::kBudaOp)
        {
            for (auto operand : graph->operand_data_edges(node))
            {
                if (forked_dram_map.find(operand) != forked_dram_map.end())
                {
                    auto edge_temp = forked_dram_map[operand];
                    forked_dram_node_map[node->name()].push_back(std::make_pair(
                        graph->node_by_id(edge_temp.producer_node_id)->name(),
                        graph->node_by_id(edge_temp.consumer_node_id)->name()));
                }
            }
        }
    }

    // Verify if the forked nodes are correct
    ASSERT_TRUE(forked_dram_node_map.find("add0") != forked_dram_node_map.end());

    std::vector<std::pair<std::string, std::string>> add0_fork = {{"act1", "add3"}};
    ASSERT_TRUE(add0_fork == forked_dram_node_map["add0"]);

    ASSERT_TRUE(forked_dram_node_map.find("add1") != forked_dram_node_map.end());

    std::vector<std::pair<std::string, std::string>> add1_fork = {{"act0", "add0"}};
    ASSERT_TRUE(add1_fork == forked_dram_node_map["add1"]);

    ASSERT_TRUE(forked_dram_node_map.find("add2") != forked_dram_node_map.end());

    std::vector<std::pair<std::string, std::string>> add2_fork = {{"act1", "add3"}};
    ASSERT_TRUE(add2_fork == forked_dram_node_map["add2"]);

    ASSERT_TRUE(forked_dram_node_map.find("add3") != forked_dram_node_map.end());

    std::vector<std::pair<std::string, std::string>> add3_fork = {{"buff_queue0", "add2"}};
    ASSERT_TRUE(add3_fork == forked_dram_node_map["add3"]);
}

// This test evaluates a regular dram forking inputs with scheduled ops on the different epochs , all ops with same
// block shape
TEST_F(ForkedDramGraph, forked_dram_test_epoch_break)
{
    graphlib::Graph *graph = get_graph();

    // get balancer solution
    balancer::BalancerConfig balancer_config = create_balancer_config();
    std::shared_ptr<balancer::BalancerCacheCollection> cache_collection = create_balancer_cache_collection();
    // adding a break at add2 node  to be scheduled in a separate epoch
    balancer_config.op_names_to_epoch_break.push_back({"add2"});
    // run balancer and placer
    auto balancer_solution = run_balancer_and_placer(graph, balancer_config, cache_collection);
    // run dram forking pass
    auto forked_dram_map = tt::passes::get_forked_dram_inputs(
        true, graph, &balancer_solution->placer_solution.name_to_op_placement, &balancer_solution->op_models);

    // Mapping input node\buffer node -> forked consumer node1, forked consumer node2
    std::unordered_map<std::string, std::vector<std::pair<std::string, std::string>>> forked_dram_node_map;
    for (Node *node : graphlib::topological_sort(*graph))
    {
        if (node->node_type() == graphlib::NodeType::kInput || node->node_type() == graphlib::NodeType::kQueue)
        {
            for (auto user_edge : graph->user_data_edges(node))
            {
                if (forked_dram_map.find(user_edge) != forked_dram_map.end())
                {
                    auto edge_temp = forked_dram_map[user_edge];
                    forked_dram_node_map[node->name()].push_back(std::make_pair(
                        graph->node_by_id(edge_temp.consumer_node_id)->name(),
                        graph->node_by_id(user_edge.consumer_node_id)->name()));
                }
            }
        }
    }

    // Verify if the forked nodes are correct, remember epoch break is at node add2
    // Verify that we have 3 different forking dram inputs
    // Act0 -> add0, add1;
    // Act1 -> add0, add1; add3, add2
    // buff_queue0 -> add2, add3
    ASSERT_TRUE(forked_dram_node_map.size() == 3);

    // Verify Act0 -> add0, add1
    ASSERT_TRUE(forked_dram_node_map.find("act0") != forked_dram_node_map.end());
    ASSERT_TRUE(forked_dram_node_map["act0"].size() == 1);
    std::vector<std::pair<std::string, std::string>> act0_fork = {{"add0", "add1"}};
    ASSERT_TRUE(act0_fork == forked_dram_node_map["act0"]);

    // Verify Act1 -> add0, add1; add3, add2
    ASSERT_TRUE(forked_dram_node_map.find("act1") != forked_dram_node_map.end());
    ASSERT_TRUE(forked_dram_node_map["act1"].size() == 2);
    std::vector<std::pair<std::string, std::string>> act1_fork = {{"add0", "add1"}, {"add3", "add2"}};
    ASSERT_TRUE(act1_fork == forked_dram_node_map["act1"]);

    // Verify buff_queue0 -> add2, add3
    ASSERT_TRUE(forked_dram_node_map.find("buff_queue0") != forked_dram_node_map.end());
    ASSERT_TRUE(forked_dram_node_map["buff_queue0"].size() == 1);
    std::vector<std::pair<std::string, std::string>> buff_queue0_fork = {{"add2", "add3"}};
    ASSERT_TRUE(buff_queue0_fork == forked_dram_node_map["buff_queue0"]);
}

// This graph has data dependency between two nodes
struct ForkedDramGraphDependency : public BudaGraphTest
{
   protected:
    virtual std::vector<OpType *> create_graph() override
    {
        std::uint32_t seq_len = 32;
        std::uint32_t embed = 32;

        auto in0 = create_input("act0", shape(1, 1, seq_len, embed), tt::graphlib::InputNodeType::Activation);
        auto in1 = create_input("act1", shape(1, 1, seq_len, embed), tt::graphlib::InputNodeType::Activation);

        auto add0 = create_op("add", {in0, in1});
        auto add1 = create_op("add", {in1, add0});

        auto e2e1 = create_buffering_queue(add1, 1);
        auto add2 = create_op("add", {e2e1, in1});
        (void)add2;
        auto add3 = create_op("add", {in1, e2e1});

        auto gelu = create_op("gelu", {add3});
        return {gelu};
    }
};

// This test evaluates dram forking inputs with data dependency between two nodes and scheduled ops the same epoch , all
// ops with same block shape
TEST_F(ForkedDramGraphDependency, forked_dram_test_node_dependency)
{
    graphlib::Graph *graph = get_graph();

    // get balancer solution
    balancer::BalancerConfig balancer_config = create_balancer_config();
    std::shared_ptr<balancer::BalancerCacheCollection> cache_collection = create_balancer_cache_collection();
    // run balancer and placer
    auto balancer_solution = run_balancer_and_placer(graph, balancer_config, cache_collection);
    // run dram forking pass
    auto forked_dram_map = tt::passes::get_forked_dram_inputs(
        true, graph, &balancer_solution->placer_solution.name_to_op_placement, &balancer_solution->op_models);

    // Can store main node and forked nodes info(similar to what netlist.cpp will do)
    std::unordered_map<std::string, std::vector<std::pair<std::string, std::string>>> forked_dram_node_map;
    for (Node *node : graphlib::topological_sort(*graph))
    {
        if (node->node_type() == graphlib::NodeType::kBudaOp)
        {
            for (auto operand : graph->operand_data_edges(node))
            {
                if (forked_dram_map.find(operand) != forked_dram_map.end())
                {
                    auto edge_temp = forked_dram_map[operand];

                    forked_dram_node_map[node->name()].push_back(std::make_pair(
                        graph->node_by_id(edge_temp.producer_node_id)->name(),
                        graph->node_by_id(edge_temp.consumer_node_id)->name()));
                }
            }
        }
    }

    // Verify if the forked nodes are correct, remember add1 reads from add0 (i.e there is a data dependency between
    // add1 and add0, add1 and add2, add1 and add3)

    ASSERT_TRUE(forked_dram_node_map.find("add3") != forked_dram_node_map.end());

    std::vector<std::pair<std::string, std::string>> add3_fork = {{"buff_queue0", "add2"}};
    ASSERT_TRUE(add3_fork == forked_dram_node_map["add3"]);
}

// This graph has Prologue node(constants or weights ), same epoch, same block shapes
struct ForkedDramGraphPrologue : public BudaGraphTest
{
   protected:
    virtual std::vector<OpType *> create_graph() override
    {
        std::uint32_t seq_len = 32;
        std::uint32_t embed = 32;
        std::uint32_t hidden = 32;

        auto in0 = create_input("act0", shape(1, 1, seq_len, embed), tt::graphlib::InputNodeType::Activation);
        auto in1 = create_input("act1", shape(1, 1, seq_len, embed), tt::graphlib::InputNodeType::Activation);

        // constant parameters, weights make the node a prologue node
        auto Win = create_parameter(shape(1, 1, embed, hidden));
        auto matmul0 = create_op("matmul", {in0, Win});
        (void)matmul0;
        auto add0 = create_op("add", {in0, in1});

        auto e2e1 = create_buffering_queue(add0, 1);
        auto add1 = create_op("add", {e2e1, in1});
        (void)add1;
        auto add2 = create_op("add", {in1, e2e1});

        auto gelu = create_op("gelu", {add2});
        return {gelu};
    }
};

// This test evaluates dram forking inputs with a prologue node, same epoch, same block shapes, prologue nodes can not
// use dram forking optimization
TEST_F(ForkedDramGraphPrologue, forked_dram_test_prologue)
{
    graphlib::Graph *graph = get_graph();

    // get balancer solution
    balancer::BalancerConfig balancer_config = create_balancer_config();
    std::shared_ptr<balancer::BalancerCacheCollection> cache_collection = create_balancer_cache_collection();
    // run balancer and placer
    auto balancer_solution = run_balancer_and_placer(graph, balancer_config, cache_collection);
    // run dram forking pass
    auto forked_dram_map = tt::passes::get_forked_dram_inputs(
        true, graph, &balancer_solution->placer_solution.name_to_op_placement, &balancer_solution->op_models);

    // Can store main node and forked nodes info(similar to what netlist.cpp will do)
    std::unordered_map<std::string, std::vector<std::pair<std::string, std::string>>> forked_dram_node_map;
    for (Node *node : graphlib::topological_sort(*graph))
    {
        if (node->node_type() == graphlib::NodeType::kBudaOp)
        {
            for (auto operand : graph->operand_data_edges(node))
            {
                if (forked_dram_map.find(operand) != forked_dram_map.end())
                {
                    auto edge_temp = forked_dram_map[operand];

                    forked_dram_node_map[node->name()].push_back(std::make_pair(
                        graph->node_by_id(edge_temp.producer_node_id)->name(),
                        graph->node_by_id(edge_temp.consumer_node_id)->name()));
                }
            }
        }
    }
    // Verify if the forked nodes are correct, remember there is a prologue node
    ASSERT_TRUE(forked_dram_node_map.find("add0") != forked_dram_node_map.end());

    std::vector<std::pair<std::string, std::string>> add0_fork = {{"act0", "matmul0"}};
    ASSERT_TRUE(add0_fork == forked_dram_node_map["add0"]);

    ASSERT_TRUE(forked_dram_node_map.find("add1") != forked_dram_node_map.end());

    std::vector<std::pair<std::string, std::string>> add1_fork = {{"act1", "add2"}};
    ASSERT_TRUE(add1_fork == forked_dram_node_map["add1"]);

    ASSERT_TRUE(forked_dram_node_map.find("add2") != forked_dram_node_map.end());

    std::vector<std::pair<std::string, std::string>> add2_fork = {{"buff_queue0", "add1"}};
    ASSERT_TRUE(add2_fork == forked_dram_node_map["add2"]);
}

// This Graph has dram forking inputs
struct ForkedDramGraphGridShape : public BudaGraphTest
{
   protected:
    virtual std::vector<OpType *> create_graph() override
    {
        std::uint32_t seq_len = 32;
        std::uint32_t embed = 64;

        auto in0 = create_input("act0", shape(1, 1, seq_len, embed), tt::graphlib::InputNodeType::Activation);
        auto in1 = create_input("act1", shape(1, 1, seq_len, embed), tt::graphlib::InputNodeType::Activation);

        auto add0 = create_op("add", {in0, in1});
        auto add1 = create_op("add", {in0, in1});

        auto e2e1 = create_buffering_queue(add1, 1);
        auto add2 = create_op("add", {e2e1, in1});
        auto add3 = create_op("add", {in1, e2e1});

        auto gelu = create_op("gelu", {add3});
        return {add0, add2, gelu};
    }
};

// This test evaluates a regular dram forking inputs with scheduled ops on the same epoch , ops with different block
// shapes
TEST_F(ForkedDramGraphGridShape, forked_dram_test_diff_grid_shapes)
{
    graphlib::Graph *graph = get_graph();

    // get balancer solution
    balancer::BalancerConfig balancer_config = create_balancer_config();
    std::shared_ptr<balancer::BalancerCacheCollection> cache_collection = create_balancer_cache_collection();

    // force ops for different grid shapes (to have different block shapes)
    balancer_config.enable_t_streaming = false;
    balancer_config.op_overrides["add0"].grid_shape = std::make_pair(1, 1);
    balancer_config.op_overrides["add1"].grid_shape = std::make_pair(1, 1);
    balancer_config.op_overrides["add2"].grid_shape = std::make_pair(1, 2);
    balancer_config.op_overrides["add3"].grid_shape = std::make_pair(1, 1);
    // run balancer and placer
    auto balancer_solution = run_balancer_and_placer(graph, balancer_config, cache_collection);
    // run dram forking pass
    auto forked_dram_map = tt::passes::get_forked_dram_inputs(
        true, graph, &balancer_solution->placer_solution.name_to_op_placement, &balancer_solution->op_models);

    // Can store main node and forked nodes info(similar to what netlist.cpp will do)
    std::unordered_map<std::string, std::vector<std::pair<std::string, std::string>>> forked_dram_node_map;
    for (Node *node : graphlib::topological_sort(*graph))
    {
        if (node->node_type() == graphlib::NodeType::kBudaOp)
        {
            for (auto operand : graph->operand_data_edges(node))
            {
                if (forked_dram_map.find(operand) != forked_dram_map.end())
                {
                    auto edge_temp = forked_dram_map[operand];

                    forked_dram_node_map[node->name()].push_back(std::make_pair(
                        graph->node_by_id(edge_temp.producer_node_id)->name(),
                        graph->node_by_id(edge_temp.consumer_node_id)->name()));
                }
            }
        }
    }
    // Verify if the forked nodes are correct
    ASSERT_TRUE(forked_dram_node_map.find("add0") != forked_dram_node_map.end());

    std::vector<std::pair<std::string, std::string>> add0_fork = {{"act1", "add3"}};
    ASSERT_TRUE(add0_fork == forked_dram_node_map["add0"]);

    ASSERT_TRUE(forked_dram_node_map.find("add1") != forked_dram_node_map.end());

    std::vector<std::pair<std::string, std::string>> add1_fork = {{"act0", "add0"}};
    ASSERT_TRUE(add1_fork == forked_dram_node_map["add1"]);
}

}  // namespace tt::test
