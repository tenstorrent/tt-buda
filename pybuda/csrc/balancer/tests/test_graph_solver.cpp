// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include <fstream>
#include <memory>

#include "balancer/balancer_cache_collection.hpp"
#include "balancer/legalizer/graph_solver.hpp"
#include "balancer/legalizer/legalizer.hpp"
#include "gtest/gtest.h"
#include "json.hpp"
#include "test/common.hpp"
#include "test_balancer_utils.hpp"

namespace tt::test
{
using namespace balancer;

struct UnitTestConstraint : public legalizer::Constraint
{
    UnitTestConstraint(
        const DeviceConfig& device_config,
        std::shared_ptr<balancer::BalancerCacheCollection> balancer_cache_collection) :
        Constraint(device_config, balancer_cache_collection)
    {
    }

    virtual std::pair<legalizer::EdgeCost, legalizer::ConstraintFailureReason> queue_to_op_cost(
        graphlib::Graph const*, graphlib::Edge, std::optional<OpModel>, OpModel const&) override
    {
        return std::make_pair(legalizer::EdgeCost{}, legalizer::NoConstraintFailure);
    }

    virtual std::pair<legalizer::EdgeCost, legalizer::ConstraintFailureReason> op_to_queue_cost(
        graphlib::Graph const*, graphlib::Edge, OpModel const&, std::optional<OpModel>) override
    {
        return std::make_pair(legalizer::EdgeCost{}, legalizer::NoConstraintFailure);
    }

    virtual std::pair<legalizer::EdgeCost, legalizer::ConstraintFailureReason> op_to_op_cost(
        graphlib::Graph const*, graphlib::Edge, OpModel const& producer, OpModel const& consumer) override
    {
        if (producer.op_model_valid_pair_id.count(consumer.id.id) == 0 or
            consumer.op_model_valid_pair_id.count(producer.id.id) == 0)
        {
            return std::make_pair(legalizer::EdgeCost{}, legalizer::Failed);
        }

        return std::make_pair(legalizer::EdgeCost{}, legalizer::NoConstraintFailure);
    }
};

struct JsonTest
{
    using Path = std::pair<int, int>;

    struct Edge
    {
        int producer;
        int consumer;
        int input_port;
        std::vector<Path> paths;
    };

    std::vector<Edge> edges;
    std::unordered_map<std::string, std::string> node_id_to_name;
    std::vector<std::string> flags;

    bool has_flag(std::string const& flag) const { return std::find(flags.begin(), flags.end(), flag) != flags.end(); }
};

void from_json(const nlohmann::json& j, JsonTest::Edge& e)
{
    j.at("producer").get_to(e.producer);
    j.at("consumer").get_to(e.consumer);
    j.at("input_port").get_to(e.input_port);
    j.at("paths").get_to(e.paths);
}

void from_json(const nlohmann::json& j, JsonTest& t)
{
    j.at("edges").get_to(t.edges);
    if (j.contains("node_names"))
        j.at("node_names").get_to(t.node_id_to_name);
    if (j.contains("flags"))
        j.at("flags").get_to(t.flags);
}

template <typename Fn>
static void cross(std::vector<OpModel>& as, std::vector<OpModel>& bs, Fn fn)
{
    for (auto& a : as)
        for (auto& b : bs) fn(a, b);
}

// OpModel Association functions
static void a2b(OpModel& a, OpModel& b) { b.op_model_valid_pair_id.insert(a.id.id); }
static void b2a(OpModel& a, OpModel& b) { a2b(b, a); }
static void both(OpModel& a, OpModel& b)
{
    a2b(a, b);
    b2a(a, b);
}

struct GraphSolverResolveSanity : testing::Test
{
    std::unique_ptr<Graph> graph;

    void SetUp() override
    {
        graph = std::make_unique<Graph>(graphlib::IRLevel::IR_PYBUDA);

        graphlib::Shape shape = graphlib::Shape::create({1, 1, 512, 160});

        auto in0_a = create_input(*graph, "in0_a", graphlib::Shape::create({1, 1, shape[2], 256}));
        auto in0_b = create_input(*graph, "in0_b", graphlib::Shape::create({1, 1, 256, shape[3]}));
        auto matmul0 = add_node<graphlib::PyOpNode>(*graph, "matmul0", "matmul", {}, {in0_a, in0_b});

        auto in1_a = create_input(*graph, "in1_a", graphlib::Shape::create({1, 1, shape[2], 128}));
        auto in1_b = create_input(*graph, "in1_b", graphlib::Shape::create({1, 1, 128, shape[3]}));
        auto matmul1 = add_node<graphlib::PyOpNode>(*graph, "matmul1", "matmul", {}, {in1_a, in1_b});

        auto add = add_node<graphlib::PyOpNode>(*graph, "add", "add", {}, {matmul0, matmul1});

        create_output(*graph, "out0", add);

        graph = prepare_graph_for_legalizer(graph.get());
    }
};

TEST_F(GraphSolverResolveSanity, resolve)
{
    balancer::BalancerConfig balancer_config = create_balancer_config();
    std::shared_ptr<balancer::BalancerCacheCollection> cache_collection = create_balancer_cache_collection();
    balancer::LegalOpModels valid_op_models =
        balancer::legalizer::get_legal_op_models(graph.get(), balancer_config, cache_collection);
    EXPECT_EQ(valid_op_models.size(), 3);
    legalizer::GraphSolver graph_solver =
        get_graph_solver(balancer_config, cache_collection, graph.get(), valid_op_models);

    auto topo_sort = tt::graphlib::topological_sort(*graph);

    // Simple - just set first available.
    //
    for (Node* node : topo_sort)
    {
        if (node->node_type() != graphlib::NodeType::kBudaOp)
        {
            continue;
        }

        auto opmodels = graph_solver.at(node);
        graph_solver.set(node, *opmodels.begin());

        // After calling SET, we expect having only one OpModel available for this node.
        //
        auto opmodels_after_set = graph_solver.at(node);
        EXPECT_EQ(opmodels_after_set.mask.count(), 1);
    }

    balancer::legalizer::GraphSolverSolution solution = graph_solver.finish();

    EXPECT_EQ(solution.cut_edges.size(), 0);
    EXPECT_EQ(solution.selected_op_models.size(), 3);
}

TEST_F(GraphSolverResolveSanity, resolve_no_streaming_output)
{
    balancer::BalancerConfig balancer_config = create_balancer_config();
    std::shared_ptr<balancer::BalancerCacheCollection> cache_collection = create_balancer_cache_collection();
    balancer_config.enable_t_streaming = true;
    balancer::LegalOpModels valid_op_models =
        balancer::legalizer::get_legal_op_models(graph.get(), balancer_config, cache_collection);
    EXPECT_EQ(valid_op_models.size(), 3);
    legalizer::GraphSolver graph_solver =
        get_graph_solver(balancer_config, cache_collection, graph.get(), valid_op_models);

    auto topo_sort = tt::graphlib::topological_sort(*graph);

    // Validate that there is no op_model allowing streaming into output.
    //
    for (Node* node : topo_sort)
    {
        if (node->node_type() == graphlib::NodeType::kOutput)
        {
            for (auto output_op : graph->data_operands(node))
            {
                auto opmodels = graph_solver.at(output_op);

                for (auto op_model : opmodels)
                {
                    EXPECT_TRUE(op_model.t_stream_factor.none());
                }
            }
        }
    }

    // Simple - just set first available.
    //
    for (Node* node : topo_sort)
    {
        if (node->node_type() != graphlib::NodeType::kBudaOp)
        {
            continue;
        }

        auto opmodels = graph_solver.at(node);
        graph_solver.set(node, *opmodels.begin());

        // After calling SET, we expect having only one OpModel available for this node.
        //
        auto opmodels_after_set = graph_solver.at(node);
        EXPECT_EQ(opmodels_after_set.mask.count(), 1);
    }

    balancer::legalizer::GraphSolverSolution solution = graph_solver.finish();

    EXPECT_EQ(solution.cut_edges.size(), 0);
    EXPECT_EQ(solution.selected_op_models.size(), 3);
}

TEST_F(GraphSolverResolveSanity, graphsolverforking)
{
    using balancer::legalizer::GraphSolver;
    using balancer::legalizer::GraphSolverSolution;
    balancer::BalancerConfig balancer_config = create_balancer_config();
    std::shared_ptr<balancer::BalancerCacheCollection> cache_collection = create_balancer_cache_collection();
    balancer::LegalOpModels valid_op_models =
        balancer::legalizer::get_legal_op_models(graph.get(), balancer_config, cache_collection);
    EXPECT_EQ(valid_op_models.size(), 3);
    GraphSolver graph_solver = get_graph_solver(balancer_config, cache_collection, graph.get(), valid_op_models);

    // Make a clone snapshot.
    //
    GraphSolver graph_solver_fork = graph_solver;

    auto topo_sort = tt::graphlib::topological_sort(*graph);

    {
        // Clone of a clone, within limited scope on purpose.
        //
        GraphSolver graph_solver_fork_2 = graph_solver_fork;
        bool flipflop = false;  // Alternate between first and last.

        // Simple - just set first available.
        //
        for (Node* node : topo_sort)
        {
            if (node->node_type() != graphlib::NodeType::kBudaOp)
            {
                continue;
            }

            auto opmodels = graph_solver.at(node);
            graph_solver.set(node, *opmodels.begin());

            auto opmodels2 = graph_solver_fork_2.at(node);
            if (flipflop)
            {
                int opModelCount = opmodels2.mask.count();
                int currentModel = 0;
                auto it = opmodels2.begin();
                while (currentModel < opModelCount - 1)
                {
                    it++;
                    currentModel++;
                }

                graph_solver_fork_2.set(node, *it);
            }
            else
            {
                graph_solver_fork_2.set(node, *opmodels2.begin());
            }

            flipflop ^= true;

            // After calling SET, we expect having only one OpModel available for this node.
            //
            auto opmodels_after_set = graph_solver.at(node);
            EXPECT_EQ(opmodels_after_set.mask.count(), 1);
            auto opmodels_after_set2 = graph_solver_fork_2.at(node);
            EXPECT_EQ(opmodels_after_set2.mask.count(), 1);
        }

        EXPECT_EQ(graph_solver_fork_2.get_cut_edges().size(), 0);
        EXPECT_EQ(graph_solver_fork_2.get_selected_op_models().size(), 3);
    }

    EXPECT_EQ(graph_solver.get_cut_edges().size(), 0);
    EXPECT_EQ(graph_solver.get_selected_op_models().size(), 3);

    // Invoking SET on original graph solver should not impact forks.
    //
    EXPECT_EQ(graph_solver_fork.get_selected_op_models().size(), 0);

    // This time set last available.
    //
    for (Node* node : topo_sort)
    {
        if (node->node_type() != graphlib::NodeType::kBudaOp)
        {
            continue;
        }

        auto opmodels = graph_solver_fork.at(node);
        int opModelCount = opmodels.mask.count();
        int currentModel = 0;
        auto it = opmodels.begin();
        while (currentModel < opModelCount - 1)
        {
            it++;
            currentModel++;
        }

        graph_solver_fork.set(node, *it);

        // After calling SET, we expect having only one OpModel available for this node.
        //
        auto opmodels_after_set = graph_solver_fork.at(node);
        EXPECT_EQ(opmodels_after_set.mask.count(), 1);
    }

    GraphSolverSolution secondary_solution = graph_solver_fork.finish();
    EXPECT_EQ(secondary_solution.cut_edges.size(), 0);
    EXPECT_EQ(secondary_solution.selected_op_models.size(), 3);
}

TEST_F(GraphSolverResolveSanity, graphsolverforking_cut)
{
    using balancer::legalizer::GraphSolver;
    using balancer::legalizer::GraphSolverSolution;
    balancer::BalancerConfig balancer_config = create_balancer_config();
    std::shared_ptr<balancer::BalancerCacheCollection> cache_collection = create_balancer_cache_collection();
    balancer::LegalOpModels valid_op_models =
        balancer::legalizer::get_legal_op_models(graph.get(), balancer_config, cache_collection);
    EXPECT_EQ(valid_op_models.size(), 3);
    GraphSolver graph_solver = get_graph_solver(balancer_config, cache_collection, graph.get(), valid_op_models);

    GraphSolver graph_solver_fork = graph_solver;

    auto topo_sort = tt::graphlib::topological_sort(*graph);
    bool should_cut = true;
    int edges_cut = 0;

    // For original just set first available, for fork lets cut a bit and set last.
    //
    for (Node* node : topo_sort)
    {
        if (node->node_type() != graphlib::NodeType::kBudaOp)
        {
            continue;
        }

        auto opmodels = graph_solver.at(node);
        graph_solver.set(node, *opmodels.begin());
        bool made_cut = false;
        Node* cutNode = nullptr;

        if (should_cut)
        {
            std::vector<graphlib::Edge> edges_to_cut;
            for (auto& edge : graph->user_data_edges(node))
            {
                Node* consumerNode = graph->node_by_id(edge.consumer_node_id);
                if (consumerNode->node_type() == graphlib::NodeType::kBudaOp and (!cutNode || cutNode == consumerNode))
                {
                    should_cut = edges_cut < 2;
                    edges_to_cut.push_back(edge);
                    cutNode = consumerNode;
                }
            }

            if (edges_to_cut.size() > 0)
            {
                edges_cut += edges_to_cut.size();
                graph_solver_fork.cut(edges_to_cut);
                made_cut = true;
            }
        }

        auto opmodels2 = graph_solver_fork.at(node);
        int opModel2Count = opmodels2.mask.count();
        if (made_cut and node == cutNode)
        {
            // If we have cut out this node from rest of the graph, we expect:
            // 1. All op models are available
            // 2. More op models compared to non-cut graph version
            //
            EXPECT_EQ(opModel2Count, opmodels2.p->size());
            EXPECT_GT(opModel2Count, opmodels.mask.count());
        }

        int currentModel = 0;
        auto it = opmodels2.begin();
        while (currentModel < opModel2Count - 1)
        {
            it++;
            currentModel++;
        }

        graph_solver_fork.set(node, *it);

        // After calling SET, we expect having only one OpModel available for this node.
        //
        auto opmodels_after_set = graph_solver.at(node);
        EXPECT_EQ(opmodels_after_set.mask.count(), 1);
    }

    EXPECT_EQ(graph_solver.get_cut_edges().size(), 0);
    EXPECT_EQ(graph_solver.get_selected_op_models().size(), 3);

    EXPECT_EQ(graph->virtual_node_count(), edges_cut);
    auto visible_global_graph = tt::graphlib::topological_sort(*graph);
    EXPECT_EQ(visible_global_graph.size(), topo_sort.size());
    GraphSolverSolution solution2 = graph_solver_fork.finish();
    EXPECT_EQ(graph->virtual_node_count(), 0);
    EXPECT_EQ(solution2.cut_edges.size(), edges_cut);
    EXPECT_EQ(solution2.selected_op_models.size(), 3);
}

TEST_F(GraphSolverResolveSanity, graphsolverforking_cut_all_forks)
{
    using balancer::legalizer::GraphSolver;
    using balancer::legalizer::GraphSolverSolution;
    balancer::BalancerConfig balancer_config = create_balancer_config();
    std::shared_ptr<balancer::BalancerCacheCollection> cache_collection = create_balancer_cache_collection();
    balancer::LegalOpModels valid_op_models =
        balancer::legalizer::get_legal_op_models(graph.get(), balancer_config, cache_collection);
    EXPECT_EQ(valid_op_models.size(), 3);
    GraphSolver graph_solver = get_graph_solver(balancer_config, cache_collection, graph.get(), valid_op_models);

    GraphSolver graph_solver_fork = graph_solver;

    auto topo_sort = tt::graphlib::topological_sort(*graph);
    bool should_cut = true;
    int edges_cut = 0;

    // For original just set first available, for fork lets cut a bit and set last.
    //
    for (Node* node : topo_sort)
    {
        if (node->node_type() != graphlib::NodeType::kBudaOp)
        {
            continue;
        }

        auto opmodels = graph_solver.at(node);
        graph_solver.set(node, *opmodels.begin());
        bool made_cut = false;
        Node* cutNode = nullptr;

        if (should_cut)
        {
            std::vector<graphlib::Edge> edges_to_cut;
            for (auto& edge : graph->user_data_edges(node))
            {
                Node* consumerNode = graph->node_by_id(edge.consumer_node_id);
                if (consumerNode->node_type() == graphlib::NodeType::kBudaOp and (!cutNode || cutNode == consumerNode))
                {
                    should_cut = edges_cut < 2;
                    edges_to_cut.push_back(edge);
                    cutNode = consumerNode;
                }
            }

            if (edges_to_cut.size() > 0)
            {
                edges_cut += edges_to_cut.size();
                graph_solver_fork.cut(edges_to_cut);
                graph_solver.cut(edges_to_cut);
                made_cut = true;
            }
        }

        auto opmodels2 = graph_solver_fork.at(node);
        int opModel2Count = opmodels2.mask.count();
        if (made_cut and node == cutNode)
        {
            // If we have cut out this node from rest of the graph, we expect:
            // 1. All op models are available
            // 2. More op models compared to non-cut graph version
            //
            EXPECT_EQ(opModel2Count, opmodels2.p->size());
            EXPECT_GT(opModel2Count, opmodels.mask.count());
        }

        int currentModel = 0;
        auto it = opmodels2.begin();
        while (currentModel < opModel2Count - 1)
        {
            it++;
            currentModel++;
        }

        graph_solver_fork.set(node, *it);

        // After calling SET, we expect having only one OpModel available for this node.
        //
        auto opmodels_after_set = graph_solver.at(node);
        EXPECT_EQ(opmodels_after_set.mask.count(), 1);
    }

    EXPECT_EQ(graph_solver.get_cut_edges().size(), edges_cut);
    EXPECT_EQ(graph_solver.get_selected_op_models().size(), 3);

    EXPECT_EQ(graph->virtual_node_count(), edges_cut * 2);
    auto visible_global_graph = tt::graphlib::topological_sort(*graph);
    EXPECT_EQ(visible_global_graph.size(), topo_sort.size());
    GraphSolverSolution solution2 = graph_solver_fork.finish();
    EXPECT_EQ(graph->virtual_node_count(), edges_cut);
    EXPECT_EQ(solution2.cut_edges.size(), edges_cut);
    EXPECT_EQ(solution2.selected_op_models.size(), 3);
}

TEST_F(GraphSolverResolveSanity, nop_insertion)
{
    balancer::BalancerConfig balancer_config = create_balancer_config();
    std::shared_ptr<balancer::BalancerCacheCollection> cache_collection = create_balancer_cache_collection();
    balancer::LegalOpModels valid_op_models =
        balancer::legalizer::get_legal_op_models(graph.get(), balancer_config, cache_collection);
    EXPECT_EQ(valid_op_models.size(), 3);
    legalizer::GraphSolver graph_solver =
        get_graph_solver(balancer_config, cache_collection, graph.get(), valid_op_models);

    auto topo_sort = tt::graphlib::topological_sort(*graph);

    // Simple - just set first available.
    //
    for (Node* node : topo_sort)
    {
        if (node->node_type() != graphlib::NodeType::kBudaOp)
        {
            continue;
        }

        if (node->name() == "add")
        {
            vector<balancer::legalizer::BufferInfo> buffer_info;
            int nop_count = 1;
            for (Edge edge : graph->operand_data_edges(node))
            {
                buffer_info.emplace_back(
                    balancer::legalizer::BufferInfo(edge, nop_count++, true /* hoist_tms */));
            }

            // Check virtual node count is as expected before and after buffering.
            //
            EXPECT_EQ(graph->virtual_node_count(), 0);
            std::vector<Node*> buffered_nodes = graph_solver.buffer(buffer_info);
            EXPECT_EQ(graph->virtual_node_count(), 3);

            for (Node* buf_node : buffered_nodes)
            {
                auto opmodels = graph_solver.at(buf_node);
                graph_solver.set(buf_node, *opmodels.begin());
            }
        }

        auto opmodels = graph_solver.at(node);
        graph_solver.set(node, *opmodels.begin());

        // After calling SET, we expect having only one OpModel available for this node.
        //
        auto opmodels_after_set = graph_solver.at(node);
        EXPECT_EQ(opmodels_after_set.mask.count(), 1);
    }

    balancer::legalizer::GraphSolverSolution solution = graph_solver.finish();

    EXPECT_EQ(graph->virtual_node_count(), 0);
    EXPECT_EQ(solution.cut_edges.size(), 0);
    EXPECT_EQ(solution.selected_op_models.size(), 6);
    EXPECT_EQ(graph->nodes().size(), 11);
}

TEST_F(GraphSolverResolveSanity, nop_insertion_forking)
{
    balancer::BalancerConfig balancer_config = create_balancer_config();
    std::shared_ptr<balancer::BalancerCacheCollection> cache_collection = create_balancer_cache_collection();
    balancer::LegalOpModels valid_op_models =
        balancer::legalizer::get_legal_op_models(graph.get(), balancer_config, cache_collection);
    EXPECT_EQ(valid_op_models.size(), 3);
    legalizer::GraphSolver graph_solver =
        get_graph_solver(balancer_config, cache_collection, graph.get(), valid_op_models);

    auto topo_sort = tt::graphlib::topological_sort(*graph);
    int initial_node_count = topo_sort.size();
    int nop_inserted_gs = 0;
    for (Node* node : topo_sort)
    {
        if (node->node_type() != graphlib::NodeType::kBudaOp)
        {
            continue;
        }

        if (node->name() == "add")
        {
            // Fork on buffer for operand edges of add op.
            //
            legalizer::GraphSolver gs_buffer_fork = graph_solver;
            vector<balancer::legalizer::BufferInfo> buffer_info;
            vector<balancer::legalizer::BufferInfo> buffer_info_fork;
            int nop_count = 1;
            int nop_inserted_gs_fork1 = 0;
            int nop_inserted_gs_fork2 = 0;

            // Insert different number of NOPs on two GS forks.
            //
            for (Edge edge : graph->operand_data_edges(node))
            {
                nop_inserted_gs += nop_count;
                buffer_info.emplace_back(
                    balancer::legalizer::BufferInfo(edge, nop_count++, false /* hoist_tms */));
                nop_inserted_gs_fork1 += nop_count;
                buffer_info_fork.emplace_back(
                    balancer::legalizer::BufferInfo(edge, nop_count, false /* hoist_tms */));
            }

            // Check if virtual node count and graph size are as expected before and after buffering.
            //
            EXPECT_EQ(graph->virtual_node_count(), 0);
            EXPECT_EQ(graph->nodes().size(), initial_node_count);
            std::vector<Node*> buffered_nodes = graph_solver.buffer(buffer_info);
            EXPECT_EQ(graph->virtual_node_count(), nop_inserted_gs);
            EXPECT_EQ(graph->nodes().size(), initial_node_count + graph->virtual_node_count());
            {
                // Create another GS fork after one buffering pass.
                // On purpose in limited scope to test virtual node cleanup.
                //
                legalizer::GraphSolver gs_buffer_fork2 = graph_solver;
                std::vector<Node*> buffered_nodes_fork = gs_buffer_fork.buffer(buffer_info_fork);
                EXPECT_EQ(graph->virtual_node_count(), nop_inserted_gs + nop_inserted_gs_fork1);
                EXPECT_EQ(graph->nodes().size(), initial_node_count + graph->virtual_node_count());
                int edges_cut = 0;

                for (Node* buf_node : buffered_nodes)
                {
                    auto opmodels = graph_solver.at(buf_node);
                    graph_solver.set(buf_node, *opmodels.begin());
                    opmodels = gs_buffer_fork2.at(buf_node);
                    gs_buffer_fork2.set(buf_node, *opmodels.begin());
                }

                vector<balancer::legalizer::BufferInfo> gs_buffer_fork2_buffer_info;
                for (Edge edge : graph->user_data_edges(node))
                {
                    gs_buffer_fork2_buffer_info.emplace_back(
                        balancer::legalizer::BufferInfo(edge, nop_count, false /* hoist_tms */));
                    nop_inserted_gs_fork2 += nop_count;
                }

                // Invoke buffer for second fork and check nodes afterwards.
                //
                std::vector<Node*> buffered_nodes_fork2 = gs_buffer_fork2.buffer(gs_buffer_fork2_buffer_info);
                EXPECT_EQ(graph->virtual_node_count(), nop_inserted_gs + nop_inserted_gs_fork1 + nop_inserted_gs_fork2);
                EXPECT_EQ(graph->nodes().size(), initial_node_count + graph->virtual_node_count());

                {
                    // Test cuting of virtual edge. Also showcase for GS external usage
                    // of GraphTraversalContext. Here it allows traversing graph in the context
                    // of gs_buffer_fork2 GS instance externally.
                    //
                    auto graph_traversal_context = gs_buffer_fork2.get_graph_traversal_context();
                    std::vector<graphlib::Edge> edges_to_cut;
                    for (auto& edge : graph->user_data_edges(node))
                    {
                        edges_to_cut.push_back(edge);
                    }

                    gs_buffer_fork2.cut(edges_to_cut);
                    edges_cut = edges_to_cut.size();
                }

                for (Node* buf_node : buffered_nodes_fork)
                {
                    auto opmodels = gs_buffer_fork.at(buf_node);
                    gs_buffer_fork.set(buf_node, *opmodels.begin());
                }

                for (Node* buf_node : buffered_nodes_fork2)
                {
                    auto opmodels = gs_buffer_fork2.at(buf_node);
                    auto it = opmodels.begin();
                    int opModelCount = opmodels.mask.count();
                    int currentModel = 0;
                    while (currentModel < opModelCount - 1)
                    {
                        it++;
                        currentModel++;
                    }

                    gs_buffer_fork2.set(buf_node, *it);
                }

                // Now lets compare graph traversal for each instance of GraphSolver.
                //
                {
                    auto graph_traversal_context = graph_solver.get_graph_traversal_context();
                    auto gs_subgraph = tt::graphlib::topological_sort(*graph);
                    EXPECT_EQ(gs_subgraph.size(), initial_node_count + nop_inserted_gs);
                }

                {
                    auto graph_traversal_context = gs_buffer_fork.get_graph_traversal_context();
                    auto gs_subgraph = tt::graphlib::topological_sort(*graph);
                    EXPECT_EQ(gs_subgraph.size(), initial_node_count + nop_inserted_gs_fork1);
                }

                {
                    auto graph_traversal_context = gs_buffer_fork2.get_graph_traversal_context();
                    auto gs_subgraph = tt::graphlib::topological_sort(*graph);
                    EXPECT_EQ(
                        gs_subgraph.size(), initial_node_count + nop_inserted_gs + nop_inserted_gs_fork2 + edges_cut);
                }

                auto visible_global_graph = tt::graphlib::topological_sort(*graph);
                EXPECT_EQ(visible_global_graph.size(), initial_node_count);
            }

            // Check if cleanup of deleted GS(gs_buffer_fork2) works as expected.
            //
            EXPECT_EQ(graph->virtual_node_count(), nop_inserted_gs + nop_inserted_gs_fork1);
            EXPECT_EQ(graph->nodes().size(), initial_node_count + graph->virtual_node_count());
        }

        auto opmodels = graph_solver.at(node);
        graph_solver.set(node, *opmodels.begin());

        // After calling SET, we expect having only one OpModel available for this node.
        //
        auto opmodels_after_set = graph_solver.at(node);
        EXPECT_EQ(opmodels_after_set.mask.count(), 1);
    }

    balancer::legalizer::GraphSolverSolution solution = graph_solver.finish();

    // After calling finish all virtual nodes should be gone and graph state is stable/final.
    //
    EXPECT_EQ(graph->virtual_node_count(), 0);

    EXPECT_EQ(solution.cut_edges.size(), 0);
    EXPECT_EQ(solution.selected_op_models.size(), 6);
    EXPECT_EQ(graph->nodes().size(), initial_node_count + nop_inserted_gs);
}

// Buffer on edge then fork GS. On forked GS buffer again between buffer and persisted node.
// Then make forked GS expire. This used to cause source GS to end up with unconnected node/edge removed in forked GS
// cleanup.
//
TEST_F(GraphSolverResolveSanity, nop_insertion_forking_snapshot)
{
    balancer::BalancerConfig balancer_config = create_balancer_config();
    std::shared_ptr<balancer::BalancerCacheCollection> cache_collection = create_balancer_cache_collection();
    balancer::LegalOpModels valid_op_models =
        balancer::legalizer::get_legal_op_models(graph.get(), balancer_config, cache_collection);
    EXPECT_EQ(valid_op_models.size(), 3);
    legalizer::GraphSolver graph_solver =
        get_graph_solver(balancer_config, cache_collection, graph.get(), valid_op_models);

    auto topo_sort = tt::graphlib::topological_sort(*graph);
    for (Node* node : topo_sort)
    {
        if (node->node_type() != graphlib::NodeType::kBudaOp)
        {
            continue;
        }

        if (node->name() == "add")
        {
            legalizer::GraphSolver gs_buffer_fork = graph_solver;
            vector<balancer::legalizer::BufferInfo> buffer_info;
            int nop_count = 1;

            for (Edge edge : graph->operand_data_edges(node))
            {
                buffer_info.emplace_back(balancer::legalizer::BufferInfo(edge, nop_count, false /* hoist_tms */));
            }

            gs_buffer_fork.buffer(buffer_info);

            {
                legalizer::GraphSolver gs_buffer_fork_2 = gs_buffer_fork;
                auto graph_traversal_context = gs_buffer_fork_2.get_graph_traversal_context();
                buffer_info.clear();

                for (Edge edge : graph->operand_data_edges(node))
                {
                    buffer_info.emplace_back(
                        balancer::legalizer::BufferInfo(edge, nop_count, false /* hoist_tms */));
                }

                gs_buffer_fork_2.buffer(buffer_info);
            }

            // After gs_buffer_fork_2 is gone we still need to have connected data operands.
            //
            auto graph_traversal_context = gs_buffer_fork.get_graph_traversal_context();
            EXPECT_TRUE(graph->data_operands(graph->data_operands(node).back()).back() != nullptr);
        }

        auto opmodels = graph_solver.at(node);
        graph_solver.set(node, *opmodels.begin());

        // After calling SET, we expect having only one OpModel available for this node.
        //
        auto opmodels_after_set = graph_solver.at(node);
        EXPECT_EQ(opmodels_after_set.mask.count(), 1);
    }

    balancer::legalizer::GraphSolverSolution solution = graph_solver.finish();

    // After calling finish all virtual nodes should be gone and graph state is stable/final.
    //
    EXPECT_EQ(graph->virtual_node_count(), 0);

    EXPECT_EQ(solution.cut_edges.size(), 0);
    EXPECT_EQ(solution.selected_op_models.size(), 3);
}

#ifdef DEBUG
struct GraphSolverResolveEdge : testing::Test
{
    std::unique_ptr<Graph> graph;

    GraphSolverResolveEdge()
    {
        graph = std::make_unique<Graph>(graphlib::IRLevel::IR_PYBUDA);

        auto in0 = create_input(*graph, "in0", graphlib::Shape::create({1, 1, 512, 256}));
        auto add1 = add_node<graphlib::PyOpNode>(*graph, "add1", "add", {}, {in0, in0});
        auto mul1 = add_node<graphlib::PyOpNode>(*graph, "mul1", "multiply", {}, {add1, add1});
        auto add2 = add_node<graphlib::PyOpNode>(*graph, "add2", "add", {}, {mul1, mul1});
        auto mul2 = add_node<graphlib::PyOpNode>(*graph, "mul2", "multiply", {}, {add1, add2});

        create_output(*graph, "out0", mul2);

        graph = prepare_graph_for_legalizer(graph.get());
    }
};

TEST_F(GraphSolverResolveEdge, resolveedgecase)
{
    auto topo_sort = tt::graphlib::topological_sort(*graph);

    // Set valid pairs of OpModels for GraphSolver(simulation).
    //
    const Node* add1 = graph->get_node_by_name("add1");
    const Node* mul1 = graph->get_node_by_name("mul1");
    const Node* add2 = graph->get_node_by_name("add2");
    const Node* mul2 = graph->get_node_by_name("mul2");

    LegalOpModels valid_op_models = {
        {add1, std::vector<OpModel>(2)},
        {mul1, std::vector<OpModel>(2)},
        {add2, std::vector<OpModel>(2)},
        {mul2, std::vector<OpModel>(2)},
    };

    both(valid_op_models[add1][0], valid_op_models[mul1][0]);
    both(valid_op_models[add1][1], valid_op_models[mul1][1]);
    both(valid_op_models[mul1][0], valid_op_models[add2][0]);
    both(valid_op_models[add1][1], valid_op_models[mul2][0]);

    balancer::BalancerConfig balancer_config = create_balancer_config();
    std::shared_ptr<balancer::BalancerCacheCollection> balancer_cache_collection = create_balancer_cache_collection();

    balancer_config.graph_solver_self_cut_type = balancer::legalizer::GraphSolverSelfCutType::FastCut;

    // Legalizer recalc would change fixed OpModel valid pairs, thats why we are disabling it for this test.
    //
    legalizer::GraphSolver graph_solver = legalizer::GraphSolver::create<UnitTestConstraint>(
        graph.get(), valid_op_models, balancer_config, balancer_cache_collection, false /*use_op_model_recalculation*/);

    // Simple - just set first available.
    //
    for (Node* node : topo_sort)
    {
        if (node->node_type() != graphlib::NodeType::kBudaOp)
        {
            continue;
        }

        auto opmodels = graph_solver.at(node);
        graph_solver.set(node, *opmodels.begin());

        // After calling SET, we expect having only one OpModel available for this node.
        //
        auto opmodels_after_set = graph_solver.at(node);
        EXPECT_EQ(opmodels_after_set.mask.count(), 1);
    }

    EXPECT_EQ(graph->virtual_node_count(), 2);
    balancer::legalizer::GraphSolverSolution solution = graph_solver.finish();

    EXPECT_EQ(graph->virtual_node_count(), 0);
    EXPECT_EQ(solution.cut_edges.size(), 2);
    EXPECT_EQ(solution.selected_op_models.size(), 4);
}
#endif

struct ForkGraph : public BudaGraphTest
{
   protected:
    virtual std::vector<OpType*> create_graph() override
    {
        int r = 32;
        int c = 32;

        auto a = create_activation(1, 1, r, c);
        auto b = create_activation(1, 1, r, c);

        x = create_op("add", {a, b});
        f0 = create_op("void", {x}, 1, 1, r, c);
        f1 = create_op("void", {x}, 1, 1, r, c);
        out = create_op("add", {f0, f1});

        return {out};
    }

    OpType* x;
    OpType* f0;
    OpType* f1;
    OpType* out;
};

TEST_F(ForkGraph, non_overlapping_forks)
{
    int even_counter = 0;
    int odd_counter = 1;
    auto evens = [&even_counter](OpModel& a, OpModel& b)
    {
        if (even_counter++ % 2 == 0)
            b2a(a, b);
    };
    auto odds = [&odd_counter](OpModel& a, OpModel& b)
    {
        if (odd_counter++ % 2 == 0)
            b2a(a, b);
    };

    std::vector<OpModel> x_op_models(8);
    std::vector<OpModel> f0_op_models(8);
    std::vector<OpModel> f1_op_models(8);
    std::vector<OpModel> out_op_models(8);

    // Fork point
    {
        even_counter = 0;
        odd_counter = 1;
        cross(x_op_models, f0_op_models, odds);
        cross(x_op_models, f1_op_models, evens);
    }

    // Fork paths
    {
        cross(f0_op_models, out_op_models, b2a);
        cross(f1_op_models, out_op_models, b2a);
    }

    // Join point
    {
        even_counter = 0;
        odd_counter = 1;
        cross(out_op_models, f0_op_models, evens);
        cross(out_op_models, f1_op_models, odds);
    }

    LegalOpModels valid_op_models = {
        {x, x_op_models},
        {f0, f0_op_models},
        {f1, f1_op_models},
        {out, out_op_models},
    };

    graphlib::Graph* graph = get_graph();
    balancer::BalancerConfig balancer_config = create_balancer_config();
    std::shared_ptr<balancer::BalancerCacheCollection> balancer_cache_collection = create_balancer_cache_collection();

    balancer_config.graph_solver_self_cut_type = balancer::legalizer::GraphSolverSelfCutType::FastCut;
    legalizer::GraphSolver graph_solver = legalizer::GraphSolver::create<UnitTestConstraint>(
        graph, valid_op_models, balancer_config, balancer_cache_collection, false /*use_op_model_recalculation*/);

    for (auto* node : std::vector<OpType*>{x, f0, f1, out})
    {
        auto op_model_range = graph_solver.at(node);
        ASSERT_NE(op_model_range.begin(), op_model_range.end());
        graph_solver.set(node, *op_model_range.begin());
    }

    legalizer::GraphSolverSolution solution = graph_solver.finish();
    EXPECT_EQ(solution.selected_op_models.size(), 4);
    EXPECT_EQ(graph->virtual_node_count(), 0);
}

struct AggregateInputGraph : public BudaGraphTest, public testing::WithParamInterface<int>
{
    struct Constraint : public legalizer::Constraint
    {
        Constraint(
            const DeviceConfig& device_config,
            std::shared_ptr<balancer::BalancerCacheCollection> balancer_cache_collection) :
            legalizer::Constraint(device_config, balancer_cache_collection)
        {
        }

        virtual std::pair<legalizer::EdgeCost, legalizer::ConstraintFailureReason> queue_to_op_cost(
            graphlib::Graph const*, graphlib::Edge, std::optional<OpModel>, OpModel const&) override
        {
            return std::make_pair(legalizer::EdgeCost(0, 0, 0, 1), legalizer::NoConstraintFailure);
        }

        virtual std::pair<legalizer::EdgeCost, legalizer::ConstraintFailureReason> op_to_queue_cost(
            graphlib::Graph const*, graphlib::Edge, OpModel const&, std::optional<OpModel>) override
        {
            return std::make_pair(legalizer::EdgeCost{}, legalizer::NoConstraintFailure);
        }

        virtual std::pair<legalizer::EdgeCost, legalizer::ConstraintFailureReason> op_to_op_cost(
            graphlib::Graph const*, graphlib::Edge, OpModel const&, OpModel const&) override
        {
            return std::make_pair(legalizer::EdgeCost{}, legalizer::NoConstraintFailure);
        }
    };

   protected:
    virtual std::vector<OpType*> create_graph() override
    {
        int r = 32;
        int c = 32;

        num_inputs = GetParam();

        std::vector<graphlib::Node*> inputs;
        for (int i = 0; i < num_inputs; ++i) inputs.push_back(create_activation(1, 1, r, c));

        nary = create_op("void", inputs, 1, 1, r, c);
        return {nary};
    }

    int num_inputs = 0;
    OpType* nary;
};

TEST_P(AggregateInputGraph, aggregate_input_test)
{
    LegalOpModels legal_op_models = {
        {nary, std::vector<OpModel>(num_inputs)},
    };

    auto test = [this, &legal_op_models]()
    {
        legalizer::GraphSolver graph_solver = legalizer::GraphSolver::create<AggregateInputGraph::Constraint>(
            this->get_graph(),
            legal_op_models,
            create_balancer_config(),
            create_balancer_cache_collection(),
            false /*use_op_model_recalculation*/);
        legalizer::GraphSolverSolution solution = graph_solver.finish();
    };

    if (num_inputs > legalizer::EdgeCost::kMaxDRAMInQueues)
    {
        EXPECT_THROW({ test(); }, balancer::BalancerError);
    }
    else
    {
        test();
    }
}

INSTANTIATE_TEST_SUITE_P(
    AggregateInputTest,
    AggregateInputGraph,
    testing::Values(
        1,
        2,
        legalizer::EdgeCost::kMaxDRAMInQueues - 1,
        legalizer::EdgeCost::kMaxDRAMInQueues,
        legalizer::EdgeCost::kMaxDRAMInQueues + 1));

struct JsonGraph : public BudaGraphTest, public testing::WithParamInterface<std::string>
{
   protected:
    using NodeId = int;

    graphlib::Node* create_node(NodeId node_id)
    {
        graphlib::Node*& node = nodes[node_id];
        if (node)
            return node;

        auto operand_match = operands.find(node_id);
        if (operand_match == operands.end())
        {
            node = create_activation(1, 1, 32, 32);
        }
        else
        {
            std::vector<graphlib::Node*> operand_nodes;
            for (auto operand : operand_match->second) operand_nodes.push_back(create_node(operand));
            node = create_op("void", operand_nodes);
        }

        auto key = std::to_string(node_id);
        auto name_match = test.node_id_to_name.find(key);
        node->set_name(name_match == test.node_id_to_name.end() ? key : name_match->second);
        return node;
    }

    virtual std::vector<OpType*> create_graph() override
    {
        auto json_file = GetParam();
        std::string json_dir = "./pybuda/csrc/balancer/tests/json/";
        std::ifstream input(json_dir + json_file);
        nlohmann::json j = nlohmann::json::parse(input);
        from_json(j, test);

        // Init operands + users first
        for (auto const& edge : test.edges)
        {
            auto& ops = operands[edge.consumer];
            ops.resize(std::max((int)ops.size(), edge.input_port + 1));
            ops[edge.input_port] = edge.producer;
            users[edge.producer].push_back(edge.consumer);
        }

        // Create nodes / graph
        for (auto const& edge : test.edges)
        {
            create_node(edge.producer);
            create_node(edge.consumer);
        }

        // Create outputs
        std::vector<OpType*> outputs;
        for (auto const& edge : test.edges)
        {
            auto user_match = users.find(edge.consumer);
            if (user_match == users.end())
            {
                OpType* op = dynamic_cast<OpType*>(nodes.at(edge.consumer));
                TT_ASSERT(op);
                outputs.push_back(op);
            }
        }

        return outputs;
    }

    JsonTest test;
    std::unordered_map<NodeId, graphlib::Node*> nodes;
    std::unordered_map<NodeId, std::vector<NodeId>> operands;
    std::unordered_map<NodeId, std::vector<NodeId>> users;
};

TEST_P(JsonGraph, json_test)
{
    auto create_op_model_id = [](NodeId node_id, int op_model_index) -> std::uint64_t
    {
        std::uint64_t l = static_cast<std::uint64_t>(node_id);
        std::uint64_t u = static_cast<std::uint64_t>(op_model_index);
        return (u << 32) | l;
    };

    std::unordered_map<std::uint64_t, OpModel> op_model_pool;
    LegalOpModels legal_op_models;
    for (auto const& edge : test.edges)
    {
        if (nodes.at(edge.producer)->node_type() == graphlib::NodeType::kInput)
            continue;

        std::vector<OpModel>& producer_op_models = legal_op_models[nodes.at(edge.producer)];
        std::vector<OpModel>& consumer_op_models = legal_op_models[nodes.at(edge.consumer)];
        for (auto [source_idx, target_idx] : edge.paths)
        {
            auto producer_id = create_op_model_id(edge.producer, source_idx);
            auto consumer_id = create_op_model_id(edge.consumer, target_idx);

            OpModel& producer_op_model = op_model_pool[producer_id];
            producer_op_model.id.id = producer_id;  // override the id
            OpModel& consumer_op_model = op_model_pool[consumer_id];
            consumer_op_model.id.id = consumer_id;  // override the id

            both(producer_op_model, consumer_op_model);  // relate them

            if (std::find(producer_op_models.begin(), producer_op_models.end(), producer_op_model) ==
                producer_op_models.end())
                producer_op_models.push_back(producer_op_model);
            if (std::find(consumer_op_models.begin(), consumer_op_models.end(), consumer_op_model) ==
                consumer_op_models.end())
                consumer_op_models.push_back(consumer_op_model);
        }
    }

    graphlib::Graph* graph = get_graph();
    balancer::BalancerConfig balancer_config = create_balancer_config();
    std::shared_ptr<balancer::BalancerCacheCollection> balancer_cache_collection = create_balancer_cache_collection();
    balancer_config.graph_solver_self_cut_type =
        test.has_flag("FastCut") ? legalizer::GraphSolverSelfCutType::FastCut : legalizer::GraphSolverSelfCutType::None;
    legalizer::GraphSolver graph_solver = legalizer::GraphSolver::create<UnitTestConstraint>(
        graph, legal_op_models, balancer_config, balancer_cache_collection, false /*use_op_model_recalculation*/);
    legalizer::GraphSolverSolution solution = graph_solver.finish();
    EXPECT_EQ(graph->virtual_node_count(), 0);
    EXPECT_EQ(solution.selected_op_models.size(), 0);
}

INSTANTIATE_TEST_SUITE_P(JsonTest, JsonGraph, testing::Values("ghostnet_subgraph.json"));
}  // namespace tt::test
