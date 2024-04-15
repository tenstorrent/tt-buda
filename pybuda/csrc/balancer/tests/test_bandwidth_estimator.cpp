// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest-param-test.h>

#include "balancer/bandwidth_bucket.hpp"
#include "balancer/data_movement_bw_estimation.hpp"
#include "balancer/legalizer/legalizer.hpp"
#include "balancer/types.hpp"
#include "graph_lib/graph.hpp"
#include "gtest/gtest.h"
#include "test/graph_api.hpp"
#include "test_balancer_utils.hpp"

namespace tt::test
{

using namespace balancer;

struct TestNocBandwidthEstimator : testing::Test
{
    std::unique_ptr<Graph> graph;

    void SetUp() override
    {
        graph = std::make_unique<Graph>(graphlib::IRLevel::IR_PYBUDA);

        graphlib::Shape shape = graphlib::Shape::create({1, 1, 512, 160});

        auto in0_a = create_input(*graph, "in0_a", graphlib::Shape::create({1, 1, shape[2], 256}));    // 1x1x512x256
        auto in0_b = create_input(*graph, "in0_b", graphlib::Shape::create({1, 1, 256, shape[3]}));    // 1x1x256x160
        auto matmul0 = add_node<graphlib::PyOpNode>(*graph, "matmul0", "matmul", {}, {in0_a, in0_b});  // 1x1x512x160

        auto in1_a = create_input(*graph, "in1_a", graphlib::Shape::create({1, 1, shape[3], 128}));    // 1x1x160x128
        auto in1_b = create_input(*graph, "in1_b", graphlib::Shape::create({1, 1, 128, shape[2]}));    // 1x1x128x512
        auto matmul1 = add_node<graphlib::PyOpNode>(*graph, "matmul1", "matmul", {}, {in1_a, in1_b});  // 1x1x160x512

        auto matmul2 =
            add_node<graphlib::PyOpNode>(*graph, "matmul2", "matmul", {}, {matmul0, matmul1});  // 1x1x512x512

        create_output(*graph, "out0", matmul2);

        graph = prepare_graph_for_legalizer(graph.get());
        graph->set_microbatch(64);
    }
};

TEST_F(TestNocBandwidthEstimator, get_bandwidth)
{
    balancer::BalancerConfig balancer_config = create_balancer_config();
    std::shared_ptr<balancer::BalancerCacheCollection> cache_collection = create_balancer_cache_collection();
    balancer_config.enable_t_streaming = true;
    balancer::LegalOpModels valid_op_models =
        balancer::legalizer::get_legal_op_models(graph.get(), balancer_config, cache_collection);

    legalizer::GraphSolver graph_solver =
        get_graph_solver(balancer_config, cache_collection, graph.get(), valid_op_models);
    auto matmul0 = graph->get_node_by_name("matmul0");
    auto matmul1 = graph->get_node_by_name("matmul1");
    auto matmul2 = graph->get_node_by_name("matmul2");

    auto topo_sort = tt::graphlib::topological_sort(*graph);

    // Set first available OpModel for matmul0 and matmul1.
    auto opmodels_available = graph_solver.at(matmul0);
    auto mm_0_model = *opmodels_available.begin();
    graph_solver.set(matmul0, mm_0_model);

    opmodels_available = graph_solver.at(matmul1);
    auto mm_1_model = *opmodels_available.begin();
    graph_solver.set(matmul1, mm_1_model);

    auto legal_op_models_on_consumer = graph_solver.at(matmul2);

    double best_bw_sum = 0.;
    OpModel best_model;

    for (auto& consumer_op_model : legal_op_models_on_consumer)
    {
        BandwidthBucket bb0 = get_bandwidth_estimation(
            graph.get(), graph->get_edges(matmul0, matmul2)[0], mm_0_model, consumer_op_model, false);
        BandwidthBucket bb1 = get_bandwidth_estimation(
            graph.get(), graph->get_edges(matmul1, matmul2)[0], mm_1_model, consumer_op_model, false);

        if (bb0.get_bandwidth() + bb1.get_bandwidth() > best_bw_sum)
        {
            best_bw_sum = bb0.get_bandwidth() + bb1.get_bandwidth();
            best_model = consumer_op_model;
        }
    }

    EXPECT_GT(best_bw_sum, 0.);

    graph_solver.set(matmul2, best_model);

    balancer::legalizer::GraphSolverSolution solution = graph_solver.finish();
    EXPECT_EQ(solution.selected_op_models.size(), 3);
}

}  // namespace tt::test