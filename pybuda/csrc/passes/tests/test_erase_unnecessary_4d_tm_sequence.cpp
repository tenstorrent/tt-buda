// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "gtest/gtest.h"
#include "passes/erase_unnecessary_4d_tm_sequence.hpp"
#include "test/graph_api.hpp"

using namespace tt;

template <int NumOperands>
struct EraseUnnecessary4DSeq : testing::Test
{
    graphlib::Graph *graph;

    //template <int NumOperands = NumOperands>
    EraseUnnecessary4DSeq()
    {
        // Initialize graph
        graph = new graphlib::Graph(graphlib::IRLevel::IR_PYBUDA);

        // Graph definition
        auto input_0 = create_input(*graph, "input_0", graphlib::Shape::create({1, NumOperands*58, 64, 64})); 

        auto reshape_0 = add_node<graphlib::PyOpNode>(*graph, "reshape_0", "reshape", {NumOperands, 58, 64, 64}, {input_0});
        auto transpose_0 = add_node<graphlib::PyOpNode>(
            *graph,
            "transpose_0",
            graphlib::OpType("transpose", {}, {}, {{"dim0", -4}, {"dim1", -3}, {"z_dim_slice", -1}}),
            {reshape_0});
        auto reshape_1 = add_node<graphlib::PyOpNode>(*graph, "reshape_1", "reshape", {1, NumOperands*58, 64, 64}, {transpose_0});

        create_output(*graph, "output_0", reshape_1);
    }
};

using EraseUnnecessary4DSeqTwoOps = EraseUnnecessary4DSeq<2>;
using EraseUnnecessary4DSeqThreeOps = EraseUnnecessary4DSeq<3>;

TEST_F(EraseUnnecessary4DSeqTwoOps, two_operands)
{
    // Run stack/slice fuse pass
    passes::erase_unnecessary_4d_tm_sequence(graph);

    // Reshape and transpose op should be gone, and replaced with select ops and interleave op
    for (auto node : graph->nodes())
    {
        if (node->node_type() == tt::graphlib::kPyOp)
        {
            EXPECT_NE(node->as<graphlib::PyOpNode>()->op_type().op, "reshape");
            EXPECT_NE(node->as<graphlib::PyOpNode>()->op_type().op, "transpose");
        }
    }
    EXPECT_EQ(graph->nodes().size(), 5);
}

TEST_F(EraseUnnecessary4DSeqThreeOps, three_operands)
{
    // Run stack/slice fuse pass
    passes::erase_unnecessary_4d_tm_sequence(graph);

    // Reshape and transpose op should be gone, and replaced with select ops and interleave op
    for (auto node : graph->nodes())
    {
        if (node->node_type() == tt::graphlib::kPyOp)
        {
            EXPECT_NE(node->as<graphlib::PyOpNode>()->op_type().op, "reshape");
            EXPECT_NE(node->as<graphlib::PyOpNode>()->op_type().op, "transpose");
        }
    }
    EXPECT_EQ(graph->nodes().size(), 6);
}

TEST_F(EraseUnnecessary4DSeqTwoOps, na1)
{
    // Extend base test - add another reshape op before the last reshape 
    graphlib::Node *transpose_0 = graph->get_node_by_name("transpose_0");
    add_node<graphlib::PyOpNode>(*graph, "reshape_2", "reshape", {58,4,32,64}, {transpose_0}, {0, 0});

    // Run stack/slice fuse pass
    passes::erase_unnecessary_4d_tm_sequence(graph);

    // Reshape and transpose op should be gone, and replaced with select ops and interleave op
    for (auto node : graph->nodes())
    {
        if (node->node_type() == tt::graphlib::kPyOp)
        {
            EXPECT_NE(node->as<graphlib::PyOpNode>()->op_type().op, "select");
            EXPECT_NE(node->as<graphlib::PyOpNode>()->op_type().op, "interleave");
        }
    }
    EXPECT_EQ(graph->nodes().size(), 6);
}

TEST_F(EraseUnnecessary4DSeqTwoOps, na2)
{
    // Extend base test - add another reshape op before the last reshape 
    graphlib::Node *transpose_0 = graph->get_node_by_name("transpose_0");
    auto add_0 = add_node<graphlib::PyOpNode>(*graph, "add_0", "add", {}, {transpose_0, transpose_0});
    create_output(*graph, "output_1", add_0);

    // Run stack/slice fuse pass
    passes::erase_unnecessary_4d_tm_sequence(graph);

    // Reshape and transpose op should be gone, and replaced with select ops and interleave op
    for (auto node : graph->nodes())
    {
        if (node->node_type() == tt::graphlib::kPyOp)
        {
            EXPECT_NE(node->as<graphlib::PyOpNode>()->op_type().op, "select");
            EXPECT_NE(node->as<graphlib::PyOpNode>()->op_type().op, "interleave");
        }
    }
    EXPECT_EQ(graph->nodes().size(), 7);
}

