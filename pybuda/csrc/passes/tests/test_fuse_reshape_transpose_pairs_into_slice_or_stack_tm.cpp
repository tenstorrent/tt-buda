// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "gtest/gtest.h"
#include "passes/fuse_reshape_transpose_into_slice.hpp"
#include "test/graph_api.hpp"

using namespace tt;

struct FuseReshapeTransposeIntoHSlice : testing::Test
{
    graphlib::Graph *graph;

    FuseReshapeTransposeIntoHSlice()
    {
        // Initialize graph
        graph = new graphlib::Graph(graphlib::IRLevel::IR_PYBUDA);

        // Graph definition
        auto input_0 = create_input(*graph, "input_0", graphlib::Shape::create({1, 1, 1, 32}));
        auto input_1 = create_input(*graph, "input_1", graphlib::Shape::create({1, 1, 32, 4096}));
        auto matmul_0 = add_node<graphlib::PyOpNode>(*graph, "matmul_0", "matmul", {}, {input_0, input_1});

        auto reshape_0 = add_node<graphlib::PyOpNode>(*graph, "reshape_0", "reshape", {1, 1, 64, 64}, {matmul_0});
        auto transpose_0 = add_node<graphlib::PyOpNode>(*graph, "transpose_0", graphlib::OpType("transpose", {}, {}, {{"dim0", 1}, {"dim1", 2}, {"z_dim_slice", 1}}), {reshape_0});

        auto input_2 = create_input(*graph, "input_2", graphlib::Shape::create({1, 1, 1, 128}));
        auto matmul_1 = add_node<graphlib::PyOpNode>(*graph, "matmul_1", "matmul", {}, {transpose_0, input_2});

        create_output(*graph, "output_0", matmul_1);
    }

    virtual void TearDown() {
        delete graph;
    }
};

TEST_F(FuseReshapeTransposeIntoHSlice, basic_case)
{
    // Run stack/slice fuse pass
    passes::fuse_reshape_transpose_pairs_into_slice_or_stack_tm(graph);

    // Reshape and transpose op should be gone, and replaced with new fused hslice op
    for (auto node : graph->nodes())
    {
        if (node->node_type() == tt::graphlib::kPyOp)
        {
            EXPECT_NE(node->as<graphlib::PyOpNode>()->op_type().op, "reshape");
            EXPECT_NE(node->as<graphlib::PyOpNode>()->op_type().op, "transpose");
        }
    }
    EXPECT_EQ(graph->nodes().size(), 7);
}

TEST_F(FuseReshapeTransposeIntoHSlice, with_single_commute)
{
    // Expand base test - add single elementwise commutable ops on path between transpose and reshape
    auto input_3 = create_input(*graph, "input_3", graphlib::Shape::create({1, 1, 64, 64}));
    graphlib::Node *reshape_0 = graph->get_node_by_name("reshape_0");
    add_node<graphlib::PyOpNode>(*graph, "add_0", "add", {}, {reshape_0, input_3}, {0, 0});

    // Run stack/slice fuse pass
    passes::fuse_reshape_transpose_pairs_into_slice_or_stack_tm(graph);

    // Reshape and transpose op should be gone, and replaced with new fused hslice op
    for (auto node : graph->nodes())
    {
        if (node->node_type() == tt::graphlib::kPyOp)
        {
            EXPECT_NE(node->as<graphlib::PyOpNode>()->op_type().op, "reshape");
            EXPECT_NE(node->as<graphlib::PyOpNode>()->op_type().op, "transpose");
        }
    }
    EXPECT_EQ(graph->nodes().size(), 9);
}

TEST_F(FuseReshapeTransposeIntoHSlice, with_multiple_commute)
{
    // Expand base test - add few elementwise commutable ops on path between transpose and reshape
    auto input_3 = create_input(*graph, "input_3", graphlib::Shape::create({1, 1, 64, 64}));
    graphlib::Node *reshape_0 = graph->get_node_by_name("reshape_0");
    auto add_0 = add_node<graphlib::PyOpNode>(*graph, "add_0", "add", {}, {reshape_0, input_3}, {0, 0});
    auto add_1 = add_node<graphlib::PyOpNode>(*graph, "add_1", "add", {}, {add_0, add_0}, {0, 0});
    auto multiply_0 = add_node<graphlib::PyOpNode>(*graph, "multiply_0", "multiply", {}, {add_1, add_1}, {0, 0});
    add_node<graphlib::PyOpNode>(*graph, "subtract_0", "subtract", {}, {multiply_0, multiply_0}, {0, 0});

    // Run stack/slice fuse pass
    passes::fuse_reshape_transpose_pairs_into_slice_or_stack_tm(graph);

    // Reshape and transpose op should be gone, and replaced with new fused hslice op
    for (auto node : graph->nodes())
    {
        if (node->node_type() == tt::graphlib::kPyOp)
        {
            EXPECT_NE(node->as<graphlib::PyOpNode>()->op_type().op, "reshape");
            EXPECT_NE(node->as<graphlib::PyOpNode>()->op_type().op, "transpose");
        }
    }
    EXPECT_EQ(graph->nodes().size(), 12);
}

TEST_F(FuseReshapeTransposeIntoHSlice, with_multiple_forked_commute)
{
    // Expand base test - add few elementwise commutable ops on path between transpose and reshape
    // which are forked between themselves
    auto input_3 = create_input(*graph, "input_3", graphlib::Shape::create({1, 1, 64, 64}));
    graphlib::Node *reshape_0 = graph->get_node_by_name("reshape_0");
    auto add_0 = add_node<graphlib::PyOpNode>(*graph, "add_0", "add", {}, {reshape_0, input_3}, {0, 0});
    auto add_1 = add_node<graphlib::PyOpNode>(*graph, "add_1", "add", {}, {add_0, add_0}, {0, 0});
    auto multiply_0 = add_node<graphlib::PyOpNode>(*graph, "multiply_0", "multiply", {}, {add_0, add_1}, {1, 0});
    add_node<graphlib::PyOpNode>(*graph, "subtract_0", "subtract", {}, {multiply_0, input_3}, {0, 0});

    // Run stack/slice fuse pass
    passes::fuse_reshape_transpose_pairs_into_slice_or_stack_tm(graph);

    // Reshape and transpose op should be gone, and replaced with new fused hslice op
    for (auto node : graph->nodes())
    {
        if (node->node_type() == tt::graphlib::kPyOp)
        {
            EXPECT_NE(node->as<graphlib::PyOpNode>()->op_type().op, "reshape");
            EXPECT_NE(node->as<graphlib::PyOpNode>()->op_type().op, "transpose");
        }
    }
    EXPECT_EQ(graph->nodes().size(), 12);
}

TEST_F(FuseReshapeTransposeIntoHSlice, with_multiple_forked_non_commutable)
{
    // Expand base test - add few elementwise commutable and one non-commutable matmul op on path between
    // transpose and reshape
    auto input_3 = create_input(*graph, "input_3", graphlib::Shape::create({1, 1, 64, 64}));
    graphlib::Node *reshape_0 = graph->get_node_by_name("reshape_0");
    auto add_0 = add_node<graphlib::PyOpNode>(*graph, "add_0", "add", {}, {reshape_0, input_3}, {0, 0});
    auto add_1 = add_node<graphlib::PyOpNode>(*graph, "add_1", "add", {}, {add_0, add_0}, {0, 0});
    auto matmul_2 = add_node<graphlib::PyOpNode>(*graph, "matmul_2", "matmul", {}, {add_0, add_1}, {1, 0});
    add_node<graphlib::PyOpNode>(*graph, "subtract_0", "subtract", {}, {matmul_2, input_3}, {0, 0});

    // Run stack/slice fuse pass
    passes::fuse_reshape_transpose_pairs_into_slice_or_stack_tm(graph);

    // Reshape and transpose op should be still in a graph
    std::string op_type_name;
    bool reshape_in_graph = false;
    bool transpose_in_graph = false;
    for (auto node : graph->nodes())
    {
        if (node->node_type() == tt::graphlib::kPyOp)
        {
            op_type_name = node->as<graphlib::PyOpNode>()->op_type().op;
            if (op_type_name == "reshape")
                reshape_in_graph = true;
            else if (op_type_name == "transpose")
                transpose_in_graph = true;
        }
    }
    EXPECT_EQ(reshape_in_graph, true);
    EXPECT_EQ(transpose_in_graph, true);
    EXPECT_EQ(graph->nodes().size(), 13);
}

// Multiple pairs on forked paths
TEST_F(FuseReshapeTransposeIntoHSlice, with_multiple_suitable_pairs)
{
    // Expand base test - add few elementwise commutable ops on path between transpose and reshape
    // which are forked between themselves. Also, add additional suitable pair for hslice
    auto input_3 = create_input(*graph, "input_3", graphlib::Shape::create({1, 1, 64, 64}));
    graphlib::Node *reshape_0 = graph->get_node_by_name("reshape_0");
    auto add_0 = add_node<graphlib::PyOpNode>(*graph, "add_0", "add", {}, {reshape_0, input_3}, {0, 0});
    auto add_1 = add_node<graphlib::PyOpNode>(*graph, "add_1", "add", {}, {add_0, add_0}, {0, 0});
    auto multiply_0 = add_node<graphlib::PyOpNode>(*graph, "multiply_0", "multiply", {}, {add_0, add_1}, {1, 0});
    add_node<graphlib::PyOpNode>(*graph, "subtract_0", "subtract", {}, {multiply_0, input_3}, {0, 0});

    graphlib::Node *transpose_0 = graph->get_node_by_name("transpose_0");
    auto reshape_1 =
        add_node<graphlib::PyOpNode>(*graph, "reshape_1", "reshape", {1, 1, 1, 4096}, {transpose_0}, {0, 0});
    auto reshape_2 = add_node<graphlib::PyOpNode>(*graph, "reshape_2", "reshape", {1, 1, 64, 64}, {reshape_1}, {0, 0});
    add_node<graphlib::PyOpNode>(*graph, "transpose_1", graphlib::OpType("transpose", {}, {}, {{"dim0", 1}, {"dim1", 2}, {"z_dim_slice", 1}}), {reshape_2}, {0, 0});

    // Run stack/slice fuse pass
    passes::fuse_reshape_transpose_pairs_into_slice_or_stack_tm(graph);

    // All transposes should be gone and replaced with new fused hslice ops together with their reshapes. Only
    // one flattening reshape will stay in the graph
    int reshape_count = 0;
    for (auto node : graph->nodes())
    {
        if (node->node_type() == tt::graphlib::kPyOp)
        {
            if (node->as<graphlib::PyOpNode>()->op_type().op == "reshape")
                reshape_count++;
            EXPECT_NE(node->as<graphlib::PyOpNode>()->op_type().op, "transpose");
        }
    }
    EXPECT_EQ(reshape_count, 1);
    EXPECT_EQ(graph->nodes().size(), 14);
}

struct FuseTransposeReshapeIntoHStack : testing::Test
{
    graphlib::Graph *graph;

    FuseTransposeReshapeIntoHStack()
    {
        // Initialize graph
        graph = new graphlib::Graph(graphlib::IRLevel::IR_PYBUDA);

        // Graph definition
        auto input_0 = create_input(*graph, "input_0", graphlib::Shape::create({1, 64, 1, 32}));
        auto input_1 = create_input(*graph, "input_1", graphlib::Shape::create({1, 64, 32, 64}));
        auto matmul_0 = add_node<graphlib::PyOpNode>(*graph, "matmul_0", "matmul", {}, {input_0, input_1});

        auto transpose_0 = add_node<graphlib::PyOpNode>(*graph, "transpose_0", graphlib::OpType("transpose", {}, {}, {{"dim0", 1}, {"dim1", 2}, {"z_dim_slice", 1}}), {matmul_0});
        auto reshape_0 = add_node<graphlib::PyOpNode>(*graph, "reshape_0", "reshape", {1, 1, 1, 4096}, {transpose_0});

        auto input_2 = create_input(*graph, "input_2", graphlib::Shape::create({1, 1, 4096, 64}));
        auto matmul_1 = add_node<graphlib::PyOpNode>(*graph, "matmul_1", "matmul", {}, {reshape_0, input_2});

        create_output(*graph, "output_0", matmul_1);
    }

    virtual void TearDown() {
        delete graph;
    }
};

TEST_F(FuseTransposeReshapeIntoHStack, basic_case)
{
    // Run stack/slice fuse pass
    passes::fuse_reshape_transpose_pairs_into_slice_or_stack_tm(graph);

    // Reshape and transpose op should be gone, and replaced with new fused hstack op
    for (auto node : graph->nodes())
    {
        if (node->node_type() == tt::graphlib::kPyOp)
        {
            EXPECT_NE(node->as<graphlib::PyOpNode>()->op_type().op, "reshape");
            EXPECT_NE(node->as<graphlib::PyOpNode>()->op_type().op, "transpose");
        }
    }
    EXPECT_EQ(graph->nodes().size(), 7);
}

TEST_F(FuseTransposeReshapeIntoHStack, with_single_commute)
{
    // Expand base test - add single elementwise commutable ops on path between transpose and reshape
    auto input_3 = create_input(*graph, "input_3", graphlib::Shape::create({1, 1, 64, 64}));
    graphlib::Node *transpose_0 = graph->get_node_by_name("transpose_0");
    add_node<graphlib::PyOpNode>(*graph, "add_0", "add", {}, {transpose_0, input_3}, {0, 0});

    // Run stack/slice fuse pass
    passes::fuse_reshape_transpose_pairs_into_slice_or_stack_tm(graph);

    // Reshape and transpose op should be gone, and replaced with new fused hstack op
    for (auto node : graph->nodes())
    {
        if (node->node_type() == tt::graphlib::kPyOp)
        {
            EXPECT_NE(node->as<graphlib::PyOpNode>()->op_type().op, "reshape");
            EXPECT_NE(node->as<graphlib::PyOpNode>()->op_type().op, "transpose");
        }
    }
    EXPECT_EQ(graph->nodes().size(), 9);
}

TEST_F(FuseTransposeReshapeIntoHStack, with_multiple_commute)
{
    // Expand base test - add few elementwise commutable ops on path between transpose and reshape
    auto input_3 = create_input(*graph, "input_3", graphlib::Shape::create({1, 1, 64, 64}));
    graphlib::Node *transpose_0 = graph->get_node_by_name("transpose_0");
    auto add_0 = add_node<graphlib::PyOpNode>(*graph, "add_0", "add", {}, {transpose_0, input_3}, {0, 0});
    auto add_1 = add_node<graphlib::PyOpNode>(*graph, "add_1", "add", {}, {add_0, add_0}, {0, 0});
    auto multiply_0 = add_node<graphlib::PyOpNode>(*graph, "multiply_0", "multiply", {}, {add_1, add_1}, {0, 0});
    add_node<graphlib::PyOpNode>(*graph, "subtract_0", "subtract", {}, {multiply_0, multiply_0}, {0, 0});

    // Run stack/slice fuse pass
    passes::fuse_reshape_transpose_pairs_into_slice_or_stack_tm(graph);

    // Reshape and transpose op should be gone, and replaced with new fused hstack op
    for (auto node : graph->nodes())
    {
        if (node->node_type() == tt::graphlib::kPyOp)
        {
            EXPECT_NE(node->as<graphlib::PyOpNode>()->op_type().op, "reshape");
            EXPECT_NE(node->as<graphlib::PyOpNode>()->op_type().op, "transpose");
        }
    }
    EXPECT_EQ(graph->nodes().size(), 12);
}

TEST_F(FuseTransposeReshapeIntoHStack, with_multiple_forked_commute)
{
    // Expand base test - add few elementwise commutable ops on path between transpose and reshape
    // which are forked between themselves
    auto input_3 = create_input(*graph, "input_3", graphlib::Shape::create({1, 1, 64, 64}));
    graphlib::Node *transpose_0 = graph->get_node_by_name("transpose_0");
    auto add_0 = add_node<graphlib::PyOpNode>(*graph, "add_0", "add", {}, {transpose_0, input_3}, {0, 0});
    auto add_1 = add_node<graphlib::PyOpNode>(*graph, "add_1", "add", {}, {add_0, add_0}, {0, 0});
    auto multiply_0 = add_node<graphlib::PyOpNode>(*graph, "multiply_0", "multiply", {}, {add_0, add_1}, {1, 0});
    add_node<graphlib::PyOpNode>(*graph, "subtract_0", "subtract", {}, {multiply_0, input_3}, {0, 0});

    // Run stack/slice fuse pass
    passes::fuse_reshape_transpose_pairs_into_slice_or_stack_tm(graph);

    // Reshape and transpose op should be gone, and replaced with new fused hstack op
    for (auto node : graph->nodes())
    {
        if (node->node_type() == tt::graphlib::kPyOp)
        {
            EXPECT_NE(node->as<graphlib::PyOpNode>()->op_type().op, "reshape");
            EXPECT_NE(node->as<graphlib::PyOpNode>()->op_type().op, "transpose");
        }
    }
    EXPECT_EQ(graph->nodes().size(), 12);
}

TEST_F(FuseTransposeReshapeIntoHStack, with_multiple_forked_non_commutable)
{
    // Expand base test - add few elementwise commutable and one non-commutable matmul op on path between
    // transpose and reshape
    auto input_3 = create_input(*graph, "input_3", graphlib::Shape::create({1, 1, 64, 64}));
    graphlib::Node *transpose_0 = graph->get_node_by_name("transpose_0");
    auto add_0 = add_node<graphlib::PyOpNode>(*graph, "add_0", "add", {}, {transpose_0, input_3}, {0, 0});
    auto add_1 = add_node<graphlib::PyOpNode>(*graph, "add_1", "add", {}, {add_0, add_0}, {0, 0});
    auto matmul_2 = add_node<graphlib::PyOpNode>(*graph, "matmul_2", "matmul", {}, {add_0, add_1}, {1, 0});
    add_node<graphlib::PyOpNode>(*graph, "subtract_0", "subtract", {}, {matmul_2, input_3}, {0, 0});

    // Run stack/slice fuse pass
    passes::fuse_reshape_transpose_pairs_into_slice_or_stack_tm(graph);

    // Reshape and transpose op should be still in a graph
    std::string op_type_name;
    bool reshape_in_graph = false;
    bool transpose_in_graph = false;
    for (auto node : graph->nodes())
    {
        if (node->node_type() == tt::graphlib::kPyOp)
        {
            op_type_name = node->as<graphlib::PyOpNode>()->op_type().op;
            if (op_type_name == "reshape")
                reshape_in_graph = true;
            else if (op_type_name == "transpose")
                transpose_in_graph = true;
        }
    }
    EXPECT_EQ(reshape_in_graph, true);
    EXPECT_EQ(transpose_in_graph, true);
    EXPECT_EQ(graph->nodes().size(), 13);
}

TEST_F(FuseTransposeReshapeIntoHStack, with_multiple_suitable_pairs)
{
    // Expand base test - add few elementwise commutable ops on path between transpose and reshape
    // which are forked between themselves. Also, add additional suitable pair for hstack
    auto input_3 = create_input(*graph, "input_3", graphlib::Shape::create({1, 1, 64, 64}));
    graphlib::Node *transpose_0 = graph->get_node_by_name("transpose_0");
    auto add_0 = add_node<graphlib::PyOpNode>(*graph, "add_0", "add", {}, {transpose_0, input_3}, {0, 0});
    auto add_1 = add_node<graphlib::PyOpNode>(*graph, "add_1", "add", {}, {add_0, add_0}, {0, 0});
    auto multiply_0 = add_node<graphlib::PyOpNode>(*graph, "multiply_0", "multiply", {}, {add_0, add_1}, {1, 0});
    add_node<graphlib::PyOpNode>(*graph, "subtract_0", "subtract", {}, {multiply_0, input_3}, {0, 0});

    graphlib::Node *reshape_0 = graph->get_node_by_name("reshape_0");
    auto reshape_1 = add_node<graphlib::PyOpNode>(*graph, "reshape_1", "reshape", {1, 64, 1, 64}, {reshape_0}, {0, 0});
    auto transpose_1 = add_node<graphlib::PyOpNode>(*graph, "transpose_1", graphlib::OpType("transpose", {}, {}, {{"dim0", 1}, {"dim1", 2}, {"z_dim_slice", 1}}), {reshape_1}, {0, 0});
    add_node<graphlib::PyOpNode>(*graph, "reshape_2", "reshape", {1, 1, 1, 4096}, {transpose_1}, {0, 0});

    // Run stack/slice fuse pass
    passes::fuse_reshape_transpose_pairs_into_slice_or_stack_tm(graph);

    // All transposes should be gone and replaced with new fused hstack ops together with their reshapes. Only
    // reshape will stay in the graph
    int reshape_count = 0;
    for (auto node : graph->nodes())
    {
        if (node->node_type() == tt::graphlib::kPyOp)
        {
            if (node->as<graphlib::PyOpNode>()->op_type().op == "reshape")
                reshape_count++;
            EXPECT_NE(node->as<graphlib::PyOpNode>()->op_type().op, "transpose");
        }
    }
    EXPECT_EQ(reshape_count, 1);
    EXPECT_EQ(graph->nodes().size(), 14);
}

TEST_F(FuseTransposeReshapeIntoHStack, with_multiple_suitable_pairs_both_hstack_and_hslice)
{
    // Expand base test - add few elementwise commutable ops on path between transpose and reshape
    // which are forked between themselves. Also, add additional suitable pair for hslice besides hstack
    auto input_3 = create_input(*graph, "input_3", graphlib::Shape::create({1, 1, 64, 64}));
    graphlib::Node *transpose_0 = graph->get_node_by_name("transpose_0");
    auto add_0 = add_node<graphlib::PyOpNode>(*graph, "add_0", "add", {}, {transpose_0, input_3}, {0, 0});
    auto add_1 = add_node<graphlib::PyOpNode>(*graph, "add_1", "add", {}, {add_0, add_0}, {0, 0});
    auto multiply_0 = add_node<graphlib::PyOpNode>(*graph, "multiply_0", "multiply", {}, {add_0, add_1}, {1, 0});
    add_node<graphlib::PyOpNode>(*graph, "subtract_0", "subtract", {}, {multiply_0, input_3}, {0, 0});

    graphlib::Node *reshape_0 = graph->get_node_by_name("reshape_0");
    auto reshape_1 = add_node<graphlib::PyOpNode>(*graph, "reshape_1", "reshape", {1, 1, 64, 64}, {reshape_0}, {0, 0});
    add_node<graphlib::PyOpNode>(
        *graph,
        "transpose_1",
        graphlib::OpType("transpose", {}, {}, {{"dim0", 1}, {"dim1", 2}, {"z_dim_slice", 1}}),
        {reshape_1},
        {0, 0});

    // Run stack/slice fuse pass
    passes::fuse_reshape_transpose_pairs_into_slice_or_stack_tm(graph);

    // Reshape and transpose op should be gone, and replaced with suitable hstack and hslice ops
    for (auto node : graph->nodes())
    {
        if (node->node_type() == tt::graphlib::kPyOp)
        {
            EXPECT_NE(node->as<graphlib::PyOpNode>()->op_type().op, "reshape");
            EXPECT_NE(node->as<graphlib::PyOpNode>()->op_type().op, "transpose");
        }
    }
    EXPECT_EQ(graph->nodes().size(), 13);
}

struct FuseReshapeIntoVSlice : testing::Test
{
    graphlib::Graph *graph;

    FuseReshapeIntoVSlice()
    {
        // Initialize graph
        graph = new graphlib::Graph(graphlib::IRLevel::IR_PYBUDA);

        // Graph definition
        auto input_0 = create_input(*graph, "input_0", graphlib::Shape::create({1, 32, 2048, 32}));
        auto input_1 = create_input(*graph, "input_1", graphlib::Shape::create({1, 32, 32, 64}));
        auto matmul_0 = add_node<graphlib::PyOpNode>(*graph, "matmul_0", "matmul", {}, {input_0, input_1});

        auto reshape_0 = add_node<graphlib::PyOpNode>(*graph, "reshape_0", "reshape", {1, 1024, 64, 64}, {matmul_0});

        auto input_2 = create_input(*graph, "input_2", graphlib::Shape::create({1, 1024, 64, 128}));
        auto matmul_1 = add_node<graphlib::PyOpNode>(*graph, "matmul_1", "matmul", {}, {reshape_0, input_2});

        create_output(*graph, "output_0", matmul_1);
    }

    virtual void TearDown() {
        delete graph;
    }
};

TEST_F(FuseReshapeIntoVSlice, basic_case)
{
    // Run stack/slice fuse pass
    passes::fuse_reshape_transpose_pairs_into_slice_or_stack_tm(graph);

    // Reshape op should be gone, and replaced with suitable vslice op
    for (auto node : graph->nodes())
    {
        if (node->node_type() == tt::graphlib::kPyOp)
        {
            EXPECT_NE(node->as<graphlib::PyOpNode>()->op_type().op, "reshape");
        }
    }
    EXPECT_EQ(graph->nodes().size(), 7);
}

struct FuseReshapeIntoVStack : testing::Test
{
    graphlib::Graph *graph;

    FuseReshapeIntoVStack()
    {
        // Initialize graph
        graph = new graphlib::Graph(graphlib::IRLevel::IR_PYBUDA);

        // Graph definition
        auto input_0 = create_input(*graph, "input_0", graphlib::Shape::create({1, 1024, 64, 32}));
        auto input_1 = create_input(*graph, "input_1", graphlib::Shape::create({1, 1024, 32, 64}));
        auto matmul_0 = add_node<graphlib::PyOpNode>(*graph, "matmul_0", "matmul", {}, {input_0, input_1});

        auto reshape_0 = add_node<graphlib::PyOpNode>(*graph, "reshape_0", "reshape", {1, 32, 2048, 64}, {matmul_0});

        auto input_2 = create_input(*graph, "input_2", graphlib::Shape::create({1, 32, 64, 32}));
        auto matmul_1 = add_node<graphlib::PyOpNode>(*graph, "matmul_1", "matmul", {}, {reshape_0, input_2});

        create_output(*graph, "output_0", matmul_1);
    }

    virtual void TearDown() {
        delete graph;
    }
};

TEST_F(FuseReshapeIntoVStack, basic_case)
{
    // Run stack/slice fuse pass
    passes::fuse_reshape_transpose_pairs_into_slice_or_stack_tm(graph);

    // Reshape op should be gone, and replaced with suitable vslice op
    for (auto node : graph->nodes())
    {
        if (node->node_type() == tt::graphlib::kPyOp)
        {
            EXPECT_NE(node->as<graphlib::PyOpNode>()->op_type().op, "reshape");
        }
    }
    EXPECT_EQ(graph->nodes().size(), 7);
}
