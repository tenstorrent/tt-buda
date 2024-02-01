// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "gtest/gtest.h"
#include "passes/erase_inverse_ops.hpp"
#include "passes/insert_inverse_on_io.hpp"
#include "passes/replace_incommutable_patterns.hpp"
#include "test/graph_api.hpp"

using namespace tt;

struct EraseInverseOps : testing::Test
{
    graphlib::Graph *graph;

    EraseInverseOps()
    {
        // Two transposes feeding into eltwise which has a transpose after it
        graph = new graphlib::Graph(graphlib::IRLevel::IR_PYBUDA);

        graphlib::Shape shape = graphlib::Shape::create({1, 1, 512, 160});
        graphlib::Shape shapeT = graphlib::Shape::create({1, 1, 160, 512});

        auto in0_a = create_input(*graph, "in0_a", graphlib::Shape::create({1, 1, shape[2], 256}));
        auto in0_b = create_input(*graph, "in0_b", graphlib::Shape::create({1, 1, 256, shape[3]}));
        auto matmul0 = add_node<graphlib::PyOpNode>(*graph, "matmul0", "matmul", {}, {in0_a, in0_b});
        auto transpose0 = add_node<graphlib::PyOpNode>(
            *graph,
            "transpose0",
            graphlib::OpType("transpose", {}, {}, {{"dim0", 0}, {"dim1", 1}, {"z_dim_slice", 1}}),
            {matmul0});

        auto in1_a = create_input(*graph, "in1_a", graphlib::Shape::create({1, 1, shape[2], 128}));
        auto in1_b = create_input(*graph, "in1_b", graphlib::Shape::create({1, 1, 128, shape[3]}));
        auto matmul1 = add_node<graphlib::PyOpNode>(*graph, "matmul1", "matmul", {}, {in1_a, in1_b});
        auto transpose1 = add_node<graphlib::PyOpNode>(
            *graph,
            "transpose1",
            graphlib::OpType("transpose", {}, {}, {{"dim0", 0}, {"dim1", 1}, {"z_dim_slice", 1}}),
            {matmul1});

        auto add = add_node<graphlib::PyOpNode>(*graph, "add", "add", {}, {transpose0, transpose1});
        auto post_transpose = add_node<graphlib::PyOpNode>(
            *graph,
            "post_transpose",
            graphlib::OpType("transpose", {}, {}, {{"dim0", 0}, {"dim1", 1}, {"z_dim_slice", 1}}),
            {add});

        create_output(*graph, "out0", post_transpose);
    }
};

TEST_F(EraseInverseOps, erase_transpose)
{
    passes::erase_inverse_ops(graph);

    // Transposes should be gone
    for (auto node : graph->nodes())
    {
        if (node->node_type() == tt::graphlib::kPyOp)
        {
            EXPECT_NE(node->as<graphlib::PyOpNode>()->op_type().op, "transpose");
        }
    }
    EXPECT_EQ(graph->nodes().size(), 8);
}

// Two transposes feeding into eltwise which has a fork to another tranpose but also a fork-join buffer
TEST_F(EraseInverseOps, erase_transpose_fork)
{

    // fork after add into a transpose and a buffer
    graphlib::Node *add = graph->get_node_by_name("add");
    auto buffer = add_node<graphlib::PyOpNode>(*graph, "buffer", "nop", {}, {add});

    create_output(*graph, "out1", buffer);

    bool attempt_update = true;
    while (attempt_update)
    {
        attempt_update = passes::erase_inverse_ops(graph);
        if (not attempt_update) {
           attempt_update = passes::insert_inverse_on_outputs(graph);
           if (not attempt_update) {
                attempt_update = passes::insert_inverse_on_inputs(graph);
           }
        }
    }

    // Because intermediate value is read out, we have to transpose the output on the out0
    int transpose_count = 0;
    for (auto node : graph->nodes())
    {
        if (node->node_type() == tt::graphlib::kPyOp)
        {
            if (node->as<graphlib::PyOpNode>()->op_type().op == "transpose")
                transpose_count++;
        }
    }
    EXPECT_EQ(transpose_count, 1);

    EXPECT_EQ(graph->nodes().size(), 11);
}

// Two transposes feeding into eltwise which has a fork to another tranpose but also a fork-join buffer. Eventually they are joined.
TEST_F(EraseInverseOps, erase_inverse_ops_transpose_fork_join)
{
    // fork after add into a transpose and a buffer
    graphlib::Node *add = graph->get_node_by_name("add");
    auto buffer1 = add_node<graphlib::PyOpNode>(*graph, "buffer1", "nop", {}, {add});
    auto buffer2 = add_node<graphlib::PyOpNode>(*graph, "buffer2", "nop", {}, {buffer1});

    auto post_transpose = graph->get_node_by_name("post_transpose");
    auto unary = add_node<graphlib::PyOpNode>(*graph, "unary", "exp", {}, {post_transpose});
    auto unary_transpose = add_node<graphlib::PyOpNode>(
        *graph,
        "unary_transpose",
        graphlib::OpType("transpose", {}, {}, {{"dim0", 0}, {"dim1", 1}, {"z_dim_slice", 1}}),
        {unary});
    auto join = add_node<graphlib::PyOpNode>(*graph, "join", "add", {}, {unary_transpose, buffer2});

    graph->remove_node(graph->get_node_by_name("out0"));
    create_output(*graph, "out0", join);

    bool attempt_update = true;
    while (attempt_update)
    {
        attempt_update = passes::erase_inverse_ops(graph);
        if (not attempt_update) {
           attempt_update = passes::insert_inverse_on_outputs(graph);
           if (not attempt_update) {
                attempt_update = passes::insert_inverse_on_inputs(graph);
           }
        }
    }

    // Because intermediate value is read out, we have to transpose the output on the out0
    int transpose_count = 0;
    for (auto node : graph->nodes())
    {
        if (node->node_type() == tt::graphlib::kPyOp)
        {
            if (node->as<graphlib::PyOpNode>()->op_type().op == "transpose")
                transpose_count++;
        }
    }
    EXPECT_EQ(transpose_count, 1);

    EXPECT_EQ(graph->nodes().size(), 13);
}


TEST_F(EraseInverseOps, erase_inverse_ops_dual_reduce)
{
    // fork after add into a transpose and a buffer
    // graphlib::Node *add = graph->get_node_by_name("add");
    // auto buffer1 = add_node<graphlib::PyOpNode>(*graph, "buffer1", "nop", {}, {add});
    // auto buffer2 = add_node<graphlib::PyOpNode>(*graph, "buffer2", "nop", {}, {buffer1});

    auto post_transpose = graph->get_node_by_name("post_transpose");
    auto smx_1 = add_node<graphlib::PyOpNode>(*graph, "smx_1", "softmax", {-1, false}, {post_transpose});
    auto reshape_1 = add_node<graphlib::PyOpNode>(*graph, "reshape_1", "reshape", {1, 512, 10, 16}, {smx_1});
    auto reduce_1 = add_node<graphlib::PyOpNode>(*graph, "reduce_1", "reduce_sum", {-2}, {reshape_1});
    auto reduce_2 = add_node<graphlib::PyOpNode>(*graph, "reduce_2", "reduce_sum", {-1}, {reduce_1});
    auto reshape_2 = add_node<graphlib::PyOpNode>(*graph, "reshape_2", "reshape", {1, 1, 512, 1}, {reduce_2});
    auto smx_2 = add_node<graphlib::PyOpNode>(*graph, "smx_2", "softmax", {-1, false}, {reshape_2});
    graph->remove_node(graph->get_node_by_name("out0"));
    create_output(*graph, "out0", smx_2);

    bool attempt_update = true;
    while (attempt_update)
    {
        attempt_update = passes::erase_inverse_ops(graph);
        if (not attempt_update) {
           attempt_update = passes::insert_inverse_on_outputs(graph);
           if (not attempt_update) {
                attempt_update = passes::insert_inverse_on_inputs(graph);
           }
        }
    }

    // Because intermediate value is read out, we have to transpose the output on the out0
    int transpose_count = 0;
    int reduce_count = 0;
    int reshape_count = 0;
    for (auto node : graph->nodes())
    {
        if (node->node_type() == tt::graphlib::kPyOp)
        {
            if (node->as<graphlib::PyOpNode>()->op_type().op == "transpose")
                transpose_count++;
            else if (node->as<graphlib::PyOpNode>()->op_type().op == "reduce_sum")
                reduce_count++;
            else if (node->as<graphlib::PyOpNode>()->op_type().op == "reshape")
                reshape_count++;
        }
    }
    EXPECT_EQ(transpose_count, 0);
    EXPECT_EQ(reduce_count, 1);
    EXPECT_EQ(reshape_count, 0);

    // EXPECT_EQ(graph->nodes().size(), 13);
}

TEST_F(EraseInverseOps, replace_x_y_change_concat_pattern)
{
    auto post_transpose = graph->get_node_by_name("post_transpose");
    auto reshape_0 = add_node<graphlib::PyOpNode>(*graph, "reshape_0", "reshape", {256, 320}, {post_transpose});
    auto reshape_1 = add_node<graphlib::PyOpNode>(*graph, "reshape_1", "reshape", {256, 320}, {post_transpose});
    auto reshape_2 = add_node<graphlib::PyOpNode>(*graph, "reshape_2", "reshape", {256, 320}, {post_transpose});
    auto concat = add_node<graphlib::PyOpNode>(*graph, "concat", "concatenate", {-2}, {reshape_0, reshape_1, reshape_2});
    
    graph->remove_node(graph->get_node_by_name("out0"));
    create_output(*graph, "out0", concat);

    passes::replace_incommutable_patterns(graph); // Should insert inverses under each reshape and a single clone after the concat
    passes::erase_inverse_ops(graph); // Should erase all inverses, leaving just the clone after he concat

    int reshape_count = 0;
    for (auto node : graph->nodes())
    {
        if (node->node_type() == tt::graphlib::kPyOp)
        {
            if (node->as<graphlib::PyOpNode>()->op_type().op == "reshape")
                reshape_count++;
        }
    }

    EXPECT_EQ(reshape_count, 1);
}
