// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "gtest/gtest.h"
#include "passes/fuse_pad_conv2d.hpp"
#include "test/graph_api.hpp"

using namespace tt;

struct FusePadConv2d : testing::Test
{
    graphlib::Graph *graph;

    FusePadConv2d()
    {
        // Two transposes feeding into eltwise which has a transpose after it
        graph = new graphlib::Graph(graphlib::IRLevel::IR_PYBUDA);

        auto in0_a = create_input(*graph, "in0_a", graphlib::Shape::create({1, 3, 513, 513}));
        auto param0 = create_input(*graph, "param1", graphlib::Shape::create({32, 3, 3, 3}), graphlib::InputNodeType::Parameter);
        auto pad = add_node<graphlib::PyOpNode>(*graph, "pad2", "pad", {1,1,1,1,0,0}, {in0_a});
        auto conv2d = add_node<graphlib::PyOpNode>(*graph, "conv2d3", "conv2d", {2,2,1,1,1,1,1,1,0,0,0,0,0}, {pad, param0});

        create_output(*graph, "out0", conv2d);
    }
};

TEST_F(FusePadConv2d, fuse_pad_conv2d)
{
    passes::fuse_pad_conv2d(graph);

    // Transposes should be gone
    for (auto node : graph->nodes())
    {
        if (node->node_type() == tt::graphlib::kPyOp)
        {
            EXPECT_NE(node->as<graphlib::PyOpNode>()->op_type().op, "pad");
        }
    }
    EXPECT_EQ(graph->nodes().size(), 4);
}
