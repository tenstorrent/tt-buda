// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "test/common.hpp"
#include "passes/move_index_to_mm_weights.hpp"

namespace tt::test
{

struct Gpt2Split : testing::Test
{
    graphlib::Graph *graph;

    Gpt2Split()
    {
        // Initialize graph
        graph = new graphlib::Graph(graphlib::IRLevel::IR_PYBUDA);

        // Graph definition
        auto weight = create_input(*graph, "weight", graphlib::Shape::create({768, 2304}), graphlib::InputNodeType::Parameter); 
        auto bias = create_input(*graph, "bias", graphlib::Shape::create({2304}), graphlib::InputNodeType::Parameter);
        auto input = create_input(*graph, "input", graphlib::Shape::create({1, 32, 768}));

        auto reshape_0 = add_node<graphlib::PyOpNode>(*graph, "reshape_0", "reshape", {32, 768}, {input});
        auto matmul_0 = add_node<graphlib::PyOpNode>(*graph, "matmul_0", "matmul", {}, {reshape_0, weight}); 
        auto add_0 = add_node<graphlib::PyOpNode>(*graph, "add_0", "add", {}, {matmul_0, bias});
        auto reshape_1 = add_node<graphlib::PyOpNode>(*graph, "reshape_1", "reshape", {1, 32, 2304}, {add_0});
        auto index_1 = add_node<graphlib::PyOpNode>(*graph, "index_1", "index", {-1, 0, 768, 1}, {reshape_1});
        auto reshape_1_1 = add_node<graphlib::PyOpNode>(*graph, "reshape_1_1", "reshape", {1, 32, 12, 64}, {index_1});
        auto transpose_1_1 = add_node<graphlib::PyOpNode>(*graph, "transpose_1_1", graphlib::OpType("transpose", {}, {}, {{"dim0", -3}, {"dim1", -2}, {"z_dim_slice", -1}}), {reshape_1_1});
        create_output(*graph, "output_1_1", transpose_1_1);

        auto index_2 = add_node<graphlib::PyOpNode>(*graph, "index_2", "index", {-1, 768, 1536, 1}, {reshape_1});
        auto reshape_1_2 = add_node<graphlib::PyOpNode>(*graph, "reshape_1_2", "reshape", {1, 32, 12, 64}, {index_2});
        auto transpose_1_2 = add_node<graphlib::PyOpNode>(*graph, "transpose_1_2", graphlib::OpType("transpose", {}, {}, {{"dim0", -3}, {"dim1", -2}, {"z_dim_slice", -1}}), {reshape_1_2});
        create_output(*graph, "output_1_2", transpose_1_2);

        auto index_3 = add_node<graphlib::PyOpNode>(*graph, "index_3", "index", {-1, 1536, 2304, 1}, {reshape_1});
        auto reshape_1_3 = add_node<graphlib::PyOpNode>(*graph, "reshape_1_3", "reshape", {1, 32, 12, 64}, {index_3});
        auto transpose_1_3 = add_node<graphlib::PyOpNode>(*graph, "transpose_1_3", graphlib::OpType("transpose", {}, {}, {{"dim0", -3}, {"dim1", -2}, {"z_dim_slice", -1}}), {reshape_1_3});
        create_output(*graph, "output_1_3", transpose_1_3);
    }
};

TEST_F(Gpt2Split, gpt2_split)
{
    tt::passes::move_index_to_mm_weights(graph);

    // check
    std::vector<graphlib::Node*> inputs, outputs;
    for (auto n : graph->nodes())
    {
        if (n->node_type() == graphlib::NodeType::kInput)
            inputs.push_back(n);
        if (n->node_type() == graphlib::NodeType::kOutput)
            outputs.push_back(n);

        // make sure index ops are removed
        if (n->node_type() != graphlib::NodeType::kPyOp)
            continue;
        EXPECT_EQ(n->as<graphlib::OpNode>()->op_type().op != "index", true);
    }

    // input/output sanity check
    EXPECT_EQ(outputs.size(), 3);
    EXPECT_EQ(inputs.size(), 7); 
    graphlib::Shape weight_expected = graphlib::Shape::create({768, 768});
    graphlib::Shape bias_expected = graphlib::Shape::create({32, 768});
    for (auto n : inputs)
    {
        if (n->name().find("weight") != std::string::npos)
        {
            EXPECT_EQ(n->shape(), weight_expected);
        }
        else if (n->name().find("bias") != std::string::npos)
        {
            EXPECT_EQ(n->shape(), bias_expected);
        }
    }
}

struct Fuyu8bSplit : testing::Test
{
    graphlib::Graph *graph;

    Fuyu8bSplit()
    {
        // Initialize graph
        graph = new graphlib::Graph(graphlib::IRLevel::IR_PYBUDA);

        // Graph definition
        auto weight = create_input(*graph, "weight", graphlib::Shape::create({12288, 4096}), graphlib::InputNodeType::Parameter); 
        auto bias = create_input(*graph, "bias", graphlib::Shape::create({12288}), graphlib::InputNodeType::Parameter);
        auto in_1 = create_input(*graph, "in_1", graphlib::Shape::create({1, 32, 4096}), graphlib::InputNodeType::Activation);
        auto in_2 = create_input(*graph, "in_2", graphlib::Shape::create({1, 416, 4096}), graphlib::InputNodeType::Activation);

        // path 1
        auto transpose_1 = add_node<graphlib::PyOpNode>(*graph, "transpose_1", graphlib::OpType("transpose", {}, {}, {{"dim0", -2}, {"dim1", -1}, {"z_dim_slice", -1}}), {weight});
        auto reshape_1_0 = add_node<graphlib::PyOpNode>(*graph, "reshape_1_0", "reshape", {32, 4096}, {in_1});
        auto matmul_1 = add_node<graphlib::PyOpNode>(*graph, "matmul_1", "matmul", {}, {reshape_1_0, transpose_1});
        auto reshape_1_1 = add_node<graphlib::PyOpNode>(*graph, "reshape_1_1", "reshape", {1, 32, 12288}, {matmul_1});
        auto add_1 = add_node<graphlib::PyOpNode>(*graph, "add_1", "add", {}, {reshape_1_1, bias});
        auto reshape_1_2 = add_node<graphlib::PyOpNode>(*graph, "reshape_1_2", "reshape", {1, 32, 64, 3, 64}, {add_1});
        auto index_1_1 = add_node<graphlib::PyOpNode>(*graph, "index_1_1", "index", {-2, 0, 1, 1}, {reshape_1_2});
        auto reshape_1_3 = add_node<graphlib::PyOpNode>(*graph, "reshape_1_3", "reshape", {1, 32, 64, 64}, {index_1_1}); 
        create_output(*graph, "output_1_1", reshape_1_3);
        auto index_1_2 = add_node<graphlib::PyOpNode>(*graph, "index_1_2", "index", {-2, 1, 2, 1}, {reshape_1_2});
        auto reshape_1_4 = add_node<graphlib::PyOpNode>(*graph, "reshape_1_4", "reshape", {1, 32, 64, 64}, {index_1_2}); 
        create_output(*graph, "output_1_2", reshape_1_4);
        auto index_1_3 = add_node<graphlib::PyOpNode>(*graph, "index_1_3", "index", {-2, 2, 3, 1}, {reshape_1_2});
        auto reshape_1_5 = add_node<graphlib::PyOpNode>(*graph, "reshape_1_5", "reshape", {1, 32, 64, 64}, {index_1_3}); 
        create_output(*graph, "output_1_3", reshape_1_5);

        // path 2
        auto transpose_2 = add_node<graphlib::PyOpNode>(*graph, "transpose_2", graphlib::OpType("transpose", {}, {}, {{"dim0", -2}, {"dim1", -1}, {"z_dim_slice", -1}}), {weight});
        auto reshape_2_0 = add_node<graphlib::PyOpNode>(*graph, "reshape_2_0", "reshape", {416, 4096}, {in_2});
        auto matmul_2 = add_node<graphlib::PyOpNode>(*graph, "matmul_2", "matmul", {}, {reshape_2_0, transpose_2});
        auto reshape_2_1 = add_node<graphlib::PyOpNode>(*graph, "reshape_2_1", "reshape", {1, 32, 12288}, {matmul_2});
        auto add_2 = add_node<graphlib::PyOpNode>(*graph, "add_2", "add", {}, {reshape_2_1, bias});
        auto reshape_2_2 = add_node<graphlib::PyOpNode>(*graph, "reshape_2_2", "reshape", {1, 32, 64, 3, 64}, {add_2});
        auto index_2_1 = add_node<graphlib::PyOpNode>(*graph, "index_2_1", "index", {-2, 0, 1, 1}, {reshape_2_2});
        auto reshape_2_3 = add_node<graphlib::PyOpNode>(*graph, "reshape_2_3", "reshape", {1, 32, 64, 64}, {index_2_1}); 
        create_output(*graph, "output_2_1", reshape_2_3);
        auto index_2_2 = add_node<graphlib::PyOpNode>(*graph, "index_2_2", "index", {-2, 1, 2, 1}, {reshape_2_2});
        auto reshape_2_4 = add_node<graphlib::PyOpNode>(*graph, "reshape_2_4", "reshape", {1, 32, 64, 64}, {index_2_2}); 
        create_output(*graph, "output_2_2", reshape_2_4);
        auto index_2_3 = add_node<graphlib::PyOpNode>(*graph, "index_2_3", "index", {-2, 2, 3, 1}, {reshape_2_2});
        auto reshape_2_5 = add_node<graphlib::PyOpNode>(*graph, "reshape_2_5", "reshape", {1, 32, 64, 64}, {index_2_3}); 
        create_output(*graph, "output_2_3", reshape_2_5);
    }
};

TEST_F(Fuyu8bSplit, fuyu8b_split)
{
    tt::passes::move_index_to_mm_weights(graph);

    // check
    std::vector<graphlib::Node*> inputs, outputs;
    graphlib::Shape reshape_expected = graphlib::Shape::create({1, 32, 4096});
    for (auto n : graph->nodes())
    {
        if (n->node_type() == graphlib::NodeType::kInput)
            inputs.push_back(n);
        if (n->node_type() == graphlib::NodeType::kOutput)
            outputs.push_back(n);

        // make sure index ops are removed
        if (n->node_type() != graphlib::NodeType::kPyOp)
            continue;
        EXPECT_EQ(n->as<graphlib::OpNode>()->op_type().op != "index", true);

        // check if shape of reshape in the middle is updated
        if (n->as<graphlib::OpNode>()->op_type().op == "reshape" and n->shape().size() == 3)
        {
            EXPECT_EQ(n->shape(), reshape_expected);
        }
    }


    // input/output sanity check
    EXPECT_EQ(outputs.size(), 6);
    EXPECT_EQ(inputs.size(), 11);
    graphlib::Shape weight_expected = graphlib::Shape::create({4096, 4096});
    graphlib::Shape bias_expected = graphlib::Shape::create({32, 4096});
    for (auto n : inputs)
    {
        if (n->name().find("weight") != std::string::npos)
        {
            EXPECT_EQ(n->shape(), weight_expected);
            EXPECT_EQ(graph->data_users(n).size(), 2);
        }
        else if (n->name().find("bias") != std::string::npos)
        {
            EXPECT_EQ(n->shape(), bias_expected);
            EXPECT_EQ(graph->data_users(n).size(), 1);
        }
    }
}

}  // namespace tt::test
