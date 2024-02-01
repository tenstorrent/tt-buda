// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include <stdlib.h>
#include "test/common.hpp"
#include "passes/link_past_cache_ios.hpp"

namespace tt::test
{

// test past-cache pass of Whisper models (V1)
struct WhisperPastCacheBase : testing::Test
{
    graphlib::Graph *graph;

    WhisperPastCacheBase()
    {
        // Initialize graph
        graph = new graphlib::Graph(graphlib::IRLevel::IR_PYBUDA);

        // Graph definition
        // define input/weight nodes
        auto weight = create_input(*graph, "weight", graphlib::Shape::create({768, 768}), graphlib::InputNodeType::Parameter); 
        auto pkv_input_1 = create_input(*graph, "pkv_input_1", graphlib::Shape::create({1, 12, 416, 64}), tt::graphlib::InputNodeType::Activation);
        auto pkv_input_2 = create_input(*graph, "pkv_input_2", graphlib::Shape::create({1, 12, 416, 64}), tt::graphlib::InputNodeType::Activation);
        auto in_1 = create_input(*graph, "_input_1", graphlib::Shape::create({32, 768}), tt::graphlib::InputNodeType::Activation);
        auto in_2 = create_input(*graph, "_input_2", graphlib::Shape::create({32, 768}), tt::graphlib::InputNodeType::Activation);

        // define other nodes
        auto transpose_1 = add_node<graphlib::PyOpNode>(*graph, "transpose_1", graphlib::OpType("transpose", {}, {}, {{"dim0", -2}, {"dim1", -1}, {"z_dim_slice", -1}}), {weight});
        auto matmul_1 = add_node<graphlib::PyOpNode>(*graph, "matmul_1", "matmul", {}, {in_1, transpose_1});
        auto hslice_1 = add_node<graphlib::PyOpNode>(*graph, "hslice_1", "hslice", {12}, {matmul_1});
        auto concat_1 = add_node<graphlib::PyOpNode>(*graph, "concat_1", "concatenate", {-2}, {pkv_input_1, hslice_1});
        auto output_1 = create_output(*graph, "output_1", concat_1);

        auto transpose_2 = add_node<graphlib::PyOpNode>(*graph, "transpose_2", graphlib::OpType("transpose", {}, {}, {{"dim0", -2}, {"dim1", -1}, {"z_dim_slice", -1}}), {weight});
        auto matmul_2 = add_node<graphlib::PyOpNode>(*graph, "matmul_2", "matmul", {}, {in_2, transpose_2});
        auto hslice_2 = add_node<graphlib::PyOpNode>(*graph, "hslice_2", "hslice", {12}, {matmul_2});
        auto concat_2 = add_node<graphlib::PyOpNode>(*graph, "concat_2", "concatenate", {-2}, {pkv_input_2, hslice_2});
        auto output_2 = create_output(*graph, "output_2", concat_2);

        // add tags to concat nodes
        graphlib::TagValue span(std::string("OP1"));
        concat_1->as<graphlib::TaggedNode>()->hints["layer"] = span;
        concat_2->as<graphlib::TaggedNode>()->hints["layer"] = span;

        // add output/input nodes to module-output/module-input
        graph->add_module_output(output_1->id());
        graph->add_module_output(output_2->id());
        graph->add_module_input(weight->id());
        graph->add_module_input(pkv_input_1->id());
        graph->add_module_input(pkv_input_2->id());
        graph->add_module_input(in_1->id());
        graph->add_module_input(in_2->id());
    }
};

TEST_F(WhisperPastCacheBase, whisper_past_cache_base)
{
    tt::passes::link_past_cache_ios(graph);

    // get input/output nodes
    std::vector<graphlib::Node*> output_nodes, input_nodes;
    for (auto n : graph->nodes())
    {
        if (n->node_type() == graphlib::NodeType::kOutput)
            output_nodes.push_back(n);
        if (n->node_type() == graphlib::NodeType::kInput)
            input_nodes.push_back(n);
    }

    // check if output nodes are linked together with param
    EXPECT_EQ(output_nodes.size(), 2);
    EXPECT_EQ(graph->user_edges(output_nodes[0]).size(), 1);
    EXPECT_EQ(graph->user_edges(output_nodes[1]).size(), 1);
    EXPECT_EQ(graph->node_by_id(graph->user_edges(output_nodes[0])[0].consumer_node_id), graph->node_by_id(graph->user_edges(output_nodes[1])[0].consumer_node_id));

    // check if original 'pkv_input_' nodes are removed 
    for (auto input_node : input_nodes)
        EXPECT_EQ(input_node->name().find("pkv_input_"), std::string::npos);    

    // check if cached param op is connected to hslices
    auto cache = graph->node_by_id(graph->user_edges(output_nodes[0])[0].consumer_node_id);
    auto cache_users = graph->user_edges(cache);
    EXPECT_EQ(cache_users.size(), 2);
    for (auto user : cache_users)
        EXPECT_EQ(graph->node_by_id(user.consumer_node_id)->as<graphlib::OpNode>()->op_type().op, "hslice");
}

// test past-cache pass of Whisper models (V2)
struct WhisperPastCacheSubGraph : testing::Test
{
    graphlib::Graph *graph;

    WhisperPastCacheSubGraph()
    {
        // Initialize graph
        graph = new graphlib::Graph(graphlib::IRLevel::IR_PYBUDA);

        // Graph definition
        // define input/weight nodes
        auto weight = create_input(*graph, "weight", graphlib::Shape::create({384, 384}), graphlib::InputNodeType::Parameter, 0); 
        auto pkv_input_1 = create_input(*graph, "pkv_input_1", graphlib::Shape::create({1, 1536, 384}), tt::graphlib::InputNodeType::Activation, 0);
        auto pkv_input_2 = create_input(*graph, "pkv_input_2", graphlib::Shape::create({1, 6, 1536, 64}), tt::graphlib::InputNodeType::Activation, 1);

        // subgraph 1
        auto reshape_1 = add_node<graphlib::PyOpNode>(*graph, "reshape_1", "reshape", {1536, 384}, {pkv_input_1}, {}, {}, {}, 0);
        auto transpose_1 = add_node<graphlib::PyOpNode>(*graph, "transpose_1", graphlib::OpType("transpose", {}, {}, {{"dim0", -2}, {"dim1", -1}, {"z_dim_slice", -1}}), {weight}, {}, 0);
        auto matmul_1 = add_node<graphlib::PyOpNode>(*graph, "matmul_1", "matmul", {}, {reshape_1, transpose_1}, {}, {}, {}, 0);
        auto hslice_1 = add_node<graphlib::PyOpNode>(*graph, "hslice_1", "hslice", {6}, {matmul_1}, {}, {}, {}, 0);
        auto output_1_1 = create_output(*graph, "output_1_1", hslice_1, 0);
        auto reshape_1_ = add_node<graphlib::PyOpNode>(*graph, "reshape_1_", "reshape", {6, 1536, 64}, {hslice_1}, {}, {}, {}, 0);
        auto output_1_2 = create_output(*graph, "output_1_2", reshape_1_, 0);

        // subgraph 2
        auto nop_2 = add_node<graphlib::PyOpNode>(*graph, "nop_2", "nop", {}, {pkv_input_2}, {}, {}, {}, 1);
        auto output_2_1 = create_output(*graph, "output_2_1", nop_2, 1);
        auto reshape_2 = add_node<graphlib::PyOpNode>(*graph, "reshape_2", "reshape", {12, 1536, 64}, {nop_2}, {}, {}, {}, 1);
        auto output_2_2 = create_output(*graph, "output_2_2", reshape_2, 1); 

        // add tags to concat nodes
        graphlib::TagValue span(std::string("OP1"));
        hslice_1->as<graphlib::TaggedNode>()->hints["layer"] = span;
        reshape_1->as<graphlib::TaggedNode>()->hints["layer"] = span;

        // add output/input nodes to module-output/module-input
        graph->add_module_output(output_1_1->id());
        graph->add_module_output(output_1_2->id());
        graph->add_module_output(output_2_1->id());
        graph->add_module_output(output_2_2->id());
        graph->add_module_input(weight->id());
        graph->add_module_input(pkv_input_1->id());
        graph->add_module_input(pkv_input_2->id());
    }
};

TEST_F(WhisperPastCacheSubGraph, whisper_past_cache_subgraph)
{
    tt::passes::link_past_cache_ios(graph);

    // get input/output nodes
    std::vector<graphlib::Node*> output_nodes, input_nodes;
    for (auto n : graph->nodes())
    {
        if (n->node_type() == graphlib::NodeType::kOutput)
            output_nodes.push_back(n);
        if (n->node_type() == graphlib::NodeType::kInput)
            input_nodes.push_back(n);
    }

    // check if output nodes are linked together with param
    EXPECT_EQ(output_nodes.size(), 3);
    EXPECT_EQ(graph->user_edges(output_nodes[0]).size(), 1);
    EXPECT_EQ(graph->user_edges(output_nodes[1]).size(), 1);
    EXPECT_EQ(graph->user_edges(output_nodes[2]).size(), 0);

    // check if original 'pkv_input_' nodes are removed 
    for (auto input_node : input_nodes)
        EXPECT_EQ(input_node->name().find("pkv_input_2"), std::string::npos);    

    // check if cached param op is connected to hslices
    auto cache = graph->node_by_id(graph->user_edges(output_nodes[0])[0].consumer_node_id);
    auto cache_users = graph->user_edges(cache);
    EXPECT_EQ(cache_users.size(), 1); 
    EXPECT_EQ(graph->node_by_id(cache_users[0].consumer_node_id)->as<graphlib::OpNode>()->op_type().op, "nop");
}

// test past-cache with rotate of T5 models
struct T5PastCacheRotate : testing::Test
{
    graphlib::Graph *graph;

    T5PastCacheRotate()
    {
        // Initialize graph
        graph = new graphlib::Graph(graphlib::IRLevel::IR_PYBUDA);

        // Graph definition
        // define input/weight nodes
        auto weight = create_input(*graph, "weight", graphlib::Shape::create({512, 512}), graphlib::InputNodeType::Parameter); 
        auto in_1 = create_input(*graph, "in_1", graphlib::Shape::create({32, 512}), tt::graphlib::InputNodeType::Constant);
        auto in_2 = create_input(*graph, "in_2", graphlib::Shape::create({32, 512}), tt::graphlib::InputNodeType::Constant);
        auto pkv_input_1 = create_input(*graph, "pkv_input_1", graphlib::Shape::create({1, 8, 480, 64}), tt::graphlib::InputNodeType::Activation);
        auto pkv_input_2 = create_input(*graph, "pkv_input_2", graphlib::Shape::create({1, 8, 480, 64}), tt::graphlib::InputNodeType::Activation);

        // path 1
        auto transpose_1 = add_node<graphlib::PyOpNode>(*graph, "transpose_1", graphlib::OpType("transpose", {}, {}, {{"dim0", -2}, {"dim1", -1}, {"z_dim_slice", -1}}), {weight});
        auto matmul_1 = add_node<graphlib::PyOpNode>(*graph, "matmul_1", "matmul", {}, {in_1, transpose_1});
        auto hslice_1 = add_node<graphlib::PyOpNode>(*graph, "hslice_1", "hslice", {8}, {matmul_1});
        auto concat_1 = add_node<graphlib::PyOpNode>(*graph, "concat_1", "concatenate", {-2}, {pkv_input_1, hslice_1});
        auto output_1 = create_output(*graph, "output_1", concat_1);

        // path 2 
        auto transpose_2 = add_node<graphlib::PyOpNode>(*graph, "transpose_2", graphlib::OpType("transpose", {}, {}, {{"dim0", -2}, {"dim1", -1}, {"z_dim_slice", -1}}), {weight});
        auto matmul_2 = add_node<graphlib::PyOpNode>(*graph, "matmul_2", "matmul", {}, {in_2, transpose_2});
        auto hslice_2 = add_node<graphlib::PyOpNode>(*graph, "hslice_2", "hslice", {8}, {matmul_2});
        auto concat_2 = add_node<graphlib::PyOpNode>(*graph, "concat_2", "concatenate", {-2}, {pkv_input_2, hslice_2});
        auto output_2 = create_output(*graph, "output_2", concat_2);

        // add tags to concat nodes
        graphlib::TagValue span(std::string("OP1"));
        concat_1->as<graphlib::TaggedNode>()->hints["layer"] = span;
        concat_2->as<graphlib::TaggedNode>()->hints["layer"] = span;

        // add output/input nodes to module-output/module-input
        graph->add_module_output(output_1->id());
        graph->add_module_output(output_2->id());
        graph->add_module_input(weight->id());
        graph->add_module_input(in_1->id());
        graph->add_module_input(in_2->id());
        graph->add_module_input(pkv_input_1->id());
        graph->add_module_input(pkv_input_2->id());
    }
};

bool single_user_operand(graphlib::Graph *graph, graphlib::Node *n)
{
    if (graph->user_edges(n).size() != 1)
        return false;
    if (graph->operand_edges(n).size() != 1)
        return false;
    return true;
}

TEST_F(T5PastCacheRotate, t5_past_cache_rotate)
{
    setenv("PYBUDA_ROTATE_PAST_CACHE_PARAMS", "1", 1 /* overwrite */);
    tt::passes::link_past_cache_ios(graph);
    unsetenv("PYBUDA_ROTATE_PAST_CACHE_PARAMS");

    // get input/output nodes
    std::vector<graphlib::Node*> output_nodes, input_nodes;
    for (auto n : graph->nodes())
    {
        if (n->node_type() == graphlib::NodeType::kOutput)
            output_nodes.push_back(n);
        if (n->node_type() == graphlib::NodeType::kInput)
            input_nodes.push_back(n);
    }

    // check if output nodes are linked together with param
    EXPECT_EQ(output_nodes.size(), 3);
    EXPECT_EQ(graph->user_edges(output_nodes[0]).size(), 1);
    EXPECT_EQ(graph->user_edges(output_nodes[1]).size(), 1);
    EXPECT_EQ(graph->user_edges(output_nodes[2]).size(), 1);
    EXPECT_EQ(graph->node_by_id(graph->user_edges(output_nodes[0])[0].consumer_node_id), graph->node_by_id(graph->user_edges(output_nodes[1])[0].consumer_node_id));
    EXPECT_EQ(graph->node_by_id(graph->user_edges(output_nodes[0])[0].consumer_node_id), graph->node_by_id(graph->user_edges(output_nodes[2])[0].consumer_node_id));

    // check if original 'pkv_input_' nodes are removed 
    for (auto input_node : input_nodes)
        EXPECT_EQ(input_node->name().find("pkv_input"), std::string::npos);    

    // check if cached param op is connected to hslices
    auto cache = graph->node_by_id(graph->user_edges(output_nodes[0])[0].consumer_node_id);
    auto cache_users = graph->user_edges(cache);
    EXPECT_EQ(cache_users.size(), 4);
    EXPECT_EQ(graph->node_by_id(cache_users[0].consumer_node_id)->as<graphlib::OpNode>()->op_type().op, "hslice");
    EXPECT_EQ(graph->node_by_id(cache_users[1].consumer_node_id)->as<graphlib::OpNode>()->op_type().op, "hslice");

    // check the node connection in rotated parts
    auto left_sel = graph->node_by_id(cache_users[2].consumer_node_id);
    auto right_sel = graph->node_by_id(cache_users[3].consumer_node_id);
    EXPECT_EQ(left_sel->as<graphlib::OpNode>()->op_type().op, "index");
    EXPECT_EQ(right_sel->as<graphlib::OpNode>()->op_type().op, "index");
    EXPECT_EQ(single_user_operand(graph, left_sel), true);
    EXPECT_EQ(single_user_operand(graph, right_sel), true);
    auto rotate_concat_op = graph->data_users(left_sel)[0]; 
    EXPECT_EQ(rotate_concat_op->as<graphlib::OpNode>()->op_type().op, "concatenate");
    EXPECT_EQ(single_user_operand(graph, graph->data_users(rotate_concat_op)[0]), true);
    EXPECT_EQ(graph->data_users(graph->data_users(rotate_concat_op)[0])[0], output_nodes[2]); 
}

// test past-cache pass of Falcon40b
struct Falcon40bPastCache : testing::Test
{
    graphlib::Graph *graph;

    Falcon40bPastCache()
    {
        // Initialize graph
        graph = new graphlib::Graph(graphlib::IRLevel::IR_PYBUDA);

        // Graph definition
        // define input/weight nodes
        auto weight = create_input(*graph, "weight", graphlib::Shape::create({64, 8192}), graphlib::InputNodeType::Parameter);
        auto in_1_1 = create_input(*graph, "in_1_1", graphlib::Shape::create({128, 8192}), tt::graphlib::InputNodeType::Constant);
        auto in_1_2 = create_input(*graph, "in_1_2", graphlib::Shape::create({1, 1, 128, 64}), tt::graphlib::InputNodeType::Constant);
        auto in_2_1 = create_input(*graph, "in_2_1", graphlib::Shape::create({32, 8192}), tt::graphlib::InputNodeType::Constant);
        auto in_2_2 = create_input(*graph, "in_2_2", graphlib::Shape::create({1, 1, 32, 64}), tt::graphlib::InputNodeType::Constant);
        auto in_2_3 = create_input(*graph, "in_2_3", graphlib::Shape::create({1, 32, 128, 64}), tt::graphlib::InputNodeType::Constant);
        auto in_2_4 = create_input(*graph, "in_2_4", graphlib::Shape::create({1, 32, 128, 64}), tt::graphlib::InputNodeType::Constant);
        auto pkv_input = create_input(*graph, "pkv_input", graphlib::Shape::create({1, 32, 128, 64}), tt::graphlib::InputNodeType::Activation);
        auto cos_0 = create_input(*graph, "cos_0", graphlib::Shape::create({1, 1, 128, 64}), tt::graphlib::InputNodeType::Activation);
        auto cos_1 = create_input(*graph, "cos_1", graphlib::Shape::create({1, 1, 32, 64}), tt::graphlib::InputNodeType::Activation);

        // path 1
        auto transpose_1 = add_node<graphlib::PyOpNode>(*graph, "transpose_1", graphlib::OpType("transpose", {}, {}, {{"dim0", -2}, {"dim1", -1}, {"z_dim_slice", -1}}), {weight});
        auto matmul_1 = add_node<graphlib::PyOpNode>(*graph, "matmul_1", "matmul", {}, {in_1_1, transpose_1});
        auto hslice_1 = add_node<graphlib::PyOpNode>(*graph, "hslice_1", "hslice", {1}, {matmul_1});
        auto multiply_1 = add_node<graphlib::PyOpNode>(*graph, "multiply_1", "multiply", {}, {hslice_1, cos_0});
        auto add_1 = add_node<graphlib::PyOpNode>(*graph, "add_1", "add", {}, {multiply_1, in_1_2});
        auto broadcast_1 = add_node<graphlib::PyOpNode>(*graph, "broadcast_1", "broadcast", {-3, 32, 1}, {add_1});
        auto output_1 = create_output(*graph, "output_1", broadcast_1);

        // path 2
        auto transpose_2 = add_node<graphlib::PyOpNode>(*graph, "transpose_2", graphlib::OpType("transpose", {}, {}, {{"dim0", -2}, {"dim1", -1}, {"z_dim_slice", -1}}), {weight});
        auto matmul_2 = add_node<graphlib::PyOpNode>(*graph, "matmul_2", "matmul", {}, {in_2_1, transpose_2});
        auto reshape_2 = add_node<graphlib::PyOpNode>(*graph, "reshape_2", "reshape", {1, 1, 32, 64}, {matmul_2});
        auto multiply_2 = add_node<graphlib::PyOpNode>(*graph, "multiply_2", "multiply", {}, {reshape_2, cos_1});
        auto add_2 = add_node<graphlib::PyOpNode>(*graph, "add_2", "add", {}, {multiply_2, in_2_2});
        auto transpose_3 = add_node<graphlib::PyOpNode>(*graph, "transpose_3", graphlib::OpType("transpose", {}, {}, {{"dim0", -3}, {"dim1", -2}, {"z_dim_slice", 32}}), {add_2});
        auto multiply_3 = add_node<graphlib::PyOpNode>(*graph, "multiply_3", "multiply", {}, {transpose_3, in_2_3});
        auto multiply_4 = add_node<graphlib::PyOpNode>(*graph, "multiply_4", "multiply", {}, {pkv_input, in_2_4});
        auto add_3 = add_node<graphlib::PyOpNode>(*graph, "add_3", "add", {}, {multiply_3, multiply_4});
        auto reshape_3 = add_node<graphlib::PyOpNode>(*graph, "reshape_3", "reshape", {32, 128, 64}, {pkv_input});
        auto output_2_1 = create_output(*graph, "output_2_1", add_3);
        auto output_2_2 = create_output(*graph, "output_2_2", reshape_3);

        // add tags
        graphlib::TagValue span(std::string("OP1"));
        add_3->as<graphlib::TaggedNode>()->hints["layer"] = span;
        multiply_4->as<graphlib::TaggedNode>()->hints["layer"] = span;

        // add output/input nodes to module-output/module-input
        graph->add_module_output(output_1->id());
        graph->add_module_output(output_2_1->id());
        graph->add_module_output(output_2_2->id());
        graph->add_module_input(weight->id());
        graph->add_module_input(pkv_input->id());
        graph->add_module_input(in_1_1->id());
        graph->add_module_input(in_1_2->id());
        graph->add_module_input(in_2_1->id());
        graph->add_module_input(in_2_2->id());
        graph->add_module_input(in_2_3->id());
        graph->add_module_input(in_2_4->id());
        graph->add_module_input(cos_0->id());
        graph->add_module_input(cos_1->id());
    }
};

TEST_F(Falcon40bPastCache, falcon40b_past_cache)
{
    tt::passes::link_past_cache_ios(graph);

    // get input/output nodes
    std::vector<graphlib::Node*> output_nodes, input_nodes;
    for (auto n : graph->nodes())
    {
        if (n->node_type() == graphlib::NodeType::kOutput)
            output_nodes.push_back(n);
        if (n->node_type() == graphlib::NodeType::kInput)
            input_nodes.push_back(n);
    }

    // check if output nodes are linked together with param
    EXPECT_EQ(output_nodes.size(), 3);
    EXPECT_EQ(graph->user_edges(output_nodes[0]).size(), 1);
    EXPECT_EQ(graph->user_edges(output_nodes[1]).size(), 1);
    EXPECT_EQ(graph->user_edges(output_nodes[2]).size(), 0);

    // check if original 'pkv_input_' nodes are removed
    for (auto input_node : input_nodes)
        EXPECT_EQ(input_node->name().find("pkv_input"), std::string::npos);

    // check if cached param op is connected to hslices
    auto cache = graph->node_by_id(graph->user_edges(output_nodes[0])[0].consumer_node_id);
    auto cache_users = graph->user_edges(cache);
    EXPECT_EQ(cache_users.size(), 2);
    EXPECT_EQ(graph->node_by_id(cache_users[0].consumer_node_id)->as<graphlib::OpNode>()->op_type().op, "multiply");
    EXPECT_EQ(graph->node_by_id(cache_users[1].consumer_node_id)->as<graphlib::OpNode>()->op_type().op, "reshape");
}

// test past-cache pass of Fuyu-8b
struct Fuyu8bPastCache : testing::Test
{
    graphlib::Graph *graph;

    Fuyu8bPastCache()
    {
        // Initialize graph
        graph = new graphlib::Graph(graphlib::IRLevel::IR_PYBUDA);

        // Graph definition
        // define input/weight nodes
        auto weight = create_input(*graph, "weight", graphlib::Shape::create({4096, 4096}), graphlib::InputNodeType::Parameter); 
        auto in_1 = create_input(*graph, "in_1", graphlib::Shape::create({32, 4096}), tt::graphlib::InputNodeType::Constant);
        auto in_2 = create_input(*graph, "in_2", graphlib::Shape::create({416, 4096}), tt::graphlib::InputNodeType::Constant);
        auto in_add = create_input(*graph, "in_add", graphlib::Shape::create({1, 64, 416, 32}), tt::graphlib::InputNodeType::Constant);
        auto pkv_input = create_input(*graph, "pkv_input", graphlib::Shape::create({1, 64, 416, 64}), tt::graphlib::InputNodeType::Activation);

        // path 1
        auto matmul_1 = add_node<graphlib::PyOpNode>(*graph, "matmul_1", "matmul", {}, {in_1, weight});        
        auto reshape_1 = add_node<graphlib::PyOpNode>(*graph, "reshape_1", "reshape", {1, 32, 64, 64}, {matmul_1});
        auto transpose_1 = add_node<graphlib::PyOpNode>(*graph, "transpose_1", graphlib::OpType("transpose", {}, {}, {{"dim0", -3}, {"dim1", -2}, {"z_dim_slice", 64}}), {reshape_1});
        auto concat_1 = add_node<graphlib::PyOpNode>(*graph, "concat_1", "concatenate", {-2}, {pkv_input, transpose_1});
        auto output_1 = create_output(*graph, "output_1", concat_1);

        // path 2
        auto matmul_2 = add_node<graphlib::PyOpNode>(*graph, "matmul_2", "matmul", {}, {in_2, weight});
        auto reshape_2 = add_node<graphlib::PyOpNode>(*graph, "reshape_2", "reshape", {1, 416, 64, 64}, {matmul_2});
        auto transpose_2 = add_node<graphlib::PyOpNode>(*graph, "transpose_2", graphlib::OpType("transpose", {}, {}, {{"dim0", -3}, {"dim1", -2}, {"z_dim_slice", 64}}), {reshape_2});
        auto index_2_1 = add_node<graphlib::PyOpNode>(*graph, "index_2_1", "index", {-1, 0, 32, 64}, {transpose_2});
        auto index_2_2 = add_node<graphlib::PyOpNode>(*graph, "index_2_2", "index", {-1, 32, 64, 64}, {transpose_2});
        auto add_2 = add_node<graphlib::PyOpNode>(*graph, "add_2", "add", {}, {in_add, index_2_2});
        auto concat_2 = add_node<graphlib::PyOpNode>(*graph, "concat_2", "concatenate", {-1}, {index_2_1, add_2});
        auto output_2 = create_output(*graph, "output_2", concat_2);

        // add tags to concat nodes
        graphlib::TagValue span(std::string("OP1"));
        concat_1->as<graphlib::TaggedNode>()->hints["layer"] = span;

        // add output/input nodes to module-output/module-input
        graph->add_module_output(output_1->id());
        graph->add_module_output(output_2->id());
        graph->add_module_input(weight->id());
        graph->add_module_input(in_1->id());
        graph->add_module_input(in_2->id());
        graph->add_module_input(in_add->id());
        graph->add_module_input(pkv_input->id());
    }
};

TEST_F(Fuyu8bPastCache, fuyu8b_past_cache)
{
    tt::passes::link_past_cache_ios(graph);

    // get input/output nodes
    std::vector<graphlib::Node*> output_nodes, input_nodes;
    for (auto n : graph->nodes())
    {
        if (n->node_type() == graphlib::NodeType::kOutput)
            output_nodes.push_back(n);
        if (n->node_type() == graphlib::NodeType::kInput)
            input_nodes.push_back(n);
    }

    // check if output nodes are linked together with param
    EXPECT_EQ(output_nodes.size(), 2);
    EXPECT_EQ(graph->user_edges(output_nodes[0]).size(), 1);
    EXPECT_EQ(graph->user_edges(output_nodes[1]).size(), 1);
    EXPECT_EQ(graph->node_by_id(graph->user_edges(output_nodes[0])[0].consumer_node_id), graph->node_by_id(graph->user_edges(output_nodes[1])[0].consumer_node_id));

    // check if original 'pkv_input_' nodes are removed 
    for (auto input_node : input_nodes)
        EXPECT_EQ(input_node->name().find("pkv_input"), std::string::npos);    

    // check if cached param op is connected to hslices
    auto cache = graph->node_by_id(graph->user_edges(output_nodes[0])[0].consumer_node_id);
    auto cache_users = graph->user_edges(cache);
    EXPECT_EQ(cache_users.size(), 1);
    EXPECT_EQ(graph->node_by_id(cache_users[0].consumer_node_id)->as<graphlib::OpNode>()->op_type().op, "hslice");
}

}  // namespace tt::test
