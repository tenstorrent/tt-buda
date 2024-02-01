// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include <stdlib.h>

#include "gtest/gtest.h"
#include "passes/move_select_after_matmul_optional.hpp"
#include "test/graph_api.hpp"

using namespace tt;

struct MoveSelectAfterMatmulOptional : testing::Test
{
    graphlib::Graph *graph;

    MoveSelectAfterMatmulOptional()
    { 
        graph = new graphlib::Graph(graphlib::IRLevel::IR_PYBUDA);

        // define input/param/const  
        auto act = create_input(*graph, "act", graphlib::Shape::create({1, 1, 6144, 21632})); 
        auto weight = create_input(*graph, "weight", graphlib::Shape::create({1, 1, 21632, 64}));
        auto in_0 = create_input(*graph, "in_0", graphlib::Shape::create({1, 1, 64, 80})); 

        // define ops
        auto matmul_0 = add_node<graphlib::PyOpNode>(*graph, "matmul_0", "matmul", {}, {act, weight}); 
        auto vslice_0 = add_node<graphlib::PyOpNode>(*graph, "vslice_1", "vslice", {192,}, {matmul_0});
        auto select_0 = add_node<graphlib::PyOpNode>(*graph, "select_0", "select", {-3, 0, 167, 192}, {vslice_0});
        auto vstack_0 = add_node<graphlib::PyOpNode>(*graph, "vstack_0", "vstack", {167,}, {select_0});
        auto matmul_1 = add_node<graphlib::PyOpNode>(*graph, "matmul_1", "matmul", {}, {vstack_0, in_0});
 
        create_output(*graph, "out0", matmul_1);

    }
};

TEST_F(MoveSelectAfterMatmulOptional, test_move_select_after_matmul_1)
{  
    setenv("PYBUDA_MANUAL_SPLICE_DECOMP_TH", "158", 1 /* overwrite */); 
    
    passes::move_select_after_matmul_optional(graph);

    // Splice op decomposed from select op should be moved after matmul op
    size_t idx = 0; 
    std::vector<std::string> expected_op_seq = {"matmul", "matmul", "vslice", "select", "vstack"};
    for (auto *node : graphlib::topological_sort(*graph))
    { 
        if (node->node_type() == tt::graphlib::kPyOp)
        {
            if (idx < expected_op_seq.size())
            {
               EXPECT_EQ(node->as<graphlib::PyOpNode>()->op_type().op, expected_op_seq[idx]);
                idx++;
            } 
        }
    }
    EXPECT_EQ(graph->nodes().size(), 9);
 
    unsetenv("PYBUDA_MANUAL_SPLICE_DECOMP_TH");
}

TEST_F(MoveSelectAfterMatmulOptional, test_move_select_after_matmul_2)
{  
    setenv("PYBUDA_MANUAL_SPLICE_DECOMP_TH", "158", 1 /* overwrite */); 

    // Expand the base-test case
    auto weight2 = create_input(*graph, "weight2", graphlib::Shape::create({1, 1, 64, 64}));
    graphlib::Node *matmul_0 = graph->get_node_by_name("matmul_0");
    auto matmul_2 = add_node<graphlib::PyOpNode>(*graph, "matmul_2", "matmul", {}, {matmul_0, weight2});
    insert_node_on_edge(graph, graph->user_data_edges(matmul_0)[0], matmul_2);
    
    passes::move_select_after_matmul_optional(graph);

    // Splice op should be in the same order 
    size_t idx = 0; 
    std::vector<std::string> expected_op_seq = {"matmul", "matmul", "vslice", "select", "vstack", "matmul"};
    for (auto *node : graphlib::topological_sort(*graph))
    { 
        if (node->node_type() == tt::graphlib::kPyOp)
        {
            if (idx < expected_op_seq.size())
            {
               EXPECT_EQ(node->as<graphlib::PyOpNode>()->op_type().op, expected_op_seq[idx]);
                idx++;
            } 
        }
    }
    EXPECT_EQ(graph->nodes().size(), 11);
 
    unsetenv("PYBUDA_MANUAL_SPLICE_DECOMP_TH");
}

TEST_F(MoveSelectAfterMatmulOptional, test_move_select_after_matmul_3)
{  
    setenv("PYBUDA_MANUAL_SPLICE_DECOMP_TH", "158", 1 /* overwrite */); 

    // Expand the base-test case 
    graphlib::Node *select_0 = graph->get_node_by_name("select_0"); 
    graphlib::OpNode *select_op_node =  dynamic_cast<graphlib::OpNode *>(select_0); 
    std::vector<graphlib::OpType::Attr> new_select_attr = {1, 0, 64, 192};
    select_op_node->overwrite_op_attrs(new_select_attr);    
    graphlib::calculate_and_set_node_shape(graph, select_0);

    graphlib::Node *vstack_0 = graph->get_node_by_name("vstack_0");
    graphlib::OpNode *vstack_op_node =  dynamic_cast<graphlib::OpNode *>(vstack_0); 
    std::vector<graphlib::OpType::Attr> new_vstack_attr = {64,};
    vstack_op_node->overwrite_op_attrs(new_vstack_attr); 
    graphlib::calculate_and_set_node_shape(graph, vstack_0);  
    graphlib::calculate_and_set_node_shape(graph, graph->get_node_by_name("matmul_1"));  
    graphlib::calculate_and_set_node_shape(graph, graph->get_node_by_name("out0"));  
 
    passes::move_select_after_matmul_optional(graph);

    // Splice op should be in the same order 
    size_t idx = 0; 
    std::vector<std::string> expected_op_seq = {"matmul", "vslice", "select", "vstack", "matmul"};
    for (auto *node : graphlib::topological_sort(*graph))
    { 
        if (node->node_type() == tt::graphlib::kPyOp)
        {
            if (idx < expected_op_seq.size())
            {
               EXPECT_EQ(node->as<graphlib::PyOpNode>()->op_type().op, expected_op_seq[idx]);
                idx++;
            } 
        }
    }
    EXPECT_EQ(graph->nodes().size(), 9);
 
    unsetenv("PYBUDA_MANUAL_SPLICE_DECOMP_TH");
}

