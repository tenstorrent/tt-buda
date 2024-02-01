// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "passes/pre_placer_buda_passes.hpp"
#include "test/common.hpp"

namespace tt::test
{

struct TilizeGraph : public BudaGraphTest
{
   protected:
    virtual std::vector<OpType *> create_graph() override
    {
        std::uint32_t seq_len = 128;
        std::uint32_t embed = 384;

        auto act = create_input("act", shape(1, 1, seq_len, embed), tt::graphlib::InputNodeType::Activation);
        auto gelu = create_op("gelu", {act});

        gelu_name = gelu->name();
        return {gelu};
    }
    std::string gelu_name;
};

TEST_F(TilizeGraph, tilize)
{
    // This test verifies insertion of tilize op after activation node
    Node *node;
    std::vector<Node *> user;
    graphlib::Graph *graph = get_graph();

    node = graph->get_node_by_name("act");
    user = graph->data_users(node);
    graphlib::OpNode* op_node = dynamic_cast<graphlib::OpNode*>(user[0]);
    ASSERT_FALSE(op_node->is_tilize());

    // insert tililze op
    tt::insert_tilize_op_on_input(graph);

    node = graph->get_node_by_name("act");
    user = graph->data_users(node);

    op_node = dynamic_cast<graphlib::OpNode*>(user[0]);
    ASSERT_TRUE(op_node->is_tilize());
    user = graph->users(op_node);

    ASSERT_EQ(user[0]->name(), gelu_name);
}
}  // namespace tt::test
