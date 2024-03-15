// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "passes/pre_placer_buda_passes.hpp"
#include "test/common.hpp"


namespace tt::test
{
// Check that outcgoing edge contains an opnode of "type"
static bool find_outgoing_op(std::vector<graphlib::Node*> const& users, std::string const& type)
{
    for (auto* node : users)
    {
        if ((node->node_type() == graphlib::kBudaOp) && (node->as<graphlib::OpNode>()->op_type().op == type))
            return true;
    }
    return false;
}

struct TestGradEltwiseAdd : public BudaGraphTest
{
   protected:
    virtual std::vector<OpType*> create_graph() override
    {
        auto dim0 = 512;
        auto dim1 = 256;
        auto inputA = create_activation(shape(1, 1, dim0, dim1));  // 512x256
        auto inputB = create_activation(shape(1, 1, dim0, dim1));  // 512x256
        auto eltwise_op = create_op("add", {inputA, inputB});
        eltwise_op->set_gradient_op(true);
        op_name = eltwise_op->name();
        return {eltwise_op};
    }

    std::string op_name;
};

struct TestGradEltwiseSubtract : public BudaGraphTest
{
   protected:
    virtual std::vector<OpType*> create_graph() override
    {
        auto dim0 = 512;
        auto dim1 = 256;
        auto inputA = create_activation(shape(1, 1, dim0, dim1));  // 512x256
        auto inputB = create_activation(shape(1, 1, dim0, dim1));  // 512x256
        auto eltwise_op = create_op("subtract", {inputA, inputB});
        eltwise_op->set_gradient_op(true);
        op_name = eltwise_op->name();
        return {eltwise_op};
    }

    std::string op_name;
};

TEST_F(TestGradEltwiseAdd, split_unsup_grad_wh_b0)
{
    DeviceConfig device_config = create_device_config(ARCH::WORMHOLE_B0);
    graphlib::Graph* graph = get_graph();

    split_unsupported_gradient_ops(graph, device_config);

    // Check NOP doesn't exist in output outgoing edge
    std::vector<graphlib::Node*> users = graph->users(graph->get_node_by_name(op_name));
    EXPECT_EQ(find_outgoing_op(users, "nop"), false);
}

TEST_F(TestGradEltwiseAdd, split_unsup_grad_non_wh_b0)
{
    DeviceConfig device_config = create_device_config(ARCH::WORMHOLE);
    graphlib::Graph* graph = get_graph();

    split_unsupported_gradient_ops(graph, device_config);

    // Check NOP added to output outgoing edge
    std::vector<graphlib::Node*> users = graph->users(graph->get_node_by_name(op_name));
    EXPECT_EQ(find_outgoing_op(users, "nop"), true);
}

TEST_F(TestGradEltwiseSubtract, split_unsup_grad_wh_b0)
{
    DeviceConfig device_config = create_device_config(ARCH::WORMHOLE_B0);
    graphlib::Graph* graph = get_graph();

    split_unsupported_gradient_ops(graph, device_config);

    // Check NOP doesn't exist in output outgoing edge
    std::vector<graphlib::Node*> users = graph->users(graph->get_node_by_name(op_name));
    EXPECT_EQ(find_outgoing_op(users, "nop"), false);
}

TEST_F(TestGradEltwiseSubtract, split_unsup_grad_non_wh_b0)
{
    DeviceConfig device_config = create_device_config(ARCH::WORMHOLE);
    graphlib::Graph* graph = get_graph();

    split_unsupported_gradient_ops(graph, device_config);

    // Check NOP added to output outgoing edge
    std::vector<graphlib::Node*> users = graph->users(graph->get_node_by_name(op_name));
    EXPECT_EQ(find_outgoing_op(users, "nop"), true);
}

}  // namespace tt::test
