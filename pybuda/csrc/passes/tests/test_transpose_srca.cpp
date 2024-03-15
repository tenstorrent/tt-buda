// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "passes/pre_placer_buda_passes.hpp"
#include "test/common.hpp"

namespace tt::test
{

static int count_nop_node(std::vector<graphlib::Node*> const& operands)
{
    int count = 0;
    for (auto* node : operands)
    {
        if ((node->node_type() == graphlib::kBudaOp) && (node->as<graphlib::OpNode>()->op_type().op == "nop"))
            count++;
    }
    return count;
}

struct TestTransposeSrcAUnary : public GraphTest<graphlib::IRLevel::IR_BUDA>
{
   protected:
    virtual std::vector<OpType*> create_graph() override
    {
        auto inputA = create_activation(shape(1, 1, 512, 256));
        auto eltwise_op = create_op("exp", {inputA});
        append_tm("transpose", eltwise_op, 0, {{"dim0", 2}, {"dim1", 3}, {"z_dim_slice", -1}});
        op_name = eltwise_op->name();
        return {eltwise_op};
    }

    std::string op_name;
};

struct TestTransposeSrcABinary : public GraphTest<graphlib::IRLevel::IR_BUDA>
{
   protected:
    virtual std::vector<OpType*> create_graph() override
    {
        auto dim0 = 512;
        auto dim1 = 256;
        auto inputA = create_activation(shape(1, 1, dim0, dim1));  // 512x256
        auto inputB = create_activation(shape(1, 1, dim1, dim0));  // 256x512
        auto eltwise_op = create_op("add", {inputA, inputB});
        append_tm("transpose", eltwise_op, 0, {{"dim0", 2}, {"dim1", 3}, {"z_dim_slice", -1}});
        op_name = eltwise_op->name();
        return {eltwise_op};
    }

    std::string op_name;
};


struct TestTransposeSrcABinaryBoth : public GraphTest<graphlib::IRLevel::IR_BUDA>
{
   protected:
    virtual std::vector<OpType*> create_graph() override
    {
        auto dim0 = 512;
        auto dim1 = 256;
        auto inputA = create_activation(shape(1, 1, dim0, dim1));  // 512x256
        auto inputB = create_activation(shape(1, 1, dim0, dim1));  // 512x256
        auto eltwise_op = create_op("add", {inputA, inputB});

        // Tranpose both operand0 and operand1
        append_tm("transpose", eltwise_op, 0, {{"dim0", 2}, {"dim1", 3}, {"z_dim_slice", -1}});
        append_tm("transpose", eltwise_op, 1, {{"dim0", 2}, {"dim1", 3}, {"z_dim_slice", -1}});
        op_name = eltwise_op->name();
        return {eltwise_op};
    }

    std::string op_name;
};


// Currently wh_b0 optimization unsupported for nary ops
struct TestTransposeSrcANary : public GraphTest<graphlib::IRLevel::IR_BUDA>
{
   protected:
    virtual std::vector<OpType*> create_graph() override
    {
        auto dim0 = 512;
        auto dim1 = 256;
        auto inputA = create_activation(shape(1, 1, dim0, dim1));  // 512x256
        auto inputB = create_activation(shape(1, 1, dim1, dim0));  // 256x512
        auto inputC = create_activation(shape(1, 1, dim1, dim0));  // 256x512
        auto eltwise_op = create_op("index_copy", {inputA, inputB, inputC});

        auto operand_indx_to_transpose = 0;
        append_tm("transpose", eltwise_op, operand_indx_to_transpose, {{"dim0", 2}, {"dim1", 3}, {"z_dim_slice", -1}});
        op_name = eltwise_op->name();
        return {eltwise_op};
    }

    std::string op_name;
};

TEST_F(TestTransposeSrcAUnary, fix_transpose_wh_b0)
{
    DeviceConfig device_config = create_device_config(ARCH::WORMHOLE_B0);
    graphlib::Graph* graph = get_graph();

    fix_transposes(graph, device_config);

    // Check NOP DNE in graph srcA opnode
    std::vector<graphlib::Node*> operands = graph->operands(graph->get_node_by_name(op_name));
    EXPECT_EQ(count_nop_node(operands), 0);
}

TEST_F(TestTransposeSrcAUnary, fix_transpose_non_wh_b0)
{
    DeviceConfig device_config = create_device_config(ARCH::WORMHOLE);
    graphlib::Graph* graph = get_graph();

    fix_transposes(graph, device_config);

    // Check if NOP was added to all operands
    std::vector<graphlib::Node*> operands = graph->operands(graph->get_node_by_name(op_name));  // out0
    EXPECT_EQ(count_nop_node(operands), operands.size());
}

TEST_F(TestTransposeSrcABinary, fix_transpose_wh_b0)
{
    DeviceConfig device_config = create_device_config(ARCH::WORMHOLE_B0);
    graphlib::Graph* graph = get_graph();

    fix_transposes(graph, device_config);

    // Check NOP DNE in graph srcA opnode
    std::vector<graphlib::Node*> operands = graph->operands(graph->get_node_by_name(op_name));
    EXPECT_EQ(count_nop_node(operands), 0);
}

TEST_F(TestTransposeSrcABinary, fix_transpose_non_wh_b0)
{
    DeviceConfig device_config = create_device_config(ARCH::WORMHOLE);
    graphlib::Graph* graph = get_graph();

    fix_transposes(graph, device_config);

    // Check if NOP was added to operandA
    std::vector<graphlib::Node*> operands = graph->operands(graph->get_node_by_name(op_name));  // out0
    EXPECT_EQ(count_nop_node(operands), 1);
}

TEST_F(TestTransposeSrcABinaryBoth, fix_transpose_wh_b0)
{
    DeviceConfig device_config = create_device_config(ARCH::WORMHOLE_B0);
    graphlib::Graph* graph = get_graph();

    fix_transposes(graph, device_config);

    // Check NOP DNE in graph srcA opnode
    std::vector<graphlib::Node*> operands = graph->operands(graph->get_node_by_name(op_name));
    EXPECT_EQ(count_nop_node(operands), operands.size() - 1);
}

TEST_F(TestTransposeSrcABinaryBoth, fix_transpose_non_wh_b0)
{
    DeviceConfig device_config = create_device_config(ARCH::WORMHOLE);
    graphlib::Graph* graph = get_graph();

    fix_transposes(graph, device_config);

    // Check if NOP was added to all operands
    std::vector<graphlib::Node*> operands = graph->operands(graph->get_node_by_name(op_name));  // out0
    EXPECT_EQ(count_nop_node(operands), operands.size()); 
}

TEST_F(TestTransposeSrcANary, fix_transpose_wh_b0)
{
    DeviceConfig device_config = create_device_config(ARCH::WORMHOLE_B0);
    graphlib::Graph* graph = get_graph();

    fix_transposes(graph, device_config);

    // Check if NOP was added to operants
    std::vector<graphlib::Node*> operands = graph->operands(graph->get_node_by_name(op_name));  // out0
    EXPECT_EQ(count_nop_node(operands), 1); // TODO: Change to 0 when wh_b0 supports N-Nary
}

TEST_F(TestTransposeSrcANary, fix_transpose_non_wh_b0)
{
    DeviceConfig device_config = create_device_config(ARCH::WORMHOLE);
    graphlib::Graph* graph = get_graph();

    fix_transposes(graph, device_config);

    // Check if NOP was added to operants
    std::vector<graphlib::Node*> operands = graph->operands(graph->get_node_by_name(op_name));  // out0
    EXPECT_EQ(count_nop_node(operands), 1);
}

}  // namespace tt::test
