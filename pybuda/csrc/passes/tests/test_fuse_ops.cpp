// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "balancer/balancer.hpp"
#include "balancer/legalizer/graph_solver.hpp"
#include "balancer/legalizer/legalizer.hpp"
#include "balancer/tests/test_balancer_utils.hpp"
#include "graph_lib/defines.hpp"
#include "graph_lib/utils.hpp"
#include "gtest/gtest.h"
#include "lower_to_buda/fused_op.hpp"
#include "lower_to_buda/netlist.hpp"
#include "passes/dataformat.hpp"
#include "passes/fuse_ops.hpp"
#include "test/common.hpp"

using namespace tt;
namespace tt::test
{

// Fuse ops with default arguments.
void fuse_ops(graphlib::Graph* graph, DeviceConfig* device_config = nullptr)
{
    const std::vector<std::vector<std::string>> op_names_to_chip_break;
    const std::vector<std::vector<std::string>> op_names_to_epoch_break;

    if (device_config != nullptr)
    {
        tt::fuse_ops(graph, *device_config, op_names_to_chip_break, op_names_to_epoch_break, {}, {}, {});
    }
    else
    {
        tt::fuse_ops(
            graph, tt::test::create_device_config(), op_names_to_chip_break, op_names_to_epoch_break, {}, {}, {});
    }
}

// Get vector of all fused ops in the graph.
std::vector<BudaOpNode*> get_fused_ops(Graph* graph)
{
    std::vector<BudaOpNode*> fused_ops;

    for (auto node : graph->nodes())
    {
        if (node->node_type() == graphlib::kBudaOp)
        {
            BudaOpNode* op = node->as<BudaOpNode>();

            if (op->is_fused_op())
            {
                fused_ops.push_back(op);
            }
        }
    }

    return fused_ops;
}

struct FuseBroadcastCLHSMatmul : public BudaGraphTest
{
   protected:
    virtual std::vector<OpType*> create_graph() override
    {
        auto in0 = create_activation(1, 1, 32, 32);
        auto in1 = create_activation(1, 1, 32, 32);
        auto in2 = create_activation(1, 1, 32, 64);

        auto add0 = create_op("add", {in0, in1});
        auto add1 = create_op("add", {add0, in2});
        append_tm("broadcast", add1, 0, 3, 2);

        auto out = create_op("matmul", {add1, in1});

        return {out};
    }
};

TEST_F(FuseBroadcastCLHSMatmul, fuse_broadcast_c_as_lhs_matmul)
{
    graphlib::Graph* graph = get_graph();
    fuse_ops(graph);

    // Get fused ops.
    std::vector<BudaOpNode*> fused_ops = get_fused_ops(graph);

    // We expect only 1 fused op.
    ASSERT_EQ(fused_ops.size(), 1);

    graphlib::UBlockOrder u_block_order = get_output_ublock_order(graph, fused_ops[0]);

    // We expect UBlockOrder::R since fused op has brodcast C.
    ASSERT_EQ(u_block_order, graphlib::UBlockOrder::R);
}

struct FuseOpsEquivalentTest : public BudaGraphTest
{
   protected:
    virtual std::vector<OpType*> create_graph() override
    {
        auto in0 = create_activation(1, 1, 64, 64);
        auto in1 = create_activation(1, 1, 64, 64);
        auto in2 = create_activation(1, 1, 64, 64);

        auto add0 = create_op("add", {in0, in1});
        auto add1 = create_op("add", {add0, in2});

        auto add2 = create_op("add", {in0, in1});
        auto add3 = create_op("add", {add2, in2});
        add3_name = add3->name();

        auto out = create_op("matmul", {add1, add3});

        return {out};
    }

    std::string add3_name;
};

TEST_F(FuseOpsEquivalentTest, fuse_equivalent_fused_ops_without_attr)
{
    graphlib::Graph* graph = get_graph();
    fuse_ops(graph);

    // Get fused ops.
    std::vector<BudaOpNode*> fused_ops = get_fused_ops(graph);

    ASSERT_EQ(fused_ops.size(), 2);

    // Get test op_model in order to generate buda fused ops.
    balancer::BalancerConfig balancer_config = create_balancer_config();
    std::shared_ptr<balancer::BalancerCacheCollection> cache_collection = create_balancer_cache_collection();
    balancer::LegalOpModels valid_op_models =
        balancer::legalizer::get_legal_op_models(graph, balancer_config, cache_collection);
    std::vector<BudaFusedOp> buda_fused_ops;

    for (auto fused_op : fused_ops)
    {
        buda_fused_ops.push_back(create_fused_op(fused_op, valid_op_models[fused_op][0]));
    }

    ASSERT_EQ(buda_fused_ops.size(), 2);

    // Equivalent call should return true for 2 fused ops in this test.
    ASSERT_TRUE(buda_fused_ops[0].equivalent(buda_fused_ops[1]));
}

TEST_F(FuseOpsEquivalentTest, fuse_equivalent_fused_ops_with_attr)
{
    Graph* graph = get_graph();

    // Add atribute to one op.
    // This attribute should make diff between fused ops so that they are not equivalent anymore.
    BudaOpAttrs relu_attr;
    relu_attr["relu_en"] = true;
    auto add3 = graph->get_node_by_name(this->add3_name);
    BudaOpNode* add3_op = add3->as<BudaOpNode>();
    add3_op->overwrite_buda_attrs(relu_attr);

    fuse_ops(graph);

    // Get fused ops.
    std::vector<BudaOpNode*> fused_ops = get_fused_ops(graph);

    ASSERT_EQ(fused_ops.size(), 2);

    // Get test op_model in order to generate buda fused ops.
    balancer::BalancerConfig balancer_config = create_balancer_config();
    std::shared_ptr<balancer::BalancerCacheCollection> cache_collection = create_balancer_cache_collection();
    balancer::LegalOpModels valid_op_models =
        balancer::legalizer::get_legal_op_models(graph, balancer_config, cache_collection);
    std::vector<BudaFusedOp> buda_fused_ops;

    for (auto fused_op : fused_ops)
    {
        buda_fused_ops.push_back(create_fused_op(fused_op, valid_op_models[fused_op][0]));
    }

    ASSERT_EQ(buda_fused_ops.size(), 2);

    // Equivalent call should return false for 2 fused ops in this test.
    ASSERT_FALSE(buda_fused_ops[0].equivalent(buda_fused_ops[1]));
}

struct FuseOpsReuseTest : public BudaGraphTest
{
   protected:
    virtual std::vector<OpType*> create_graph() override
    {
        auto in0 = create_activation(1, 1, 32, 32);
        auto in1 = create_activation(1, 1, 32, 32);
        auto in2 = create_activation(1, 1, 32, 32);

        auto add0 = create_op("add", {in0, in1});
        auto add1 = create_op("add", {in2, add0});

        append_tm("tile_broadcast", add1, 1 /* operand id */, 2 /* broadcast dim */);
        op_tile_broadcast_name = add0->name();

        auto add2 = create_op("add", {in0, in1});
        auto add3 = create_op("add", {add2, in2});

        add2_name = add2->name();

        auto out = create_op("matmul", {add1, add3});

        return {out};
    }

    std::string add2_name;
    std::string op_tile_broadcast_name;
};

TEST_F(FuseOpsReuseTest, fuse_dont_reuse_dest_if_relu)
{
    Graph* graph = get_graph();

    // Add relu atribute to one op.
    // Its outputs should not be reused.
    BudaOpAttrs relu_attr;
    relu_attr["relu_en"] = true;
    auto add2 = graph->get_node_by_name(add2_name);
    BudaOpNode* add2_op = add2->as<BudaOpNode>();
    add2_op->overwrite_buda_attrs(relu_attr);

    fuse_ops(graph);

    // Get fused ops.
    std::vector<BudaOpNode*> fused_ops = get_fused_ops(graph);

    // Find the operator with relu activation (add2) and confirm that its outputs are not reused.
    for (auto fused_op : fused_ops)
    {
        auto fused = fused_op->get_fused_op();
        for (auto schedule : fused->get_schedules())
        {
            for (auto& op : schedule.ops)
            {
                if (op.name == add2_name)
                {
                    EXPECT_TRUE(op.output_type != FusedSubOp::OutputType::DEST);
                }
            }
        }
    }
}

TEST_F(FuseOpsReuseTest, dont_reuse_tile_broadcast)
{
    graphlib::Graph* graph = get_graph();

    // Use wormhole_b0 config, since on grayskull dest can be reused only on srcA.
    // (we want to check if srcB will be reused)
    DeviceConfig device_config = tt::test::create_device_config(ARCH::WORMHOLE_B0);
    fuse_ops(graph, &device_config);

    // Get fused ops.
    std::vector<BudaOpNode*> fused_ops = get_fused_ops(graph);

    // Find the operator with tile broadcast and confirm that its outputs are not reused.
    for (auto fused_op : fused_ops)
    {
        auto fused = fused_op->get_fused_op();
        for (auto schedule : fused->get_schedules())
        {
            for (auto& op : schedule.ops)
            {
                if (op.name == op_tile_broadcast_name)
                {
                    EXPECT_TRUE(op.output_type != FusedSubOp::OutputType::DEST);
                }
            }
        }
    }
}

struct FuseOpsReuseTestMaximum : public BudaGraphTest
{
    protected:
     virtual std::vector<OpType*> create_graph() override
     {

        auto in0 = create_activation(1, 1, 32, 32);
        auto in1 = create_activation(1, 1, 32, 32);
        auto in2 = create_activation(1, 1, 32, 32);

        auto max_op1 = create_op("maximum", {in0, in1});
        auto add_op1 = create_op("add", {max_op1, in2});
        auto max_op2 = create_op("maximum", {add_op1, in2});
        auto add_op = create_op("add", {max_op2, in2});

        max_output_op = max_op1->name();
        max_input_op= max_op2->name();

        return {add_op};
     }

     std::string max_output_op;
     std::string max_input_op;
};

TEST_F(FuseOpsReuseTestMaximum, fuse_dont_reuse_dest_if_maximum)
{
    Graph* graph = get_graph();
    fuse_ops(graph);
    std::vector<BudaOpNode*> fused_ops = get_fused_ops(graph);

    for (auto fused_op : fused_ops)
    {
        auto fused = fused_op->get_fused_op();
        for (auto schedule : fused->get_schedules())
        {
            for (auto& op : schedule.ops)
            {
                if (op.name == max_output_op)
                {
                    EXPECT_TRUE(op.output_type != FusedSubOp::OutputType::DEST);
                }
                else if (op.name == max_input_op)
                {
                    bool no_dest = std::none_of(op.inputs.begin(), op.inputs.end(), [](const auto &input) { return input.type == FusedSubOpInput::InputType::DEST; });
                    EXPECT_TRUE(no_dest);
                }
            }
        }
    }
}

struct FuseOpsDataFormatsTest : public BudaGraphTest
{
   protected:
    virtual std::vector<OpType*> create_graph() override
    {
        auto in0 = create_activation(1, 1, 64, 64);
        in0->set_output_df(DataFormat::Float16_b);
        auto in1 = create_activation(1, 1, 64, 64);
        in0->set_output_df(DataFormat::Float16_b);
        auto in2 = create_activation(1, 1, 64, 64);
        in0->set_output_df(DataFormat::Float16_b);

        auto add0 = create_op("add", {in0, in1});
        auto add1 = create_op("add", {in0, in2});

        auto out = create_op("add", {add0, add1});

        in2_name = in2->name();

        return {out};
    }

    std::string in2_name;
    ;
};

TEST_F(FuseOpsDataFormatsTest, fuse_same_data_formats)
{
    graphlib::Graph* graph = get_graph();
    fuse_ops(graph);

    // Since all ops have same data format we expect that validation will not assert.
    tt::passes::validate_data_formats(graph, tt::test::create_device_config());
}

TEST_F(FuseOpsDataFormatsTest, fuse_same_data_format_types)
{
    graphlib::Graph* graph = get_graph();

    // Change one data format to Bfp8_b
    auto in2 = graph->get_node_by_name(this->in2_name);
    in2->set_output_df(DataFormat::Bfp8_b);

    fuse_ops(graph);

    // Since all ops have same data format type (b) we expect that validation will not assert.
    tt::passes::validate_data_formats(graph, tt::test::create_device_config());
}

TEST_F(FuseOpsDataFormatsTest, fuse_data_formats_with_float32)
{
    graphlib::Graph* graph = get_graph();

    // Change one data format to Float32
    auto in2 = graph->get_node_by_name(this->in2_name);
    in2->set_output_df(DataFormat::Float32);

    fuse_ops(graph);

    // Since all ops have same data format type (b) or Float32 we expect that validation will not assert.
    tt::passes::validate_data_formats(graph, tt::test::create_device_config());
}

TEST_F(FuseOpsDataFormatsTest, fail_fuse_due_to_unaligned_data_formats)
{
    graphlib::Graph* graph = get_graph();

    // Change one data format to Float16
    auto in2 = graph->get_node_by_name(this->in2_name);
    in2->set_output_df(DataFormat::Float16);

    fuse_ops(graph);

    // Since there is op an with different data format type (a) we expect that validation will assert.
    ASSERT_ANY_THROW(tt::passes::validate_data_formats(graph, tt::test::create_device_config()));
}

struct FuseOpsLimits : public BudaGraphTest, public testing::WithParamInterface<std::tuple<int, int, int, bool>>
{
   protected:
    virtual std::vector<OpType*> create_graph() override
    {
        std::tie(num_inputs, split, num_forks, dram_inputs) = GetParam();
        auto in0 = create_activation(1, 1, 64, 64);
        graphlib::Node* out = in0;

        std::vector<OpType*> outputs;
        for (int i = 0; i < num_inputs; ++i)
        {
            graphlib::Node* in_i = create_activation(1, 1, 64, 64);
            if (not dram_inputs)
                in_i = create_op("buffer", {in_i});
            out = create_op("add", {out, in_i});

            for (int f = 0; (i % split == 0) and f < num_forks; ++f)
            {
                auto in_f = create_activation(1, 1, 64, 64);
                outputs.push_back(create_op("add", {out, in_f}));
            }
        }

        outputs.push_back(out->as<OpType>());

        return outputs;
    }

    int num_inputs = 0;
    int num_forks = 0;
    int split = 0;
    bool dram_inputs = false;
};

TEST_P(FuseOpsLimits, fuse_ops_limits)
{
    graphlib::Graph* graph = get_graph();

    fuse_ops(graph);

    for (auto* fused_op : get_fused_ops(graph))
    {
        auto operands = graph->data_operands(fused_op);
        auto users = graph->data_users(fused_op);
        std::vector<graphlib::Node*> dram_operands;
        auto num_dram_operands = std::count_if(
            operands.begin(),
            operands.end(),
            [](graphlib::Node* n) { return dynamic_cast<graphlib::QueueNode*>(n) != nullptr; });
        auto num_connections = operands.size() + users.size();
        EXPECT_LE((int)num_connections, tt::FusedOp::kMaxNumConnections);
        EXPECT_LE((int)num_dram_operands, tt::FusedOp::kMaxNumDRAMInputs);
    }
}

INSTANTIATE_TEST_SUITE_P(
    FuseOpsLimits,
    FuseOpsLimits,
    testing::Values(
        std::make_tuple(32, 1, 0, false),
        std::make_tuple(32, 2, 1, false),
        std::make_tuple(32, 4, 7, false),
        std::make_tuple(32, 7, 7, false),
        std::make_tuple(32, 15, 7, false),
        std::make_tuple(32, 1, 0, true)));

}  // namespace tt::test
