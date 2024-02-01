// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "graph_lib/defines.hpp"
#include "graph_lib/utils.hpp"
#include "graph_lib/node_types.hpp"
#include "gtest/gtest.h"
#include "passes/dataformat.hpp"
#include "test/common.hpp"

using namespace tt;
namespace tt::test
{

struct ProducerQueueDataFormatMismatch : public BudaGraphTest
{
   protected:
    virtual std::vector<OpType*> create_graph() override
    {
        auto in0 = create_activation(1, 1, 32, 32);
        auto in1 = create_activation(1, 1, 32, 32);
        auto in2 = create_activation(1, 1, 32, 64);

        auto add0 = create_op("add", {in0, in1});
        auto queue = create_queue(add0);
        auto add1 = create_op("add", {queue, in2});

        queue_name = queue->name();

        return {add1};
    }
    std::string queue_name;
};

TEST_F(ProducerQueueDataFormatMismatch, test_data_formats_mismatch)
{
    graphlib::Graph* graph = get_graph();

    Node* queue = graph->get_node_by_name(queue_name);
    queue->set_output_df(DataFormat::Bfp8_b);

    ASSERT_ANY_THROW(tt::passes::validate_data_formats(graph, tt::test::create_device_config()));
}

TEST_F(ProducerQueueDataFormatMismatch, test_data_formats_fixed)
{
    graphlib::Graph* graph = get_graph();

    Node* queue = graph->get_node_by_name(queue_name);
    queue->set_output_df(DataFormat::Bfp8_b);

    tt::passes::satisfy_data_format_constraints(graph, true /* fp32_acc_supported */);
    tt::passes::validate_data_formats(graph, tt::test::create_device_config());
}


struct ReduceFeedingSplice : public BudaGraphTest
{
   protected:
    virtual std::vector<OpType*> create_graph() override
    {
        auto constant = create_constant(1, 32, 32, 32);
        auto activations = create_activation(1, 1, 4096, 32);
        auto reduce = create_op(
            "reduce",
            {activations},
            tt::BudaOpAttrs{{"dim", std::string("z")}, {"type", std::string("max")}, {"z", 4}},
            1,
            std::string("max"),
            4);

        append_tm(graphlib::OpType("vslice", {4}), reduce, 0);
        reduce->set_shape(graphlib::Shape::create({1, 1, 1024, 32}));
        reduce_name = reduce->name();
        auto splice = create_op(
            "splice",
            {reduce, constant},
            {
                {"splice_type", std::string("concatenate")},
                {"dim", 3},
                {"input_shapes", std::vector{reduce->shape().as_tuple(), constant->shape().as_tuple()}},
            });
        append_tm(graphlib::OpType("vslice", {32}), splice, 0);

        // Configure dfs
        activations->set_output_df(DataFormat::Bfp8_b);
        reduce->as<graphlib::BudaOpNode>()->set_intermediate_df(DataFormat::Bfp8_b);
        reduce->set_output_df(DataFormat::Bfp8_b);


        return {splice};
    }
    std::string reduce_name;
};

struct AliasedQueueWriteback : public BudaGraphTest
{
   protected:
    virtual std::vector<OpType*> create_graph() override
    {
        auto activations = create_activation(1, 1, 4096, 32);
        auto gelu = create_op("gelu", {activations});
        auto nop = create_op("nop", {gelu});
        this->activations_name = activations->name();
        this->nop_name = nop->name();

        activations->set_output_df(DataFormat::Float32);
        gelu->set_output_df(DataFormat::Float16_b);

        return {nop};
    }
    std::string activations_name;
    std::string nop_name;
};

TEST_F(AliasedQueueWriteback, test_aliased_queue_writeback)
{
    graphlib::Graph* graph = get_graph();
    auto input = graph->get_node_by_name(activations_name);
    auto output = graph->get_node_by_name("output0");

    graph->add_edge(output, input, tt::graphlib::EdgeType::kPartialDataCopy);

    ASSERT_ANY_THROW(::passes::validate_data_formats(graph,  tt::test::create_device_config()));
}

TEST_F(AliasedQueueWriteback, test_aliased_queue_writeback_satisfied)
{
    graphlib::Graph* graph = get_graph();
    auto input = graph->get_node_by_name(activations_name);
    auto output = graph->get_node_by_name("output0");

    graph->add_edge(output, input, tt::graphlib::EdgeType::kPartialDataCopy);

    tt::passes::satisfy_data_format_constraints(graph, true /* fp32_acc_supported */);
    tt::passes::validate_data_formats(graph,  tt::test::create_device_config());
}

struct NopDataFormatPropagation : public BudaGraphTest
{
   protected:
    virtual std::vector<OpType*> create_graph() override
    {
        auto activation = create_activation(1, 1, 128, 128);
        auto weight = create_parameter(shape(1, 1, 128, 128));
        auto matmul = create_op("matmul", {activation, weight});
        auto nop = create_op("nop", {matmul});
        auto gelu = create_op("gelu", {nop});

        this->nop_name = nop->name();

        // Configure dfs
        activation->set_output_df(DataFormat::Bfp8_b);
        weight->set_output_df(DataFormat::Bfp8_b);
        matmul->set_output_df(DataFormat::Float16_b); // Format promotion on the matmul
        nop->set_output_df(DataFormat::Bfp8_b);
        gelu->set_output_df(DataFormat::Bfp8_b);

        return {gelu};
    }
    std::string nop_name;
};

TEST_F(NopDataFormatPropagation, test_nop_format_promotion)
{
    graphlib::Graph* graph = get_graph();
    tt::passes::satisfy_data_format_constraints(graph, true /* fp32_acc_supported */);

    EXPECT_EQ(graph->get_node_by_name(nop_name)->output_df(), DataFormat::Float16_b);
}

}  // namespace tt::test
