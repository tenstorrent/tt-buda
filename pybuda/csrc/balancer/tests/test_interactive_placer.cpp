// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include <fstream>

#include "graph_lib/query.hpp"
#include "gtest/gtest.h"
#include "json.hpp"
#include "placer/interactive_placer.hpp"
#include "placer/placer.hpp"
#include "test/common.hpp"
#include "test_balancer_utils.hpp"

namespace tt::test
{
using namespace balancer;

struct InteractivePlacerSanity : testing::Test
{
};

// Unit test for InteractivePlacer::rewind_to(const std::string &op_name).
//
TEST_F(InteractivePlacerSanity, rewind_to)
{
    balancer::BalancerConfig balancer_config = create_balancer_config();
    placer::InteractivePlacer interactive_placer(nullptr /*graph*/, balancer_config);
    std::map<std::string, placer::GridShape> op_to_grid_shape = {
        {"op1", placer::GridShape(1, 1)},
        {"op2", placer::GridShape(1, 10)},
        {"op3_pair1", placer::GridShape(5, 1)},
        {"op4_pair2", placer::GridShape(5, 3)},
        {"op5", placer::GridShape(1, 9)},
        {"op6_pair1", placer::GridShape(5, 2)},
        {"op7_pair2", placer::GridShape(5, 2)},
        {"op8", placer::GridShape(3, 3)}};

    std::optional<std::pair<std::string, placer::GridShape>> buffered_op;
    std::optional<placer::CoordRange> op_placement;
    interactive_placer.get_op_overrides().emplace("op5", placer::PlacerOpOverride::force_op_transpose());
    for (const auto& to_place : op_to_grid_shape)
    {
        if (to_place.first.find("pair1") != std::string::npos)
        {
            buffered_op = to_place;
            continue;
        }

        if (buffered_op.has_value())
        {
            op_placement = interactive_placer.place_two_ops_rowwise(
                buffered_op->first, buffered_op->second, to_place.first, to_place.second, true /* enable_transpose */);
            buffered_op.reset();
        }
        else
        {
            op_placement = interactive_placer.place_op(to_place.first, to_place.second, true /* enable_transpose */);
        }

        ASSERT_TRUE(op_placement.has_value());
    }

    // Snapshot placement configuration prior to rewind. Needs to remain unchanged after rewind.
    //
    std::unordered_map<std::string, placer::OpPlacement> pre_rewind_placements =
        interactive_placer.get_current_name_to_op_placement();
    interactive_placer.rewind_to(std::prev(op_to_grid_shape.end())->first);

    // Verify that user overrides are preserved after rewind.
    //
    EXPECT_EQ(interactive_placer.get_op_overrides().at("op5"), placer::PlacerOpOverride::force_op_transpose());

    // Verify that placement is identical for rewinded ops.
    //
    for (const auto& pre_rewind_placement : pre_rewind_placements)
    {
        if (pre_rewind_placement.first == std::prev(op_to_grid_shape.end())->first)
        {
            continue;
        }

        auto post_rewind_placement =
            interactive_placer.get_current_name_to_op_placement().find(pre_rewind_placement.first);
        ASSERT_TRUE(post_rewind_placement != interactive_placer.get_current_name_to_op_placement().end());
        EXPECT_EQ(pre_rewind_placement.second, post_rewind_placement->second);
    }
}

// Unit test for InteractivePlacer::place_op for multi_chip
//
TEST_F(InteractivePlacerSanity, chip_id_override)
{
    const std::vector<std::uint32_t> chip_ids = {0, 1, 2, 3};
    balancer::BalancerConfig balancer_config = create_balancer_config(ARCH::WORMHOLE_B0, chip_ids, balancer::PolicyType::Ribbon);
    placer::InteractivePlacer interactive_placer(nullptr /*graph*/, balancer_config);
    std::map<std::string, placer::GridShape> op_to_grid_shape = {
        {"op1", placer::GridShape(8, 8)},
        {"op2", placer::GridShape(8, 8)},
        {"op3", placer::GridShape(8, 8)},
        {"op4", placer::GridShape(8, 8)},
    };

    interactive_placer.get_op_overrides()["op1"].chip_id = 3;
    interactive_placer.get_op_overrides()["op2"].chip_id = 2;
    interactive_placer.get_op_overrides()["op3"].chip_id = 1;

    for (const auto& to_place : op_to_grid_shape)
    {
        std::optional<placer::CoordRange> op_placement = interactive_placer.place_op(to_place.first, to_place.second);
        ASSERT_TRUE(op_placement.has_value());
    }

    const unordered_map<string, placer::OpPlacement>& name_to_op_placement = interactive_placer.get_current_name_to_op_placement();
    EXPECT_EQ(name_to_op_placement.at("op1").chip_id, 3);
    EXPECT_EQ(name_to_op_placement.at("op2").chip_id, 2);
    EXPECT_EQ(name_to_op_placement.at("op3").chip_id, 1);
    EXPECT_EQ(name_to_op_placement.at("op4").chip_id, 0);
    EXPECT_EQ(interactive_placer.get_current_epoch_index(), 3);
}

// Unit test for ChipPlacementPolicy::SNAKE - order the given chip_ids on a snake pattern based on budabackend/cluster_desc.yaml
//
TEST_F(InteractivePlacerSanity, chip_id_galaxy_snake)
{
    const std::vector<std::uint32_t> chip_ids = {1, 2, 18, 25, 19, 24, 20, 23};
    balancer::BalancerConfig balancer_config = create_balancer_config(
        ARCH::WORMHOLE_B0,
        chip_ids,
        balancer::PolicyType::Ribbon,
        "pybuda/test/galaxy/one_shelf_eth_connections.yaml",
        "pybuda/test/galaxy/one_shelf_runtime_params.yaml",
        placer::ChipPlacementPolicy::SNAKE
    );
    placer::InteractivePlacer interactive_placer(nullptr /*graph*/, balancer_config);
    std::map<std::string, placer::GridShape> op_to_grid_shape = {
        {"op1", placer::GridShape(8, 8)},
        {"op2", placer::GridShape(8, 8)},
        {"op3", placer::GridShape(8, 8)},
        {"op4", placer::GridShape(8, 8)},
        {"op5", placer::GridShape(8, 8)},
        {"op6", placer::GridShape(8, 8)},
        {"op7", placer::GridShape(8, 8)},
        {"op8", placer::GridShape(8, 8)},
    };

    for (const auto& to_place : op_to_grid_shape)
    {
        std::optional<placer::CoordRange> op_placement = interactive_placer.place_op(to_place.first, to_place.second);
        ASSERT_TRUE(op_placement.has_value());
    }

    const unordered_map<string, placer::OpPlacement>& name_to_op_placement = interactive_placer.get_current_name_to_op_placement();
    EXPECT_EQ(name_to_op_placement.at("op1").chip_id, 1);
    EXPECT_EQ(name_to_op_placement.at("op2").chip_id, 25);
    EXPECT_EQ(name_to_op_placement.at("op3").chip_id, 24);
    EXPECT_EQ(name_to_op_placement.at("op4").chip_id, 23);
    EXPECT_EQ(name_to_op_placement.at("op5").chip_id, 20);
    EXPECT_EQ(name_to_op_placement.at("op6").chip_id, 19);
    EXPECT_EQ(name_to_op_placement.at("op7").chip_id, 18);
    EXPECT_EQ(name_to_op_placement.at("op8").chip_id, 2);
    EXPECT_EQ(interactive_placer.get_current_epoch_index(), 7);
}

struct MultiLayerGraph : public BudaGraphTest
{
   protected:
    std::string layer_string(int chip_id) const { return "layer." + std::to_string(chip_id); }

    virtual std::vector<OpType*> create_graph() override
    {
        std::uint32_t seq_len = 128;
        std::uint32_t embed = 128;
        std::uint32_t hidden = 128;

        auto act = create_activation(shape(1, 1, seq_len, embed));
        auto w0 = create_parameter(shape(1, 1, embed, hidden));
        auto w1 = create_parameter(shape(1, 1, hidden, embed));
        auto w2 = create_parameter(shape(1, 1, hidden, embed));

        auto e0 = create_op("matmul", {act, w0});
        auto g0 = create_op("gelu", {e0});
        auto e1 = create_op("matmul", {g0, w1});
        auto g1 = create_op("gelu", {e1});
        auto e2 = create_op("matmul", {g1, w2});
        auto g2 = create_op("gelu", {e2});

        e0->tag("layer", layer_string(chip_ids[0]));
        g0->tag("layer", layer_string(chip_ids[0]));
        e1->tag("layer", layer_string(chip_ids[1]));
        g1->tag("layer", layer_string(chip_ids[1]));
        e2->tag("layer", layer_string(chip_ids[2]));
        g2->tag("layer", layer_string(chip_ids[2]));

        return {g2};
    }

    const std::vector<std::uint32_t> chip_ids = {0, 1, 2};
};

// Unit test for InteractivePlacer::place_op using layer predicate
//
TEST_F(MultiLayerGraph, chip_id_layer_override)
{
    graphlib::Graph* graph = get_graph();

    balancer::BalancerConfig balancer_config =
        create_balancer_config(ARCH::WORMHOLE_B0, chip_ids, balancer::PolicyType::Ribbon);
    balancer_config.op_name_to_placer_overrides = placer::match_op_names_to_placer_overrides(
        graph,
        {
            {graphlib::query::layer_regex("l.*\\.0"), placer::PlacerOpOverride::override_chip_id(chip_ids.at(0))},
            {graphlib::query::layer_regex("l.*\\.1"), placer::PlacerOpOverride::override_chip_id(chip_ids.at(1))},
            {graphlib::query::layer_regex("l.*\\.2"), placer::PlacerOpOverride::override_chip_id(chip_ids.at(2))},
        });

    // Last layer is output layer, hence chip 2 needs to be mmio
    balancer_config.device_config.chips_with_mmio.push_back(2);

    placer::InteractivePlacer interactive_placer(graph, balancer_config);

    for (auto* node : graphlib::topological_sort(*graph))
    {
        if (node->node_type() != graphlib::NodeType::kBudaOp)
            continue;
        std::optional<placer::CoordRange> op_placement = interactive_placer.place_op(node->name(), GridShape(1, 1));
        ASSERT_TRUE(op_placement.has_value());
    }

    const unordered_map<string, placer::OpPlacement>& name_to_op_placement = interactive_placer.get_current_name_to_op_placement();

    for (auto* node : graphlib::topological_sort(*graph))
    {
        if (node->node_type() != graphlib::NodeType::kBudaOp)
            continue;

        int chip_id = name_to_op_placement.at(node->name()).chip_id;
        std::string expected_layer_string = node->as<graphlib::TaggedNode>()->tag_value<std::string>("layer");
        std::string actual_layer_string = layer_string(chip_id);
        ASSERT_EQ(actual_layer_string, expected_layer_string);
    }
}

TEST_F(InteractivePlacerSanity, nebula_grid_8x8)
{
    ASSERT_FALSE(getenv("PYBUDA_NEBULA_GALAXY_PLACER"));
    setenv("PYBUDA_NEBULA_GALAXY_PLACER", "1", 0);
    const std::vector<std::uint32_t> chip_ids = {0};
    balancer::BalancerConfig balancer_config = create_balancer_config(ARCH::WORMHOLE_B0, chip_ids, balancer::PolicyType::Ribbon);
    placer::InteractivePlacer interactive_placer(nullptr /*graph*/, balancer_config);
    std::map<std::string, placer::GridShape> op_to_grid_shape = {
        {"op1", placer::GridShape(9, 1)},
    };

    interactive_placer.get_op_overrides()["op1"].chip_id = 0;

    std::optional<placer::CoordRange> op_placement = interactive_placer.place_op("op1", op_to_grid_shape["op1"]);

    // cannot fit on nebula 8x8 grid
    EXPECT_EQ(op_placement.has_value(), false);
    unsetenv("PYBUDA_NEBULA_GALAXY_PLACER");
}

}  // namespace tt::test
