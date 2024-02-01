// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "gtest/gtest.h"

#include "graph_lib/defines.hpp"
#include "placer/placer.hpp"
#include "placer/lowering_utils.hpp"
#include "placer/best_fit_allocator.hpp"
#include "placer/chip_id_assignment.hpp"
#include "test/common.hpp"

#include "third_party/json/json.hpp"
#include <unordered_map>
#include <optional>
#include <stdexcept>
#include <stdlib.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"

using namespace tt::placer;
using std::unordered_map;
using std::string;
using std::runtime_error;
using tt::graphlib::NodeEpochType;

namespace test
{

unordered_map<string, NodeEpochType> map_ops_to_forward_epoch(const vector<string>& scheduled_ops)
{
    unordered_map<string, NodeEpochType> op_to_epoch_type;
    for (const string& op : scheduled_ops)
    {
        op_to_epoch_type[op] = NodeEpochType::Forward;
    }
    return op_to_epoch_type;
}

} // namespace test

TEST(Placer, single_row)
{
    vector<string> scheduled_ops = {
        "matmul0",
        "matmul1",
        "matmul2",
    };

    tt::DeviceConfig device_config = tt::test::create_device_config();
    PlacerConfig placer_config = {
        .chip_ids = std::vector<std::uint32_t>{0},
        .device_config = device_config,
        .device_grid = {10, 12},
        .strategy = PlacementStrategy::LeftToRight,
        .op_to_grid_shape = lowering::get_op_to_grid_shape(scheduled_ops),
        .op_to_epoch_type = test::map_ops_to_forward_epoch(scheduled_ops),
        .ops_tagged_for_chip_id_break = {},
        .ops_tagged_for_epoch_break = {},
        .fwd_to_bwd_nodes = {},
        .fwd_to_opt_nodes = {},
    };
    vector<OpGroupToPlace> placer_op_group_workload = lowering::generate_simple_placer_workload(placer_config, scheduled_ops);
    PlacerSolution solution = place_onto_chip(placer_config, placer_op_group_workload);
    EXPECT_TRUE(solution.num_epochs == 1);
}

TEST(Placer, multiple_row)
{
    vector<string> scheduled_ops = {
        "matmul0",
        "matmul1",
        "matmul2",
        "matmul3",
        "matmul4",
        "matmul5",
        "matmul6",
        "matmul7",
        "matmul8",
        "matmul9",
        "matmul10",
        "matmul11",
        "matmul12",
    };

    tt::DeviceConfig device_config = tt::test::create_device_config();
    PlacerConfig placer_config = {
        .chip_ids = std::vector<std::uint32_t>{0},
        .device_config = device_config,
        .device_grid = {10, 12},
        .op_to_grid_shape = lowering::get_op_to_grid_shape(scheduled_ops),
        .op_to_epoch_type = test::map_ops_to_forward_epoch(scheduled_ops),
        .ops_tagged_for_chip_id_break = {},
        .ops_tagged_for_epoch_break = {},
        .fwd_to_bwd_nodes = {},
        .fwd_to_opt_nodes = {},
    };

    vector<OpGroupToPlace> placer_op_group_workload = lowering::generate_simple_placer_workload(placer_config, scheduled_ops);
    PlacerSolution solution = place_onto_chip(placer_config, placer_op_group_workload);
    EXPECT_TRUE(solution.num_epochs == 1);
}

TEST(Placer, multiple_epochs)
{
    vector<string> scheduled_ops = {
        "matmul0",
        "matmul1",
        "matmul2",
        "matmul3",
    };

    // Each epoch should hold two ops
    unordered_map<string, GridShape> op_to_grid_shape = {
        {"matmul0", {.rows = 10, .columns=6}},
        {"matmul1", {.rows = 10, .columns=6}},
        {"matmul2", {.rows = 10, .columns=6}},
        {"matmul3", {.rows = 10, .columns=6}},
    };

    tt::DeviceConfig device_config = tt::test::create_device_config();
    PlacerConfig placer_config = {
        .chip_ids = std::vector<std::uint32_t>{0},
        .device_config = device_config,
        .device_grid = {10, 12},
        .op_to_grid_shape = op_to_grid_shape,
        .op_to_epoch_type = test::map_ops_to_forward_epoch(scheduled_ops),
        .ops_tagged_for_chip_id_break = {},
        .ops_tagged_for_epoch_break = {},
        .fwd_to_bwd_nodes = {},
        .fwd_to_opt_nodes = {},
    };
    vector<OpGroupToPlace> placer_op_group_workload = lowering::generate_simple_placer_workload(placer_config, scheduled_ops);
    PlacerSolution solution = place_onto_chip(placer_config, placer_op_group_workload);

    EXPECT_TRUE(solution.num_epochs == 2);
}

TEST(Placer, test_fwd_bwd_epoch_splitting)
{
    vector<string> scheduled_ops = {
        "matmul0",
        "matmul0_bwd",
    };

    // Each epoch should hold two ops
    unordered_map<string, GridShape> op_to_grid_shape = {
        {"matmul0", {.rows = 10, .columns=6}},
        {"matmul0_bwd", {.rows = 10, .columns=6}},
    };
    unordered_map<string, NodeEpochType> op_to_epoch_type = {
        {"matmul0", NodeEpochType::Forward},
        {"matmul0_bwd", NodeEpochType::Backward},
    };

    tt::DeviceConfig device_config = tt::test::create_device_config();
    PlacerConfig placer_config = {
        .chip_ids = std::vector<std::uint32_t>{0},
        .device_config = device_config,
        .device_grid = {10, 12},
        .op_to_grid_shape = op_to_grid_shape,
        .op_to_epoch_type = op_to_epoch_type,
        .ops_tagged_for_chip_id_break = {},
        .ops_tagged_for_epoch_break = {},
        .fwd_to_bwd_nodes = {},
        .fwd_to_opt_nodes = {},
    };
    vector<OpGroupToPlace> placer_op_group_workload = lowering::generate_simple_placer_workload(placer_config, scheduled_ops);
    PlacerSolution solution = place_onto_chip(placer_config, placer_op_group_workload);

    EXPECT_TRUE(solution.num_epochs == 2);
}

TEST(Placer, test_multichip_fwd)
{
    vector<string> scheduled_ops = {
        "matmul0",
        "matmul1",
    };

    // Each epoch should hold two ops
    unordered_map<string, GridShape> op_to_grid_shape = {
        {"matmul0", {.rows = 10, .columns=6}},
        {"matmul1", {.rows = 10, .columns=6}},
    };
    unordered_map<string, NodeEpochType> op_to_epoch_type = {
        {"matmul0", NodeEpochType::Forward},
        {"matmul1", NodeEpochType::Forward},
    };

    ChipPlacerConfig chip_placer_config = {
        .chip_ids = std::vector<std::uint32_t>{0, 1},
        .arch_name = "grayskull",
        .op_to_epoch_type = test::map_ops_to_forward_epoch(scheduled_ops),
        .ops_tagged_for_chip_id_break = {"matmul1"},
        .ops_tagged_for_epoch_break = {},
        .fwd_to_bwd_nodes = {},
        .fwd_to_opt_nodes = {},
    };
    OpToChipIdAssignment op_to_chip_id_assignment = get_op_to_chip_id_assignment(chip_placer_config, scheduled_ops);

    tt::DeviceConfig device_config = tt::test::create_device_config();
    PlacerConfig placer_config = {
        .chip_ids = std::vector<std::uint32_t>{0, 1},
        .device_config = device_config,
        .device_grid = {10, 12},
        .op_to_grid_shape = op_to_grid_shape,
        .op_to_epoch_type = op_to_epoch_type,
        .ops_tagged_for_chip_id_break = {"matmul1"},
        .ops_tagged_for_epoch_break = {},
        .fwd_to_bwd_nodes = {},
        .fwd_to_opt_nodes = {},
        .op_to_chip_id_assignment = op_to_chip_id_assignment,
    };


    PlacerSolution solution = placer(placer_config, scheduled_ops);

    EXPECT_EQ(solution.num_epochs, 2);
}


TEST(Placer, test_multichip_fwd_and_bwd)
{
    vector<string> scheduled_ops = {
        "matmul0",
        "matmul1",
        "matmul0_bwd",
        "matmul1_bwd",
    };

    // Each epoch should hold two ops
    unordered_map<string, GridShape> op_to_grid_shape = {
        {"matmul0",     {.rows = 10, .columns = 6}},
        {"matmul0_bwd", {.rows = 10, .columns = 6}},
        {"matmul1",     {.rows = 10, .columns = 6}},
        {"matmul1_bwd", {.rows = 10, .columns = 6}},
    };
    unordered_map<string, NodeEpochType> op_to_epoch_type = {
        {"matmul0",     NodeEpochType::Forward},
        {"matmul0_bwd", NodeEpochType::Backward},
        {"matmul1",     NodeEpochType::Forward},
        {"matmul1_bwd", NodeEpochType::Backward},
    };
    ChipPlacerConfig chip_placer_config = {
        .chip_ids = std::vector<std::uint32_t>{0, 1},
        .arch_name = "grayskull",
        .op_to_epoch_type = test::map_ops_to_forward_epoch(scheduled_ops),
        .ops_tagged_for_chip_id_break = {"matmul1"},
        .ops_tagged_for_epoch_break = {},
        .fwd_to_bwd_nodes = {{"matmul0", {"matmul0_bwd"}}, {"matmul1", {"matmul1_bwd"}}},
        .fwd_to_opt_nodes = {},
    };
    OpToChipIdAssignment op_to_chip_id_assignment = get_op_to_chip_id_assignment(chip_placer_config, scheduled_ops);

    tt::DeviceConfig device_config = tt::test::create_device_config();
    PlacerConfig placer_config = {
        .chip_ids = std::vector<std::uint32_t>{0, 1},
        .device_config = device_config,
        .device_grid = {10, 12},
        .op_to_grid_shape = op_to_grid_shape,
        .op_to_epoch_type = op_to_epoch_type,
        .ops_tagged_for_chip_id_break = {"matmul1"},
        .ops_tagged_for_epoch_break = {},
        .fwd_to_bwd_nodes = {{"matmul0", {"matmul0_bwd"}}, {"matmul1", {"matmul1_bwd"}}},
        .fwd_to_opt_nodes = {},
        .op_to_chip_id_assignment = op_to_chip_id_assignment,
    };
    PlacerSolution solution = placer(placer_config, scheduled_ops);

    EXPECT_EQ(solution.num_epochs, 4);
}


TEST(Placer, triplet_placement)
{
    setenv("PYBUDA_TRIPLET_PLACEMENT", "1", 0);
    GridShape matmul0_grid = {.rows = 10, .columns = 6};
    GridShape matmul0_bwd0_grid = {.rows = 2, .columns = 2};
    GridShape matmul0_bwd1_grid = {.rows = 2, .columns = 2};
    GridShape matmul0_bwd2_grid = {.rows = 2, .columns = 2};

    unordered_map<string, GridShape> op_to_grid_shape = {
        {"matmul0", matmul0_grid},
        {"matmul0_bwd0", matmul0_bwd0_grid},
        {"matmul0_bwd1", matmul0_bwd1_grid},
        {"matmul0_bwd2", matmul0_bwd2_grid},
    };

    unordered_map<string, NodeEpochType> op_to_epoch_type ={
        {"matmul0", NodeEpochType::Forward},
        {"matmul0_bwd0", NodeEpochType::Backward},
        {"matmul0_bwd1", NodeEpochType::Backward},
        {"matmul0_bwd2", NodeEpochType::Backward},
    };

    // We can also annotate properties on op-groupings like:
    //   force-epoch-break, force-chip-break, force-new-row
    // and future support for partial placements
    OpGroupToPlace op_group0 = {
        .op_names = {"matmul0"},
        .op_name_to_relative_offset_from_first_op = {},
    };

    OpGroupToPlace op_group1 = {
        .op_names = {"matmul0_bwd0", "matmul0_bwd1", "matmul0_bwd2"},
        .op_name_to_relative_offset_from_first_op = {
            {"matmul0_bwd1", {.row_offset = 2, .column_offset = 0}},
            {"matmul0_bwd2", {.row_offset = 4, .column_offset = 0}},
        }
    };

    vector<OpGroupToPlace> placer_op_group_workload = {
        op_group0,
        op_group1,
    };

    tt::DeviceConfig device_config = tt::test::create_device_config();
    PlacerConfig placer_config = {
        .chip_ids = std::vector<std::uint32_t>{0},
        .device_config = device_config,
        .device_grid = {10, 12},
        .op_to_grid_shape = op_to_grid_shape,
        .op_to_epoch_type = op_to_epoch_type,
        .ops_tagged_for_chip_id_break = {},
        .ops_tagged_for_epoch_break = {},
        .fwd_to_bwd_nodes = {},
        .fwd_to_opt_nodes = {},
    };

    PlacerSolution solution = place_onto_chip(placer_config, placer_op_group_workload);

    // Final Checks
    const OpPlacement& matmul_bwd1_placement = solution.name_to_op_placement.at("matmul0_bwd1");
    const CoordRange& matmul_bwd1_coords = matmul_bwd1_placement.placed_cores;

    EXPECT_EQ(solution.num_epochs, 2);
    EXPECT_EQ(matmul_bwd1_placement.epoch_id(), 1);
    EXPECT_EQ(matmul_bwd1_coords.start.row, 2); // expect bwd1 to be right below bwd0
    EXPECT_EQ(matmul_bwd1_coords.start.col, 0);
}


TEST(Placer, test_epoch_breaks)
{
    vector<string> scheduled_ops = {
        "matmul0",
        "matmul1",
        "matmul2",
        "matmul3",
    };

    // Each epoch should hold two ops
    unordered_map<string, GridShape> op_to_grid_shape = {
        {"matmul0",     {.rows = 2, .columns = 2}},
        {"matmul1",     {.rows = 2, .columns = 2}},
        {"matmul2",     {.rows = 2, .columns = 2}},
        {"matmul3",     {.rows = 2, .columns = 2}},
    };
    unordered_map<string, NodeEpochType> op_to_epoch_type = {
        {"matmul0",     NodeEpochType::Forward},
        {"matmul1",     NodeEpochType::Forward},
        {"matmul2",     NodeEpochType::Forward},
        {"matmul3",     NodeEpochType::Forward},
    };
    ChipPlacerConfig chip_placer_config = {
        .chip_ids = std::vector<std::uint32_t>{0},
        .arch_name = "grayskull",
        .op_to_epoch_type = test::map_ops_to_forward_epoch(scheduled_ops),
        .ops_tagged_for_chip_id_break = {},
        .ops_tagged_for_epoch_break = {
            "matmul1",
            "matmul2"
        },
        .fwd_to_bwd_nodes = {},
        .fwd_to_opt_nodes = {},
    };
    OpToChipIdAssignment op_to_chip_id_assignment = get_op_to_chip_id_assignment(chip_placer_config, scheduled_ops);

    tt::DeviceConfig device_config = tt::test::create_device_config();
    PlacerConfig placer_config = {
        .chip_ids = std::vector<std::uint32_t>{0, 1},
        .device_config = device_config,
        .device_grid = {10, 12},
        .op_to_grid_shape = op_to_grid_shape,
        .op_to_epoch_type = op_to_epoch_type,
        .ops_tagged_for_chip_id_break = {},
        .ops_tagged_for_epoch_break = {
            "matmul1",
            "matmul2"
        },
        .fwd_to_bwd_nodes = {},
        .fwd_to_opt_nodes = {},
        .op_to_chip_id_assignment = op_to_chip_id_assignment,
    };

    PlacerSolution solution = placer(placer_config, scheduled_ops);
    //dump_placer_solution_json_to_file(solution);

    EXPECT_EQ(solution.num_epochs, 3);
}


TEST(Placer, test_row_harvesting)
{
    vector<string> scheduled_ops = {
        "matmul0",
        "matmul1",
        "matmul2",
    };

    uint32_t default_row_cores = 5;
    uint32_t default_column_cores = 6;

    tt::DeviceConfig device_config = tt::test::create_device_config();
    PlacerConfig placer_config = {
        .chip_ids = std::vector<std::uint32_t>{0},
        .device_config = device_config,
        .device_grid = {10, 12},
        .harvested_rows = {5, 6, 7, 8, 9},
        .strategy = PlacementStrategy::LeftToRight,
        .op_to_grid_shape = lowering::get_op_to_grid_shape(scheduled_ops, default_row_cores, default_column_cores),
        .op_to_epoch_type = test::map_ops_to_forward_epoch(scheduled_ops),
        .ops_tagged_for_chip_id_break = {},
        .ops_tagged_for_epoch_break = {},
        .fwd_to_bwd_nodes = {},
        .fwd_to_opt_nodes = {},
    };
    vector<OpGroupToPlace> placer_op_group_workload = lowering::generate_simple_placer_workload(placer_config, scheduled_ops);
    PlacerSolution solution = place_onto_chip(placer_config, placer_op_group_workload);

    //std::cout << std::setw(4) << solution.to_json() << std::endl;
    // technically all three ops can be placed on the same epoch, but because the bottom half of the chip is harvested,
    // this needs to spill into two epochs
    EXPECT_TRUE(solution.num_epochs == 2);
}

TEST(Placer, test_manual_transpose_ops)
{
    GridShape matmul0_grid = {.rows = 10, .columns = 6}; 

    unordered_map<string, GridShape> op_to_grid_shape = {
        {"matmul0", matmul0_grid}, 
    };

    unordered_map<string, NodeEpochType> op_to_epoch_type ={
        {"matmul0", NodeEpochType::Forward}, 
    };
 
    OpGroupToPlace op_group0 = {
        .op_names = {"matmul0"},
        .op_name_to_relative_offset_from_first_op = {},
    }; 

    vector<OpGroupToPlace> placer_op_group_workload = {
        op_group0, 
    };
    std::optional<Coord> start = std::nullopt;
    tt::DeviceConfig device_config = tt::test::create_device_config();
    PlacerConfig placer_config = {
        .chip_ids = std::vector<std::uint32_t>{0},
        .device_config = device_config,
        .device_grid = {10, 12},
        .op_to_grid_shape = op_to_grid_shape,
        .op_to_epoch_type = op_to_epoch_type,
        .ops_tagged_for_chip_id_break = {},
        .ops_tagged_for_epoch_break = {},
        .fwd_to_bwd_nodes = {},
        .fwd_to_opt_nodes = {},
        .op_to_overrides = {
            {
                "matmul0", PlacerOpOverride(start, true /* transpose_op */, std::nullopt /* chip_id */)
            }
        },
    };

    PlacerSolution solution = place_onto_chip(placer_config, placer_op_group_workload);

    // Final Checks
    const OpPlacement& matmul0_placement = solution.name_to_op_placement.at("matmul0");
    const CoordRange& matmul0_coords = matmul0_placement.placed_cores;
    const uint32_t matmul0_coords_r = matmul0_coords.size_r();
    const uint32_t matmul0_coords_c = matmul0_coords.size_c(); 
    const bool matmul0_grid_transpose = matmul0_placement.grid_transpose;
 
    EXPECT_EQ(matmul0_coords_r, 6); // expect matmul0 to be transposed to (6x10) 
    EXPECT_EQ(matmul0_coords_c, 10);
    EXPECT_EQ(matmul0_grid_transpose, true);
}

TEST(Placer, test_auto_transpose_ops)
{
    GridShape matmul0_grid = {.rows = 2, .columns = 6};
    GridShape matmul1_grid = {.rows = 3, .columns = 2}; 

    unordered_map<string, GridShape> op_to_grid_shape = {
        {"matmul0", matmul0_grid}, 
        {"matmul1", matmul1_grid},
    };

    unordered_map<string, NodeEpochType> op_to_epoch_type ={
        {"matmul0", NodeEpochType::Forward}, 
        {"matmul1", NodeEpochType::Forward}, 
    };
 
    OpGroupToPlace op_group0 = {
        .op_names = {"matmul0", "matmul1"},
        .op_name_to_relative_offset_from_first_op = {},
    }; 

    vector<OpGroupToPlace> placer_op_group_workload = {
        op_group0, 
    };

    tt::DeviceConfig device_config = tt::test::create_device_config();
    PlacerConfig placer_config = {
        .chip_ids = std::vector<std::uint32_t>{0},
        .device_config = device_config,
        .device_grid = {10, 12},
        .op_to_grid_shape = op_to_grid_shape,
        .op_to_epoch_type = op_to_epoch_type,
        .ops_tagged_for_chip_id_break = {},
        .ops_tagged_for_epoch_break = {},
        .fwd_to_bwd_nodes = {},
        .fwd_to_opt_nodes = {},
        .enable_auto_transposing_placement = true, 
    };

    PlacerSolution solution = place_onto_chip(placer_config, placer_op_group_workload);

    // Final Checks
    const OpPlacement& matmul1_placement = solution.name_to_op_placement.at("matmul1");
    const CoordRange& matmul1_coords = matmul1_placement.placed_cores;
    const uint32_t matmul1_coords_r = matmul1_coords.size_r();
    const uint32_t matmul1_coords_c = matmul1_coords.size_c(); 
    const bool matmul1_grid_transpose = matmul1_placement.grid_transpose;
 
    EXPECT_EQ(matmul1_coords_r, 2); // expect matmul1 to be transposed to (2x3) based on the row size
    EXPECT_EQ(matmul1_coords_c, 3);
    EXPECT_EQ(matmul1_grid_transpose, true);
}



/* Turn off until deallocate is back on
TEST(Placer, best_fit_allocator)
{
    std::uint32_t start_addr = 0x100;
    std::uint32_t end_addr = 0x8100;
    std::uint32_t size = end_addr - start_addr;
    auto bfa = BestFitAllocator(start_addr, end_addr);

    // Allocate everything, and deallocate
    std::uint32_t addr;
    EXPECT_TRUE(bfa.allocate(size, addr));
    EXPECT_EQ(addr, start_addr);
    bfa.deallocate(addr);

    // Allocate two half-pieces
    std::uint32_t addr1, addr2;
    EXPECT_TRUE(bfa.allocate(size/2, addr1));
    EXPECT_TRUE(bfa.allocate(size/2, addr2));
    EXPECT_EQ(addr1, start_addr);
    EXPECT_EQ(addr2, start_addr + size/2);
    EXPECT_FALSE(bfa.allocate(0x10, addr));

    // Deallocate out of order
    bfa.deallocate(addr2);
    bfa.deallocate(addr1);

    // Allocate three pieces of 0x100, check that they are merged back with the whole area
    std::uint32_t addr3;
    EXPECT_TRUE(bfa.allocate(0x100, addr1));
    EXPECT_TRUE(bfa.allocate(0x100, addr2));
    EXPECT_TRUE(bfa.allocate(0x100, addr3));
    EXPECT_FALSE(bfa.allocate(size, addr)); // no room

    // Deallocate out of order
    bfa.deallocate(addr3);
    bfa.deallocate(addr1);
    bfa.deallocate(addr2);
    EXPECT_TRUE(bfa.allocate(size, addr)); // should have room now
    bfa.deallocate(addr);
}
*/

#pragma GCC diagnostic pop
