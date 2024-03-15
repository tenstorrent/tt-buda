// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "placer/dram.hpp"

#include "balancer/types.hpp"
#include "graph_lib/defines.hpp"
#include "graph_lib/node_types.hpp"
#include "gtest/gtest.h"
#include "placer/best_fit_allocator.hpp"
#include "placer/chip_id_assignment.hpp"
#include "placer/dram_allocator.hpp"
#include "placer/lowering_utils.hpp"
#include "placer/placer.hpp"
#include "test/common.hpp"

using namespace tt::placer;
using std::runtime_error;
using std::string;
using std::unordered_map;
using tt::graphlib::NodeEpochType;
using tt::ARCH;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"

//
// Tests for DRAM allocators
//
namespace test
{
extern unordered_map<string, NodeEpochType> map_ops_to_forward_epoch(const vector<string> &scheduled_ops);

// Test parameters for each test
struct TestConfig
{
    DRAMPlacementAlgorithm algo = DRAMPlacementAlgorithm::ROUND_ROBIN;
    bool input_queues_on_host = true;
    bool output_queues_on_host = true;
    tt::DramQueueMap manual_dram_queue_placemenet = {};
};

class DRAMPlacerTest : public testing::TestWithParam<ARCH>
{
    // List of queues to be placed
    std::vector<std::pair<QueuePlacement, QueueDRAMPlacementParameters>> queue_placement_params;

    // DRAM allocator, DUT
    std::unique_ptr<DramAllocator> allocator;

   public:
    // Configs
    tt::DeviceConfig device_config = tt::test::create_device_config();
    std::unique_ptr<DramPlacerConfig> dram_config;

    // Main graph
    std::unique_ptr<Graph> graph;

    // Use user-friendly test parameter names
    struct PrintToStringParamName
    {
        template <class ParamType>
        std::string operator()(const testing::TestParamInfo<ParamType> &info) const
        {
            auto arch = static_cast<ARCH>(info.param);
            return to_string_arch(arch);
        }
    };

    // Alias to make code more readable
    ARCH get_arch() { return GetParam(); }

    // Common overrides
    void SetUp(DRAMPlacementAlgorithm algo)
    {
        TestConfig test_cfg;
        test_cfg.algo = algo;
        SetUp(test_cfg);
    }

    void SetUp(TestConfig test_cfg)
    {
        device_config = tt::test::create_device_config(get_arch());
        std::vector<std::string> scheduled_ops;
        PlacerConfig placer_config = {
            .chip_ids = std::vector<std::uint32_t>{0},
            .device_config = device_config,
            .device_grid = {(std::uint32_t)device_config.grid_size.r, (std::uint32_t)device_config.grid_size.c},
            .strategy = PlacementStrategy::LeftToRight,
            .op_to_grid_shape = lowering::get_op_to_grid_shape(scheduled_ops),
            .op_to_epoch_type = test::map_ops_to_forward_epoch(scheduled_ops),
            .ops_tagged_for_chip_id_break = {},
            .ops_tagged_for_epoch_break = {},
            .fwd_to_bwd_nodes = {},
            .fwd_to_opt_nodes = {},
        };

        std::vector<Blocks> allocated_blocks;
        dram_config = std::make_unique<DramPlacerConfig>(
            device_config,
            test_cfg.input_queues_on_host,
            test_cfg.output_queues_on_host,
            test_cfg.manual_dram_queue_placemenet);
        allocator =
            std::make_unique<DramAllocator>(*dram_config, "unit_test_graph", 0, allocated_blocks, test_cfg.algo);

        graph = std::make_unique<Graph>(tt::graphlib::IRLevel::IR_BUDA);
    }

    std::pair<const Node *, QueueDRAMPlacementParameters &> add_e2e_queue(
        std::uint32_t grid_r,
        std::uint32_t grid_c,
        std::uint32_t producer_epoch = 0,
        std::uint32_t last_consumer_epoch = 0,
        QueueDRAMPlacementParameters::ConsumerMap consumer_loc = {},
        QueueDRAMPlacementParameters::ProducerMap producer_loc = {})
    {
        std::uint32_t node_number = 0;
        std::string node_name = "queue_" + std::to_string(node_number);
        while (graph->has_node_with_name(node_name))
        {
            node_number++;
            node_name = "queue_" + std::to_string(node_number);
        }

        auto *node =
            graph->add_node(tt::graphlib::create_node<tt::graphlib::EpochToEpochQueueNode>(node_name, true, false), 0);

        CoordRange queue_coord_range = {0, 0, grid_r, grid_c};

        bool is_input = false;
        bool in_p2p_region_soft = false;
        bool in_p2p_region_hard = false;
        tt::balancer::BlockShape block_shape = {1, 1, 1, tt::balancer::UBlockShape{1, 1}};

        std::string input_name = "foo";
        queue_placement_params.push_back(
            {QueuePlacement{
                 .name = node->name(),
                 .input_name = input_name,
                 .grid_shape = {queue_coord_range.size_r(), queue_coord_range.size_c()},
                 .on_host = false,
                 .chip_id = 0,
                 .dram_buffers = {},
                 .host_buffers = {},
                 .epoch_allocate = -1,
                 .epoch_deallocate = -1},
             QueueDRAMPlacementParameters{
                 .config = dram_config.get(),
                 .node = node,
                 .grid_shape = {queue_coord_range.size_r(), queue_coord_range.size_c()},
                 .consumer_loc = consumer_loc,
                 .producer_loc = producer_loc,
                 .block_shape = block_shape,
                 .producer_epoch = producer_epoch,
                 .last_consumer_epoch = last_consumer_epoch,
                 .in_p2p_region_soft = in_p2p_region_soft,
                 .in_p2p_region_hard = in_p2p_region_hard,
                 .is_input = is_input,
             }});

        return {node, queue_placement_params.back().second};
    }

    std::unordered_map<const Node *, std::vector<QueueBufferPlacement>> run_allocator()
    {
        allocator->allocate_queues(queue_placement_params, dram_config->disable_dynamic_dram);
        std::unordered_map<const Node *, std::vector<QueueBufferPlacement>> ret;
        for (auto &[queue_placement, queue_dram_placement_params] : queue_placement_params)
        {
            ret[queue_dram_placement_params.node] = queue_placement.dram_buffers;
        }
        return ret;
    }
};  // namespace test

TEST_P(DRAMPlacerTest, RoundRobin)
{
    SetUp(DRAMPlacementAlgorithm::ROUND_ROBIN);
    auto q1 = add_e2e_queue(5, 4);
    auto results = run_allocator();

    // Check results
    std::uint32_t expected_channel = 0;
    std::uint32_t expected_subchannel = 0;
    for (auto b : results.at(q1.first))
    {
        EXPECT_EQ(b.dram_channel, expected_channel);
        if (get_arch() == ARCH::WORMHOLE_B0)
        {
            // Each channel is "two in one"
            if (expected_subchannel == 0)
            {
                EXPECT_LT(b.dram_address, dram_config->dram_config[0].channel_size / 2);
                expected_subchannel = 1;
            }
            else
            {
                EXPECT_GE(b.dram_address, dram_config->dram_config[0].channel_size / 2);
                expected_subchannel = 0;
                expected_channel++;
            }
        }
        else
        {
            expected_channel++;
        }
        if (expected_channel >= device_config.get_dram_num_channels())
        {
            expected_channel = 0;
        }
    }
}

TEST_P(DRAMPlacerTest, RoundRobinFlipFlop)
{
    SetUp(DRAMPlacementAlgorithm::ROUND_ROBIN_FLIP_FLOP);
    auto q1 = add_e2e_queue(2, 4);  // group 0
    auto q2 = add_e2e_queue(2, 4);  // group 1
    q2.second.producer_epoch = 1;
    auto q3 = add_e2e_queue(2, 4);  // group 0
    auto results = run_allocator();

    // Check results
    std::vector<std::uint32_t> expected_channel = {0, device_config.get_dram_num_channels() / 2};
    std::vector<std::uint32_t> expected_subchannel = {0, 0};
    auto check_group = [&](int group, const std::vector<QueueBufferPlacement> &results)
    {
        for (auto b : results)
        {
            EXPECT_EQ(b.dram_channel, expected_channel[group]);
            if (get_arch() == ARCH::WORMHOLE_B0)
            {
                // Each channel is "two in one"
                if (expected_subchannel[group] == 0)
                {
                    EXPECT_LT(b.dram_address, dram_config->dram_config[0].channel_size / 2);
                    expected_subchannel[group] = 1;
                }
                else
                {
                    EXPECT_GE(b.dram_address, dram_config->dram_config[0].channel_size / 2);
                    expected_subchannel[group] = 0;
                    expected_channel[group]++;
                }
            }
            else
            {
                expected_channel[group]++;
            }
            if (group == 0)
            {
                if (expected_channel[group] >= device_config.get_dram_num_channels() / 2)
                {
                    expected_channel[group] = 0;
                }
            }
            else
            {
                if (expected_channel[group] >= device_config.get_dram_num_channels())
                {
                    expected_channel[group] = device_config.get_dram_num_channels() / 2;
                }
            }
        }
    };

    check_group(0, results.at(q1.first));
    check_group(1, results.at(q2.first));
    check_group(0, results.at(q3.first));
}

/*
 wormhole dram channels
     0 1 2 3 4 5 6 7 8 9
    +-+-+-+-+-+-+-+-+-+-+
  0 |0| | | | |2| | | | |
    +-+-+-+-+-+-+-+-+-+-+
  1 |0| | | | |2| | | | |
    +-+-+-+-+-+-+-+-+-+-+
  2 | | | | | |3| | | | |
    +-+-+-+-+-+-+-+-+-+-+
  3 | | | | | |4| | | | |
    +-+-+-+-+-+-+-+-+-+-+
  4 | | | | | |4| | | | |
    +-+-+-+-+-+-+-+-+-+-+
  5 |1| | | | |5| | | | |
    +-+-+-+-+-+-+-+-+-+-+
  6 |1| | | | |5| | | | |
    +-+-+-+-+-+-+-+-+-+-+
  7 |1| | | | |5| | | | |
    +-+-+-+-+-+-+-+-+-+-+
  8 | | | | | |4| | | | |
    +-+-+-+-+-+-+-+-+-+-+
  9 | | | | | |3| | | | |
    +-+-+-+-+-+-+-+-+-+-+
 10 | | | | | |3| | | | |
    +-+-+-+-+-+-+-+-+-+-+
 11 |0| | | | |2| | | | |
    +-+-+-+-+-+-+-+-+-+-+
*/

TEST_P(DRAMPlacerTest, Closest)
{
    if (get_arch() != ARCH::WORMHOLE_B0)
    {
        GTEST_SKIP();  // Focus on WH for now
    }
    SetUp(DRAMPlacementAlgorithm::CLOSEST);
    QueueDRAMPlacementParameters::ConsumerMap consumer_loc = {};
    QueueDRAMPlacementParameters::ProducerMap producer_loc = {};

    producer_loc[0][0] = {Coord{1, 1}, 0};
    consumer_loc[0][0].push_back({Coord{2, 2}, 1});
    auto q1 = add_e2e_queue(1, 1, 0, 1, consumer_loc, producer_loc);
    auto q2 = add_e2e_queue(
        1, 1, 0, 1, consumer_loc, producer_loc);  // same locations, same epoch, should pick the other subchannel

    producer_loc.clear();
    consumer_loc.clear();
    producer_loc[0][0] = {Coord{2, 9}, 0};
    producer_loc[0][1] = {Coord{8, 6}, 0};
    consumer_loc[0][0].push_back({Coord{8, 2}, 1});
    consumer_loc[0][0].push_back({Coord{8, 3}, 1});
    consumer_loc[0][1].push_back({Coord{8, 4}, 1});
    auto q3 = add_e2e_queue(1, 2, 0, 1, consumer_loc, producer_loc);

    auto results = run_allocator();

    EXPECT_EQ(results.at(q1.first)[0].dram_channel, 0);
    EXPECT_EQ(results.at(q2.first)[0].dram_channel, 0);
    EXPECT_GE(results.at(q2.first)[0].dram_address, dram_config->dram_config[0].channel_size / 2);
    EXPECT_EQ(results.at(q3.first)[0].dram_channel, 3);
    EXPECT_EQ(results.at(q3.first)[1].dram_channel, 4);
}

// Test without a producer core
TEST_P(DRAMPlacerTest, Closest_no_producer)
{
    if (get_arch() != ARCH::WORMHOLE_B0)
    {
        GTEST_SKIP();  // Focus on WH for now
    }
    SetUp(DRAMPlacementAlgorithm::CLOSEST);
    QueueDRAMPlacementParameters::ConsumerMap consumer_loc = {};
    QueueDRAMPlacementParameters::ProducerMap producer_loc = {};

    consumer_loc[0][0].push_back({Coord{1, 1}, 0});
    consumer_loc[0][1].push_back({Coord{1, 2}, 0});
    consumer_loc[0][2].push_back({Coord{1, 3}, 0});
    auto q1 = add_e2e_queue(1, 3, 0, 0, consumer_loc, producer_loc);

    auto results = run_allocator();

    EXPECT_EQ(results.at(q1.first)[0].dram_channel, 0);
}

//
// Tests that check that reader core is calculated correctly for various ops and grids
//

class ReaderCoreTest : public testing::TestWithParam<bool>
{
public:
    // Alias to make code more readable
    bool grid_transpose() { return GetParam(); }

    Coord position;
    GridShape op_dim, queue_dim;
    std::unique_ptr<tt::graphlib::BudaOpNode> test_op;
    OpPlacement placement;

    void SetUp(const std::string &op_type, GridShape op_dim_ = {2, 3}, GridShape queue_dim_ = {2, 3}, Coord position_ = {1, 5})
    {
        op_dim = op_dim_;
        queue_dim = queue_dim_;
        position = position_;

        test_op = tt::graphlib::create_node<tt::graphlib::BudaOpNode>("test_op", op_type);
        CoordRange placed_cores = {position, position + (grid_transpose() ? op_dim.transposed() : op_dim)};
        placement = {0, test_op->name(), 0, 0, grid_transpose(), placed_cores};
    }

    std::vector<Coord> calculate_readers(const Coord &dram_core, std::uint32_t operand = 0)
    {
        return get_reader_cores(test_op.get(), placement, operand, dram_core, queue_dim);
    }

};

//
// Test get_reader_cores function for various ops and placements
//

TEST_P(ReaderCoreTest, ReaderCores_EltwiseOnetoOne)
{
    // Eltwise, 2x3 reading 2x3 buffer
    SetUp("add", {2, 3}, {2, 3});

    for (std::uint32_t x = 0; x < queue_dim.columns; x++)
    {
        for (std::uint32_t y = 0; y < queue_dim.rows; y++)
        {
            Coord dram_core = {y, x};
            std::vector<Coord> reader_cores = calculate_readers(dram_core);
            Coord expected_offset = {grid_transpose() ? x : y, grid_transpose() ? y : x};
            EXPECT_EQ(reader_cores.size(), 1) << "Expect one reader core for each dram core";
            EXPECT_EQ(reader_cores.at(0), position + expected_offset) << "One to one mapping";
        }
    }
}

TEST_P(ReaderCoreTest, ReaderCores_EltwiseSingleBuffer)
{
    // Eltwise, 4x2 reading 1x1 buffer
    SetUp("add", {4, 2}, {1, 1});

    Coord dram_core = {0, 0};
    std::vector<Coord> reader_cores = calculate_readers(dram_core);
    EXPECT_EQ(reader_cores.size(), op_dim.volume()) << "1x1 buffer, expected all cores to be readers";
    std::uint32_t index = 0;
    for (std::uint32_t y = 0; y < op_dim.rows; y++)
    {
        for (std::uint32_t x = 0; x < op_dim.columns; x++)
        {
            Coord expected_offset = {grid_transpose() ? x : y, grid_transpose() ? y : x};
            EXPECT_EQ(reader_cores.at(index), position + expected_offset) << "1x1 buffer, op x=" << x << ", y=" << y;
            index++;
        }
    }

}

TEST_P(ReaderCoreTest, ReaderCores_MatmulOnetoOne)
{
    // Matmul, 2x3 reading 2x3 buffer
    SetUp("matmul", {2, 3}, {2, 3});

    for (std::uint32_t x = 0; x < queue_dim.columns; x++)
    {
        for (std::uint32_t y = 0; y < queue_dim.rows; y++)
        {
            Coord dram_core = {y, x};
            
            // Activations - first column (x=0) reads only
            {
                std::vector<Coord> reader_cores = calculate_readers(dram_core, 0);
                Coord expected_offset = {grid_transpose() ? 0 : y, grid_transpose() ? y : 0};
                EXPECT_EQ(reader_cores.size(), 1) << "Expect one reader core for each dram core";
                EXPECT_EQ(reader_cores.at(0), position + expected_offset);
            }

            // Weights - last row (y = op_dim.rows - 1) reads
            {
                std::vector<Coord> reader_cores = calculate_readers(dram_core, 1);
                Coord expected_offset = {grid_transpose() ? x : op_dim.rows - 1, grid_transpose() ? op_dim.rows - 1 : x};
                EXPECT_EQ(reader_cores.size(), 1) << "Expect one reader core for each dram core";
                EXPECT_EQ(reader_cores.at(0), position + expected_offset);
            }

        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    DRAMPlacerTests,
    DRAMPlacerTest,
    ::testing::Values(ARCH::WORMHOLE_B0, ARCH::GRAYSKULL),
    DRAMPlacerTest::PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(DRAMPlacerTests, ReaderCoreTest, ::testing::Values(false, true));

}  // namespace test
