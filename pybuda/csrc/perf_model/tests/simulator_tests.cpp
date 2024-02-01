// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "gtest/gtest.h"
#include "perf_model/perf_model.hpp"
#include "perf_model/simulator.hpp"

using namespace tt::perf_model;

// Utility for making perf data structure with the correct size in tiles
PerfDataP perf_data(std::uint32_t out_size_in_tiles, std::vector<std::uint32_t> in_size_in_tiles, bool is_op)
{
    auto td = [](std::uint32_t size, std::uint32_t t = 1) {
        return TensorData{.shape = tt::graphlib::Shape::create({1, 1, 32, 32 * size}), .t = t};
    };

    std::vector<TensorData> inputs;
    for (auto in : in_size_in_tiles) inputs.push_back(td(in));

    if (is_op)
    {
        OpGrid grid1 = OpGrid{.loc_r = 0, .loc_c = 0, .size_r = 1, .size_c = 1};
        tt::balancer::OpModel op_model;
        return std::make_shared<PerfData>(
            PerfData(inputs, td(out_size_in_tiles), OpPerfData{.grid = grid1, .op_model=op_model}));
    }

    return std::make_shared<PerfData>(PerfData(inputs, td(out_size_in_tiles), QueuePerfData{}));
};

// Utility functions for quick graph creations
NodeP add_queue(
    tt::perf_model::Graph *graph,
    const std::string &name,
    bool is_input,
    NodeP operand = nullptr,
    std::uint32_t out_size_in_tiles = 1)
{
    auto q_type = is_input ? tt::graphlib::QueueNodeType::Input : tt::graphlib::QueueNodeType::Output;
    std::vector<std::uint32_t> in_size_in_tiles = {};
    if (operand != nullptr)
        in_size_in_tiles = std::vector<std::uint32_t>(1, out_size_in_tiles);
    return graph->add_queue(name, q_type, operand, perf_data(out_size_in_tiles, in_size_in_tiles, false), is_input);
}

NodeP add_op(
    tt::perf_model::Graph *graph,
    const std::string &name,
    const std::string &type,
    std::vector<NodeP> operands,
    std::uint32_t out_size_in_tiles = 1,
    std::vector<std::uint32_t> in_size_in_tiles = {})
{
    if (in_size_in_tiles.size() == 0)
        in_size_in_tiles = std::vector<std::uint32_t>(operands.size(), 1);
    return graph->add_op(name, type, operands, perf_data(out_size_in_tiles, in_size_in_tiles, true), false);
}

TEST(PerfModel, basic)
{
    tt::perf_model::Graph *graph = new tt::perf_model::Graph();

    auto in0 = add_queue(graph, "input0", true);
    auto in1 = add_queue(graph, "input1", true);
    auto op0 = add_op(graph, "op0", "add", {in0, in1});
    add_queue(graph, "output", false, op0);

    Simulator sim(graph, 3);
    EXPECT_TRUE(sim.run());
}

TEST(PerfModel, fork)
{
    tt::perf_model::Graph *graph = new tt::perf_model::Graph();

    auto in0 = add_queue(graph, "input0", true);
    auto in1 = add_queue(graph, "input1", true);
    auto op0 = add_op(graph, "op0", "exp", {in0});
    auto op1a = add_op(graph, "op1a", "add", {op0, in1});
    auto op1b = add_op(graph, "op1b", "sqrt", {op0});
    auto op2 = add_op(graph, "op1b", "mul", {op1a, op1b});
    add_queue(graph, "output", false, op2);

    Simulator sim(graph, 3);
    EXPECT_TRUE(sim.run());
}

TEST(PerfModel, t_streaming)
{
    tt::perf_model::Graph *graph = new tt::perf_model::Graph();

    auto in0 = add_queue(graph, "input0", true, nullptr, 4);
    auto in1 = add_queue(graph, "input1", true, nullptr, 4);
    auto op0 = add_op(graph, "op0", "add", {in0, in1}, 4, {4, 4});
    auto op1 = add_op(graph, "op1t", "exp", {op0}, 1, {1});
    op1->get_perf_data()->output.t = 4;
    add_queue(graph, "output", false, op1, 4);

    Simulator sim(graph, 3);
    EXPECT_TRUE(sim.run());
}

TEST(PerfModel, matmul)
{
    tt::perf_model::Graph *graph = new tt::perf_model::Graph();
    auto in0 = add_queue(graph, "input0", true, nullptr, 4);
    auto in1 = add_queue(graph, "input1", true, nullptr, 4);

    auto op0 = add_op(graph, "matmul0", "matmul", {in0, in1}, 1, {4, 4});
    op0->get_perf_data()->attr.m_k = 4;
    add_queue(graph, "output", false, op0, 1);

    Simulator sim(graph, 3);
    EXPECT_TRUE(sim.run());
}

TEST(PerfModel, fork_join_perf)
{
    // Fork 5 to 0 ops, show that perf gets better as we add more buffering

    auto create_graph = [](std::uint32_t buf_count) -> tt::perf_model::Graph *
    {
        tt::perf_model::Graph *graph = new tt::perf_model::Graph();

        auto in0 = add_queue(graph, "input0", true);
        auto op0 = add_op(graph, "op0", "exp", {in0});

        auto prev = op0;
        for (std::uint32_t b = 0; b < buf_count; b++)
        {
            auto buf = add_op(graph, "buf" + std::to_string(b), "nop", {prev});
            prev = buf;
        }

        auto f_prev = op0;
        for (std::uint32_t f = 0; f < 6; f++)
        {
            auto op = add_op(graph, "op_f" + std::to_string(f), "exp", {f_prev});
            f_prev = op;
        }

        auto join = add_op(graph, "op_join", "add", {prev, f_prev});
        add_queue(graph, "output", false, join);
        return graph;
    };

    std::vector<std::uint32_t> time;
    for (std::uint32_t i = 0; i < 3; i++)
    {
        Simulator sim(create_graph(i), 32);
        EXPECT_TRUE(sim.run());
        time.push_back(sim.get_timestamp());
    }

    // Perf should get better with buffering on the short side
    EXPECT_GT(time[0], time[1]);
    EXPECT_GT(time[1], time[2]);
}
