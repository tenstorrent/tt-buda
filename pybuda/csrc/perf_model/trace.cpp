// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "perf_model/trace.hpp"

namespace tt::perf_model
{
void StallWait::start_stall(
    std::uint32_t input_index, std::uint32_t operand, std::uint32_t timestamp, std::uint32_t num_tiles)
{
    auto &w = stalls[input_index][operand];
    if (w.start.size() > w.end.size())
        return;  // repeated stall, ignore until unpack data is available

    if (w.start.size() == 0)
    {
        // first stall
        w.input_index = input_index;
        w.operand = operand;
        w.num_tiles = num_tiles;
    }

    w.start.push_back(timestamp);
}

void StallWait::stop_stall(std::uint32_t input_index, std::uint32_t operand, std::uint32_t timestamp)
{
    auto &w = stalls[input_index][operand];
    if (w.start.size() == w.end.size())
        return;  // no stall

    w.end.push_back(timestamp);
}

void TraceOp::unpack_stall(
    std::uint32_t input_index, std::uint32_t operand, std::uint32_t timestamp, std::uint32_t num_tiles)
{
    t0.stall.start_stall(input_index, operand, timestamp, num_tiles);
}

void TraceOp::unpack_data_available(std::uint32_t input_index, std::uint32_t operand, std::uint32_t timestamp)
{
    auto &uf = t0.unpack_first_instruction[input_index];
    if (uf.value == 0)
        uf.value = timestamp;

    t0.stall.stop_stall(input_index, operand, timestamp);
}

void TraceOp::set_math_data(std::uint32_t input_index, std::uint32_t total_cycles, std::uint32_t useful_cycles)
{
    t1.math_perf_counter[input_index] = MathTrace::MathPerfCounter{
        .input_index = input_index,
        .activity = useful_cycles,
        .utilization = (float)(1.0 * useful_cycles / total_cycles),
        .total_period = total_cycles};
}

void TraceOp::pack_stall(
    std::uint32_t input_index, std::uint32_t operand, std::uint32_t timestamp, std::uint32_t num_tiles)
{
    t2.stall.start_stall(input_index, operand, timestamp, num_tiles);
}

void TraceOp::pack_started(std::uint32_t input_index, std::uint32_t operand, std::uint32_t timestamp)
{
    auto &p = t2.packer_outer_loop[input_index];
    if (p.start == 0)
        p.start = timestamp;

    t2.stall.stop_stall(input_index, operand, timestamp);
}

void TraceOp::pack_ended(std::uint32_t input_index, std::uint32_t timestamp)
{
    auto &p = t2.packer_outer_loop[input_index];
    p.end = timestamp;
}
    
void TraceOp::add_to_json(json &j, const std::vector<std::uint32_t> &input_indices) const
{
    auto &op = j[std::to_string(grid.loc_c) + "-" + std::to_string(grid.loc_r) + "-" + name];
    op["NCRISC"] = {};
    op["T0"] = t0.to_json(input_indices);
    op["T1"] = t1.to_json(input_indices);
    op["T2"] = t2.to_json(input_indices);

    op["inputs-common-events"] = {
        {"op-type", op_type},
        {"pack-dst-data-format", 6},
        {"unpack-src-data-format-op0", 6},
        {"unpack-src-data-format-op1", 6}};

    for (std::uint32_t input_index : input_indices)
    {
        std::uint32_t unpack_start = t0.unpack_first_instruction.at(input_index).value;
        std::uint32_t pack_end = t2.packer_outer_loop.at(input_index).end;
        std::uint32_t pack_start = t2.packer_outer_loop.at(input_index).start;
        op["per-thread-events"]["input-" + std::to_string(input_index)] = {

            {"first-unpack-to-last-pack", pack_end - unpack_start},
            {"first-unpack-to-last-pack-without-wait-tile", "N/A"},
            {"math-runtime", pack_end - unpack_start},
            {"math-utilization-first-unpack-to-last-pack", 0.33},
            {"math-utilization-first-unpack-to-last-pack-without-wait-tile", "N/A"},
            {"math-utilization-over-math-thread", 0.1},
            {"pack-end-outer-loop", pack_end},
            {"pack-runtime", pack_end - pack_start},
            {"pack-start-outer-loop", pack_start},
            {"total-unpack-wait-for-tile-after-first-unpack", 0},
            {"total-wait-for-free-tile-after-first-unpack", 0},
            {"unpack-first-block-data-available", t0.unpack_first_instruction.at(input_index).value}
        };
    }
}

json UnpackerTrace::to_json(const std::vector<std::uint32_t> &input_indices) const
{
    json ret;
    ret["out-of-memory"] = "false";
    for (auto input_index : input_indices)
    {
        ret["unpack-first-instruction-outer-loop-" + std::to_string(input_index)]["value"] =
            std::vector<std::uint32_t>(1, unpack_first_instruction.at(input_index).value);

        auto name = [](std::uint32_t input_index, std::uint32_t operand, std::uint32_t num_tiles) -> std::string
        {
            return "wait-for-incoming-tiles-outer-loop-" + std::to_string(input_index) + "-operand-" +
                   std::to_string(operand) + "-num-tiles-" + std::to_string(num_tiles);
        };

        stall.add_to_json(ret, input_indices, name);
    }

    return ret;
}

json PackerTrace::to_json(const std::vector<std::uint32_t> &input_indices) const
{
    json ret;
    ret["out-of-memory"] = "false";
    for (auto input_index : input_indices)
    {
        auto &p = ret["packer-each-input-outer-loop-" + std::to_string(input_index)];
        auto &d = packer_outer_loop.at(input_index);
        p["start"] = std::vector<std::uint32_t>(1, d.start);
        p["end"] = std::vector<std::uint32_t>(1, d.end);
        p["diff"] = std::vector<std::uint32_t>(1, d.end - d.start);

        auto name = [](std::uint32_t input_index, std::uint32_t operand, std::uint32_t num_tiles) -> std::string
        {
            return "wait-for-free-tiles-outer-loop-" + std::to_string(input_index) + "-operand-" +
                   std::to_string(operand) + "-num-tiles-" + std::to_string(num_tiles);
        };

        stall.add_to_json(ret, input_indices, name);
    }

    return ret;
}

json MathTrace::to_json(const std::vector<std::uint32_t> &input_indices) const
{
    json ret;
    for (auto input_index : input_indices)
    {
        auto &m = ret["math-perf-counter-outer-loop-" + std::to_string(input_index)];
        m["math-activity"] = 100;
        m["total-period"] = 100;
        m["math-utilization"] = 0.99;
    }
    return ret;
}

void StallWait::add_to_json(
    json &j,
    const std::vector<std::uint32_t> &input_indices,
    std::function<std::string(std::uint32_t, std::uint32_t, std::uint32_t)> c) const
{
    for (auto input_index : input_indices)
    {
        if (stalls.size() <= input_index) continue;
        for (auto &[operand, s] : stalls.at(input_index))
        {
            auto &w = j[c(input_index, operand, s.num_tiles)];
            std::vector<std::uint32_t> diff;
            for (std::size_t i = 0; i < s.end.size(); i++) diff.push_back(s.end[i] - s.start[i]);
            w["start"] = s.start;
            w["end"] = s.end;
            w["diff"] = diff;
        }
    }
}

};  // namespace tt::perf_model
