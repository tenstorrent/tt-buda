// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <perf_model/graph.hpp>
#include <string>

#include "json.hpp"
using json = nlohmann::json;

//
// Structures for keeping track of perf trace data
//

namespace tt::perf_model
{

struct StallWait
{
    struct Stall
    {
        std::uint32_t input_index;
        std::uint32_t operand;
        std::vector<std::uint32_t> start, end;
        std::uint32_t num_tiles;
    };
    std::unordered_map<std::uint32_t, std::unordered_map<std::uint32_t, Stall>> stalls;

    void start_stall(
        std::uint32_t input_index, std::uint32_t operand, std::uint32_t timestamp, std::uint32_t num_tiles);
    void stop_stall(std::uint32_t input_index, std::uint32_t operand, std::uint32_t timestamp);

    void add_to_json(
        json &j,
        const std::vector<std::uint32_t> &input_indices,
        std::function<std::string(std::uint32_t, std::uint32_t, std::uint32_t)> c) const;
};

struct UnpackerTrace
{
    struct UnpackFirstInstruction
    {
        std::uint32_t input_index;
        std::uint32_t value = 0;
    };
    std::unordered_map<std::uint32_t, UnpackFirstInstruction> unpack_first_instruction;

    StallWait stall;

    json to_json(const std::vector<std::uint32_t> &input_indices) const;
};

struct MathTrace
{
    struct MathPerfCounter
    {
        std::uint32_t input_index;
        std::uint32_t activity;
        float utilization;
        std::uint32_t total_period;
    };
    std::unordered_map<std::uint32_t, MathPerfCounter> math_perf_counter;

    json to_json(const std::vector<std::uint32_t> &input_indices) const;
};

struct PackerTrace
{
    struct PackerOuterLoop
    {
        std::uint32_t input_index;
        std::uint32_t start = 0, end = 0;
    };
    std::unordered_map<std::uint32_t, PackerOuterLoop> packer_outer_loop;
    StallWait stall;

    json to_json(const std::vector<std::uint32_t> &input_indices) const;
};

struct PerThreadEvents
{
    struct Event
    {
        std::uint32_t input_index;
        std::uint32_t first_unpack_to_last_pack;
    };
    std::unordered_map<std::uint32_t, Event> events;
};

class TraceOp
{
    std::string name;
    std::string op_type;
    OpGrid grid;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-private-field"
    DataFormat pack_format;
#pragma GCC diagnostic pop
    std::vector<DataFormat> unpack_format;

    UnpackerTrace t0;
    MathTrace t1;
    PackerTrace t2;
    PerThreadEvents per_thread;

   public:
    TraceOp(
        std::string name,
        std::string op_type,
        OpGrid grid,
        DataFormat pack_format,
        std::vector<DataFormat> unpack_format) :
        name(name), op_type(op_type), grid(grid), pack_format(pack_format), unpack_format(unpack_format)
    {
    }

    void unpack_stall(
        std::uint32_t input_index, std::uint32_t operand, std::uint32_t timestamp, std::uint32_t num_tiles);
    void unpack_data_available(std::uint32_t input_index, std::uint32_t operand, std::uint32_t timestamp);

    void set_math_data(std::uint32_t input_index, std::uint32_t total_cycles, std::uint32_t useful_cycles);

    void pack_stall(std::uint32_t input_index, std::uint32_t operand, std::uint32_t timestamp, std::uint32_t num_tiles);
    void pack_started(std::uint32_t input_index, std::uint32_t operand, std::uint32_t timestamp);
    void pack_ended(std::uint32_t input_index, std::uint32_t timestamp);

    void add_to_json(json &j, const std::vector<std::uint32_t> &input_indices) const;
};

}  // namespace tt::perf_model
