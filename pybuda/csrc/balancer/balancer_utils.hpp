// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <iomanip>

#include "balancer/python_interface.hpp"
#include "balancer/types.hpp"
#include "graph_lib/graph.hpp"
#include "graph_lib/node.hpp"
#include "graph_lib/utils.hpp"
#include "placer/placer.hpp"
#include "types.hpp"
#include "utils/assert.hpp"

using Graph = tt::graphlib::Graph;
using Node = tt::graphlib::Node;
using Edge = tt::graphlib::Edge;
using DataFormat = tt::DataFormat;

namespace tt::balancer
{
inline std::size_t tile_size_bytes(DataFormat data_format, bool include_header_padding = true)
{
    std::size_t size = 0xbadface;
    std::size_t header_padding_size = include_header_padding ? 32 : 0;
    switch (data_format)
    {
        // clang-format off
        case DataFormat::Float32  : size = 32*32*4    + header_padding_size /* header and pad */ ; break;
        case DataFormat::Float16  : size = 32*32*2    + header_padding_size /* header and pad */ ; break;
        case DataFormat::Bfp8     : size = 32*32 + 64 + header_padding_size /* header and pad */ ; break;
        case DataFormat::Bfp4     : size = 512   + 64 + header_padding_size /* header and pad */ ; break;
        case DataFormat::Bfp2     : size = 256   + 64 + header_padding_size /* header and pad */ ; break;
        case DataFormat::Float16_b: size = 32*32*2    + header_padding_size /* header and pad */ ; break;
        case DataFormat::Bfp8_b   : size = 32*32 + 64 + header_padding_size /* header and pad */ ; break;
        case DataFormat::Bfp4_b   : size = 512   + 64 + header_padding_size /* header and pad */ ; break;
        case DataFormat::Bfp2_b   : size = 256   + 64 + header_padding_size /* header and pad */ ; break;
        case DataFormat::Lf8      : size = 32*32      + header_padding_size /* header and pad */ ; break;
        case DataFormat::UInt16   : size = 32*32*2    + header_padding_size /* header and pad */ ; break;
        case DataFormat::Int8     : size = 32*32      + header_padding_size /* header and pad */ ; break;
        case DataFormat::Int32    : size = 32*32*4    + header_padding_size /* header and pad */ ; break;
        case DataFormat::RawUInt8 : size = 32*32      + header_padding_size /* header and pad */ ; break;
        case DataFormat::RawUInt16: size = 32*32*2    + header_padding_size /* header and pad */ ; break;
        case DataFormat::RawUInt32: size = 32*32*4    + header_padding_size /* header and pad */ ; break;
        case DataFormat::Invalid  : size = 0xbadface ; break;
            // clang-format on
    }
    return size;
}

template <typename T>
T round_up_div(T n, T d)
{
    return (n + d - 1) / d;
}

inline std::size_t calculate_dst_size_tiles(
    std::size_t dst_size_bytes, DataFormat accumulate_df, int tile_volume, int num_buffers = 2)
{
    TT_ASSERT(is_valid_accumulate_df(accumulate_df));
    std::size_t available_dst_size = dst_size_bytes / num_buffers;  // half-dst for double-buffering
    std::size_t bytes_per_datum = data_format_byte_size(accumulate_df);
    TT_ASSERT(bytes_per_datum == 2 or bytes_per_datum == 4);
    std::size_t bytes_per_tile = tile_volume * bytes_per_datum;
    return round_up_div(available_dst_size, bytes_per_tile);
}

template <typename T>
std::string round_float(const T num, int precision)
{
    static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value, "Only floats and doubles allowed!");
    TT_ASSERT(!(precision < 0), "precision can't be negative!");

    std::stringstream ss;
    ss << std::fixed << std::setprecision(precision) << num;
    return ss.str();
}

template <typename T>
inline T gcd(T a, T b)
{
    T r = 1;
    for (T i = std::min(a, b); i > 0; --i)
    {
        if (((a % i) == 0) and ((b % i) == 0))
        {
            r = i;
            break;
        }
    }
    return r;
}

inline bool divisible_either_direction(int a, int b) { return ((a % b) == 0) || ((b % a) == 0); }

OpShape get_op_shape(
    Graph const *graph,
    Node const *node,
    GridShape grid_shape = {1, 1},
    int u_kt = 1,
    int u_rt = 1,
    TStreamFactor t_stream_factor = {},
    int fracture_factor = 1,
    bool calculate_sparse_in0_in2_shapes = false);
std::vector<tt::graphlib::OpType> calculate_undo_t_streaming_tms(
    Graph const *graph, Node const *node, OpModel const &op_model);
std::vector<tt::graphlib::OpType> calculate_t_streaming_tms(
    Graph const *graph, Node const *node, OpModel const &op_model);

std::pair<CanCoord, TensorShape> map_inverse_tms(
    CanCoord coord, TensorShape shape, std::vector<graphlib::OpType> const &tms);

int detect_repetitive_pattern(std::unordered_map<Pipe, int> *const kb_cache, Pipe const &pipe);

ResourceUsage get_edge_resource_usage(std::unordered_map<Pipe, ResourceUsage> &pipe_to_ru_cache, Pipe pipe);

// This path does a full tile order check to ensure that the pipe doesn't violate any HW constraints
ResourceUsage get_edge_resource_usage(
    Graph const *graph,
    std::unordered_map<Pipe, ResourceUsage> &pipe_to_ru_cache,
    graphlib::Edge edge,
    OpModel const &producer_op_model,
    OpModel const &consumer_op_model,
    bool is_queue = false);

// This path uses the old path, super simple heuristic based check that really only enforces some grid forking
// constraints
ResourceUsage get_edge_resource_usage_simple(
    Graph const *graph,
    graphlib::Edge edge,
    OpModel const &producer_op_model,
    OpModel const &consumer_op_model,
    bool is_queue = false);

std::shared_ptr<const OpModel::SparseMetadata> get_sparse_matmul_metadata(balancer::OpModel const &grid);

}  // namespace tt::balancer

namespace std
{
template <>
struct hash<tt::balancer::BlockShape>
{
    std::size_t operator()(const tt::balancer::BlockShape &block_shape) const
    {
        std::size_t seed = 0;
        tt::hash_combine(seed, static_cast<size_t>(block_shape.tblock_m));
        tt::hash_combine(seed, static_cast<size_t>(block_shape.tblock_n));
        tt::hash_combine(seed, static_cast<size_t>(block_shape.mblock_m));
        tt::hash_combine(seed, static_cast<size_t>(block_shape.mblock_m));
        tt::hash_combine(seed, static_cast<size_t>(block_shape.ublock.rt));
        tt::hash_combine(seed, static_cast<size_t>(block_shape.ublock.ct));
        return seed;
    }
};
}  // namespace std
