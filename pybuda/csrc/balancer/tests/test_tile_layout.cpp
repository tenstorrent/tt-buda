// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include <random>
#include <unordered_map>

#include "balancer/balancer_utils.hpp"
#include "balancer/types.hpp"
#include "graph_lib/utils.hpp"
#include "gtest/gtest.h"
#include "third_party/budabackend/src/net2pipe/inc/tile_maps.h"
#include "utils/logger.hpp"

namespace tt::test
{
using namespace balancer;
using graphlib::UBlockOrder;

static std::vector<int> factorize(int begin, int f)
{
    std::vector<int> factors;
    for (int i = begin; i <= f; ++i)
        if (i % f == 0)
            factors.push_back(i);
    return factors;
}

static std::vector<int> factorize(int f) { return factorize(1, f); }

template <typename... Attrs>
graphlib::OpType tm(std::string const& type, Attrs... attrs)
{
    return graphlib::OpType(type, {attrs...}, {});
}

template <typename... Attrs>
graphlib::OpType vslice(Attrs... attrs)
{
    return tm("vslice", attrs...);
}

template <typename... Attrs>
graphlib::OpType hslice(Attrs... attrs)
{
    return tm("hslice", attrs...);
}

template <typename... Attrs>
graphlib::OpType vstack(Attrs... attrs)
{
    return tm("vstack", attrs...);
}

template <typename... Attrs>
graphlib::OpType hstack(Attrs... attrs)
{
    return tm("hstack", attrs...);
}

template <typename... Attrs>
graphlib::OpType broadcast(Attrs... attrs)
{
    return tm("broadcast", attrs...);
}

template <typename... Attrs>
graphlib::OpType tile_broadcast(Attrs... attrs)
{
    return tm("tile_broadcast", attrs...);
}

inline graphlib::OpType transpose()
{
    return graphlib::OpType("transpose", {}, {}, {{"dim0", 2}, {"dim1", 3}, {"z_dim_slice", -1}});
}

template <typename... Attrs>
graphlib::OpType buda_pad(Attrs... attrs)
{
    return tm("buda_pad", attrs...);
}

template <typename... Attrs>
graphlib::OpType buda_unpad(Attrs... attrs)
{
    return tm("buda_unpad", attrs...);
}

static std::tuple<int, int, int> get_net2pipe_resource_usage(
    GridShape producer_grid_shape,
    BlockShape producer_block_shape,
    graphlib::UBlockOrder producer_ublock_order,
    int producer_out_buf_mb,
    std::vector<graphlib::OpType> const& tms,
    std::string const& consumer_op_type,
    int consumer_input_port_id,
    GridShape consumer_grid_shape,
    BlockShape consumer_block_shape,
    graphlib::UBlockOrder consumer_ublock_order,
    std::string const& producer_name = "producer",
    std::string const& consumer_name = "consumer")
{
    three_d_array_tile_src_map tile_map(
        producer_name,
        consumer_name,
        producer_block_shape.t,
        producer_block_shape.ublock.rt,
        producer_block_shape.ublock.ct,
        producer_block_shape.mblock_m,
        producer_block_shape.mblock_n,
        producer_grid_shape.r,
        producer_grid_shape.c,
        producer_out_buf_mb / 2,
        producer_ublock_order == graphlib::UBlockOrder::R);

    for (graphlib::OpType const& tm : tms)
    {
        if (tm.op == "tile_broadcast")
            continue;

        if (tm.op == "broadcast")
        {
            int dim = std::get<int>(tm.attr[0]);
            std::string dims[] = {"w", "z", "r", "c"};
            tile_map = tile_map.apply_tm(dims[dim] + "_" + tm.op, {std::get<int>(tm.attr[1])});
        }
        else if (tm.op == "buda_pad")
        {
            int rt = std::get<int>(tm.attr[0]);
            int ct = std::get<int>(tm.attr[1]);
            tile_map = tile_map.pad(rt, ct);
        }
        else if (tm.op == "buda_unpad")
        {
            int rt = std::get<int>(tm.attr[0]);
            int ct = std::get<int>(tm.attr[1]);
            tile_map = tile_map.unpad(rt, ct);
        }
        else if (tm.op == "transpose")
        {
            tile_map = tile_map.apply_tm(tm.op, {});
        }
        else
        {
            tile_map = tile_map.apply_tm(tm.op, {std::get<int>(tm.attr[0])});
        }
    }

    constexpr int kernel_broadcast_tiles = 0;

    consumer_to_producer_tile_map edge_tile_map;
    if (consumer_op_type == "matmul")
    {
        if (consumer_input_port_id == 0)
        {
            edge_tile_map = tile_map.get_op_matmul_row_input(
                kernel_broadcast_tiles,
                false /*kernel_bcast_tiles_per_t*/,
                consumer_block_shape.t,
                consumer_block_shape.ublock.rt,
                consumer_block_shape.ublock.ct,
                consumer_block_shape.mblock_m,
                consumer_block_shape.mblock_n,
                consumer_grid_shape.r,
                consumer_grid_shape.c);
        }
        else
        {
            edge_tile_map = tile_map.get_op_matmul_col_input(
                kernel_broadcast_tiles,
                false /*kernel_bcast_tiles_per_t*/,
                consumer_block_shape.t,
                consumer_block_shape.ublock.rt,
                consumer_block_shape.ublock.ct,
                consumer_block_shape.mblock_m,
                consumer_block_shape.mblock_n,
                consumer_grid_shape.r,
                consumer_grid_shape.c);
        }
    }
    else
    {
        edge_tile_map = tile_map.get_op_eltwise_input(
            kernel_broadcast_tiles,
            false /*kernel_bcast_tiles_per_t*/,
            consumer_block_shape.t,
            consumer_block_shape.ublock.rt,
            consumer_block_shape.ublock.ct,
            consumer_block_shape.mblock_m,
            consumer_block_shape.mblock_n,
            consumer_grid_shape.r,
            consumer_grid_shape.c,
            consumer_ublock_order == graphlib::UBlockOrder::R);
    }

    return std::make_tuple(
        edge_tile_map.max_producer_core_fan_out(),
        edge_tile_map.max_producer_core_phases(),
        edge_tile_map.max_consumer_core_phases());
}

void test_inverse(CanCoord c0, TensorShape s0, std::vector<graphlib::OpType> const& tms)
{
    auto [c1, s1] = map_inverse_tms(c0, s0, tms);
    ASSERT_EQ(c1, c0);
    ASSERT_EQ(s1, s0);
}

TEST(TileLayoutTest, test_tm_inverses)
{
    std::vector<std::pair<std::string, std::string>> inverses = {
        {"vslice", "vstack"},
        {"vstack", "vslice"},
        {"hslice", "hstack"},
        {"hstack", "hslice"},
        {"transpose", "transpose"},
    };

    int max_t = 8;
    int max_r = 8;
    int max_c = 8;

    for (auto [from, to] : inverses)
        for (int tdim = 1; tdim <= max_t; ++tdim)
            for (int rdim = 1; rdim <= max_r; ++rdim)
                for (int cdim = 1; cdim <= max_c; ++cdim)
                    for (int t = 0; t < tdim; ++t)
                        for (int r = 0; r < rdim; ++r)
                            for (int c = 0; c < cdim; ++c)
                            {
                                TensorShape shape(1, tdim, rdim, cdim);
                                CanCoord coord(t, r, c);
                                if (from == "transpose")
                                {
                                    test_inverse(coord, shape, {transpose(), transpose()});
                                }
                                else
                                {
                                    int dim = (from == "vstack" or from == "hstack") ? tdim
                                              : (from == "vslice")                   ? rdim
                                                                                     : cdim;
                                    for (int f : factorize(dim))
                                    {
                                        test_inverse(coord, shape, {tm(from, f), tm(to, f)});
                                    }
                                }
                            }
}

TEST(TileLayoutTest, test_tile_layout)
{
    int max_t = 16;
    int max_r = 16;
    int max_c = 16;
    for (int tdim = 1; tdim <= max_t; ++tdim)
        for (int rdim = 1; rdim <= max_r; ++rdim)
            for (int cdim = 1; cdim <= max_c; ++cdim)
                for (int grid_r : factorize(rdim))
                    for (int grid_c : factorize(cdim))
                        for (int ublock_r : factorize(rdim / grid_r))
                            for (int ublock_c : factorize(cdim / grid_c))
                                for (auto ublock_order : std::vector<UBlockOrder>{UBlockOrder::R, UBlockOrder::C})
                                {
                                    ASSERT_EQ(rdim % (grid_r * ublock_r), 0);
                                    ASSERT_EQ(cdim % (grid_c * ublock_c), 0);
                                    int m = rdim / (grid_r * ublock_r);
                                    int n = cdim / (grid_c * ublock_c);
                                    TileLayout layout(
                                        GridShape(grid_r, grid_c),
                                        BlockShape(tdim, m, n, UBlockShape(ublock_r, ublock_c)),
                                        ublock_order);
                                    for (int t = 0; t < tdim; ++t)
                                        for (int r = 0; r < rdim; ++r)
                                            for (int c = 0; c < cdim; ++c)
                                            {
                                                CanCoord coord(t, r, c);
                                                LinCoord lin = layout.map(coord);
                                                CanCoord coord1 = layout.map(lin);
                                                ASSERT_EQ(coord, coord1);
                                            }
                                }
}

template <typename T>
T sample(std::mt19937& gen, std::vector<T> v, int start = 0)
{
    TT_ASSERT(not v.empty());
    if ((int)v.size() <= (start + 1))
        return v.front();
    std::uniform_int_distribution<> distrib(start, v.size() - 1);
    return v[distrib(gen)];
}

static int randint(std::mt19937& gen, int from, int to)
{
    std::uniform_int_distribution<> distrib(from, to);
    return distrib(gen);
}

static std::vector<std::string> rand_tm_types(std::mt19937& gen, int num_samples)
{
    static std::vector<std::string> types = {
        "broadcast",
        "vslice",
        "hslice",
        "vstack",
        "hstack",
        "transpose",
    };

    std::vector<std::string> selected;
    selected.reserve(num_samples);
    for (int i = 0; i < num_samples; ++i) selected.push_back(sample(gen, types));
    return selected;
}

std::vector<graphlib::OpType> rand_tms(std::mt19937& gen, TensorShape shape, int num_samples)
{
    int max_bcast = 6;
    auto types = rand_tm_types(gen, num_samples);
    std::vector<graphlib::OpType> tms;
    tms.reserve(types.size());
    for (auto type : types)
    {
        if (type == "broadcast")
        {
            int dim = randint(gen, 1, 3);
            int factor = randint(gen, 2, max_bcast);
            if (dim == 1)
                shape.z *= factor;
            else if (dim == 2)
                shape.rt *= factor;
            else if (dim == 3)
                shape.ct *= factor;
            if (factor > 1)
                tms.push_back(tm(type, dim, factor));
        }
        else if (type == "vslice")
        {
            int factor = sample(gen, factorize(shape.rt), 2);
            TT_ASSERT(shape.rt % factor == 0);
            shape.z *= factor;
            shape.rt /= factor;
            if (factor > 1)
                tms.push_back(tm(type, factor));
        }
        else if (type == "hslice")
        {
            int factor = sample(gen, factorize(shape.ct), 2);
            TT_ASSERT(shape.ct % factor == 0);
            shape.z *= factor;
            shape.ct /= factor;
            if (factor > 1)
                tms.push_back(tm(type, factor));
        }
        else if (type == "vstack")
        {
            int factor = sample(gen, factorize(shape.z), 2);
            TT_ASSERT(shape.z % factor == 0);
            shape.rt *= factor;
            shape.z /= factor;
            if (factor > 1)
                tms.push_back(tm(type, factor));
        }
        else if (type == "hstack")
        {
            int factor = sample(gen, factorize(shape.z), 2);
            TT_ASSERT(shape.z % factor == 0);
            shape.ct *= factor;
            shape.z /= factor;
            if (factor > 1)
                tms.push_back(tm(type, factor));
        }
        else if (type == "transpose")
        {
            std::swap(shape.rt, shape.ct);
            tms.push_back(transpose());
        }
    }
    return tms;
}

static graphlib::Shape to_shape(TensorShape shape)
{
    return graphlib::Shape::create_buda(
        shape.w, shape.z, shape.rt * graphlib::Shape::BUDA_TILE_DIM, shape.ct * graphlib::Shape::BUDA_TILE_DIM);
}

TileLayout random_layout(std::mt19937& gen, TensorShape shape)
{
    int grid_r = sample(gen, factorize(shape.rt));
    int grid_c = sample(gen, factorize(shape.ct));
    int ublock_r = sample(gen, factorize(shape.rt / grid_r));
    int ublock_c = sample(gen, factorize(shape.ct / grid_c));
    int m = shape.rt / (grid_r * ublock_r);
    int n = shape.ct / (grid_c * ublock_c);
    auto ublock_order = sample(gen, std::vector<UBlockOrder>{UBlockOrder::R, UBlockOrder::C});
    return TileLayout(
        GridShape(grid_r, grid_c), BlockShape(shape.z, m, n, UBlockShape(ublock_r, ublock_c)), ublock_order);
}

TEST(TileLayoutTest, test_tile_layout_random)
{
    static std::mt19937 gen;

    int start_seed = 0;
    int num_tests = 1024;
    int max_t = 16;
    int max_r = 16;
    int max_c = 16;
    int max_tms = 4;

    for (int seed = start_seed; seed < (num_tests + start_seed); ++seed)
    {
        gen.seed(seed);

        int tdim = randint(gen, 1, max_t);
        int rdim = randint(gen, 1, max_r);
        int cdim = randint(gen, 1, max_c);
        TensorShape producer_shape(1, tdim, rdim, cdim);
        auto tms = rand_tms(gen, producer_shape, randint(gen, 0, max_tms));
        TensorShape consumer_shape = graphlib::post_tms_shape(to_shape(producer_shape), tms);

        TileLayout producer_layout = random_layout(gen, producer_shape);
        TileLayout consumer_layout = random_layout(gen, consumer_shape);

        for (int grid_r = 0; grid_r < consumer_layout.grid_shape.r; ++grid_r)
        {
            for (int grid_c = 0; grid_c < consumer_layout.grid_shape.c; ++grid_c)
            {
                int max_core_t = 0;
                for (int address = 0; address < consumer_layout.block_shape.volume(); ++address)
                {
                    LinCoord consumer_linear(grid_r, grid_c, address);
                    CanCoord consumer_coord = consumer_layout.map(consumer_linear);
                    auto [producer_coord, p_shape] = map_inverse_tms(consumer_coord, consumer_shape, tms);
                    EXPECT_EQ(p_shape, producer_shape);
                    LinCoord producer_linear = producer_layout.map(producer_coord);
                    EXPECT_GE(producer_linear.address(), 0);

                    if (address < consumer_layout.block_shape.volume_no_t())
                        max_core_t = std::max(max_core_t, producer_coord.t);
                }
            }
        }
    }
}

#if 0
TEST(TileLayoutTest, test_tile_layout_perf)
{
    static std::mt19937 gen;

    int start_seed = 0;
    int num_tests = 4098;
    int max_t = 16;
    int max_r = 64;
    int max_c = 64;
    int max_tms = 4;

    volatile int* ptr = (volatile int*)malloc(sizeof(int));
    for (int seed = start_seed; seed < (num_tests + start_seed); ++seed)
    {
        gen.seed(seed);

        int tdim = randint(gen, 1, max_t);
        int rdim = randint(gen, 1, max_r);
        int cdim = randint(gen, 1, max_c);
        TensorShape producer_shape(1, tdim, rdim, cdim);
        auto tms = rand_tms(gen, producer_shape, randint(gen, 0, max_tms));
        TensorShape consumer_shape = graphlib::post_tms_shape(to_shape(producer_shape), tms);

        TileLayout producer_layout = random_layout(gen, producer_shape);
        TileLayout consumer_layout = random_layout(gen, consumer_shape);
        Pipe pipe(producer_layout, 2, tms, consumer_layout);
        auto resource_usage = get_edge_resource_usage(pipe);
        *ptr = resource_usage.producer_fan_out;
        *ptr = resource_usage.consumer_fan_in;
        *ptr = resource_usage.producer_phases;
        *ptr = resource_usage.consumer_phases;
    }
}
#endif

class PipeTest : public ::testing::TestWithParam<Pipe>
{
};

TEST_P(PipeTest, test_tile_layout_targeted)
{
    std::unordered_map<Pipe, ResourceUsage> dummy_cache;
    auto pipe = GetParam();
    ResourceUsage usage = get_edge_resource_usage(
        dummy_cache, Pipe(pipe.producer_layout, pipe.producer_out_buf_mb, pipe.tms, pipe.consumer_layout));

    auto [n2p_producer_core_fan_out, n2p_producer_core_phases, n2p_consumer_core_phases] = get_net2pipe_resource_usage(
        pipe.producer_layout.grid_shape,
        pipe.producer_layout.block_shape,
        pipe.producer_layout.ublock_order,
        pipe.producer_out_buf_mb,
        pipe.tms,
        "eltwise",
        0,
        pipe.consumer_layout.grid_shape,
        pipe.consumer_layout.block_shape,
        pipe.consumer_layout.ublock_order);

    EXPECT_EQ(usage.producer_fan_out, n2p_producer_core_fan_out);
    // GE for now because we don't account for some special cases that can loop
    EXPECT_GE(usage.producer_phases, n2p_producer_core_phases);
    EXPECT_EQ(usage.consumer_phases, n2p_consumer_core_phases);

    log_debug(LogTest, "Test:");
    log_debug(LogTest, "  producer:");
    log_debug(LogTest, "    {}", pipe.producer_layout.grid_shape);
    log_debug(LogTest, "    {}", pipe.producer_layout.block_shape);
    log_debug(LogTest, "    {}", pipe.producer_layout.ublock_order);
    log_debug(LogTest, "  tms: {}", pipe.tms);
    log_debug(LogTest, "  consumer:");
    log_debug(LogTest, "    {}", pipe.consumer_layout.grid_shape);
    log_debug(LogTest, "    {}", pipe.consumer_layout.block_shape);
    log_debug(LogTest, "    {}", pipe.consumer_layout.ublock_order);
    log_debug(LogTest, "Result:");
    log_debug(LogTest, "  calculated: {} {} {}", usage.producer_fan_out, usage.producer_phases, usage.consumer_phases);
    log_debug(
        LogTest,
        "  net2pipe:   {} {} {}",
        n2p_producer_core_fan_out,
        n2p_producer_core_phases,
        n2p_consumer_core_phases);

    log_debug(LogTest, "  fan_in: {}", usage.consumer_fan_in);
}

INSTANTIATE_TEST_SUITE_P(
    TileLayoutTest,
    PipeTest,
    testing::Values(
        Pipe(
            TileLayout(GridShape(1, 1), BlockShape(1, 8, 8, UBlockShape(1, 1)), UBlockOrder::R),
            2,
            {},
            TileLayout(GridShape(1, 1), BlockShape(1, 8, 8, UBlockShape(1, 1)), UBlockOrder::R)),
        Pipe(
            TileLayout(GridShape(2, 2), BlockShape(1, 16, 16, UBlockShape(1, 1)), UBlockOrder::R),
            2,
            {},
            TileLayout(GridShape(1, 1), BlockShape(1, 32, 32, UBlockShape(1, 1)), UBlockOrder::R)),

        Pipe(
            TileLayout(GridShape(2, 1), BlockShape(2, 32, 1, UBlockShape(1, 1)), UBlockOrder::R),
            2,
            {tm("vstack", 2)},
            TileLayout(GridShape(2, 1), BlockShape(1, 64, 1, UBlockShape(1, 1)), UBlockOrder::R)),

        Pipe(
            TileLayout(GridShape(6, 2), BlockShape(1, 96, 1, UBlockShape(1, 1)), UBlockOrder::R),
            2,
            {tm("vslice", 12)},
            TileLayout(GridShape(1, 1), BlockShape(12, 48, 1, UBlockShape(1, 2)), UBlockOrder::R)),

        Pipe(
            TileLayout(GridShape(1, 1), BlockShape(1, 1, 1, UBlockShape(1, 1)), UBlockOrder::R),
            2,
            {},
            TileLayout(GridShape(1, 1), BlockShape(1, 1, 1, UBlockShape(1, 1)), UBlockOrder::R)),

        Pipe(
            TileLayout(GridShape(1, 1), BlockShape(1, 16, 16, UBlockShape(1, 1)), UBlockOrder::R),
            2,
            {},
            TileLayout(GridShape(1, 1), BlockShape(1, 16, 16, UBlockShape(1, 1)), UBlockOrder::C)),

        Pipe(
            TileLayout(GridShape(2, 2), BlockShape(14, 112, 1, UBlockShape(1, 1)), graphlib::UBlockOrder::R),
            2,
            {vslice(16), hstack(16)},
            TileLayout(GridShape(1, 1), BlockShape(14, 14, 2, UBlockShape(1, 16)), graphlib::UBlockOrder::C)),

        Pipe(
            TileLayout(GridShape(8, 8), BlockShape(1, 16, 1, UBlockShape(1, 4)), graphlib::UBlockOrder::R),
            2,
            {vslice(64)},
            TileLayout(GridShape(1, 1), BlockShape(64, 2, 8, UBlockShape(1, 4)), graphlib::UBlockOrder::C)),
        Pipe(
            TileLayout(GridShape(1, 4), BlockShape(1, 25, 1, UBlockShape(1, 1)), graphlib::UBlockOrder::R),
            2,
            {},
            TileLayout(GridShape(1, 1), BlockShape(1, 25, 2, UBlockShape(1, 2)), graphlib::UBlockOrder::C)),
        Pipe(
            TileLayout(GridShape(9, 2), BlockShape(1, 98, 1, UBlockShape(1, 1)), graphlib::UBlockOrder::R),
            2,
            {vslice(9), hstack(9)},
            TileLayout(GridShape(7, 1), BlockShape(1, 14, 18, UBlockShape(1, 1)), graphlib::UBlockOrder::C)),
        Pipe(
            TileLayout(GridShape(3, 4), BlockShape(1, 75, 1, UBlockShape(1, 1)), graphlib::UBlockOrder::R),
            2,
            {vslice(9), hstack(9)},
            TileLayout(GridShape(1, 1), BlockShape(1, 25, 1, UBlockShape(1, 36)), graphlib::UBlockOrder::C)),
        Pipe(
            TileLayout(GridShape(10, 2), BlockShape(16, 6, 1, UBlockShape(1, 1)), graphlib::UBlockOrder::R),
            2,
            {hstack(16)},
            TileLayout(GridShape(1, 1), BlockShape(1, 60, 16, UBlockShape(1, 2)), graphlib::UBlockOrder::C)),
        Pipe(
            TileLayout(GridShape(8, 2), BlockShape(1, 64, 1, UBlockShape(1, 1)), graphlib::UBlockOrder::R),
            2,
            {transpose(), hslice(512), vstack(512)},
            TileLayout(GridShape(1, 1), BlockShape(1, 64, 1, UBlockShape(16, 1)), graphlib::UBlockOrder::R)),
        Pipe(
            TileLayout(GridShape(4, 3), BlockShape(1, 32, 3, UBlockShape(1, 1)), graphlib::UBlockOrder::R),
            2,
            {hslice(9), vstack(9)},
            TileLayout(GridShape(1, 1), BlockShape(1, 24, 1, UBlockShape(48, 1)), graphlib::UBlockOrder::R)),
        Pipe(
            TileLayout(GridShape(1, 1), BlockShape(1, 5, 1, UBlockShape(2, 1)), graphlib::UBlockOrder::R),
            2,
            {broadcast(3, 128), vslice(5)},
            TileLayout(GridShape(1, 2), BlockShape(5, 1, 16, UBlockShape(2, 4)), graphlib::UBlockOrder::R)),
        // operand[0] layernorm_251.dc.add.14 -> matmul_256
        Pipe(
            TileLayout(GridShape(2, 5), BlockShape(1, 32, 1, UBlockShape(2, 2)), graphlib::UBlockOrder::C),
            2,
            {broadcast(2, 10), vslice(10)},
            TileLayout(GridShape(1, 1), BlockShape(10, 64, 1, UBlockShape(2, 10)), graphlib::UBlockOrder::C)),
        // operand[2] unet.down_blocks.0.attentions.0.transformer_blocks.0.ff.net.0.proj.bias_0 -> matmul_119
        Pipe(
            TileLayout(GridShape(1, 1), BlockShape(1, 1, 40, UBlockShape(1, 1)), graphlib::UBlockOrder::R),
            2,
            {broadcast(2, 128), hslice(40)},
            TileLayout(GridShape(1, 1), BlockShape(40, 64, 1, UBlockShape(2, 1)), graphlib::UBlockOrder::R)),
        // Pipe(
        //     TileLayout(GridShape(1, 1), BlockShape(12, 48, 1, UBlockShape(1, 2)), graphlib::UBlockOrder::R),
        //     2,
        //     {vstack(12)},
        //     TileLayout(GridShape(6, 2), BlockShape(1, 96, 1, UBlockShape(1, 1)), graphlib::UBlockOrder::R)),
        //  operand[0] dc.input_tensor.resize2d_204.2 -> resize2d_204.dc.matmul.3
        Pipe(
            TileLayout(GridShape(8, 1), BlockShape(1, 49, 98, UBlockShape(1, 1)), graphlib::UBlockOrder::C),
            2,
            {vslice(49)},
            TileLayout(GridShape(8, 1), BlockShape(49, 1, 98, UBlockShape(1, 1)), graphlib::UBlockOrder::C)),
        Pipe(
            TileLayout(GridShape(1, 1), BlockShape(4, 4, 1, UBlockShape(1, 1)), graphlib::UBlockOrder::R),
            2,
            {tile_broadcast(3), broadcast(3, 4, 0)},
            TileLayout(GridShape(1, 1), BlockShape(4, 4, 1, UBlockShape(1, 4)), graphlib::UBlockOrder::C)),
        Pipe(
            TileLayout(GridShape(4, 8), BlockShape(1, 4, 4, UBlockShape(1, 4)), graphlib::UBlockOrder::R),
            2,
            {transpose(), vslice(128), hslice(16), vstack(2048)},
            TileLayout(GridShape(1, 1), BlockShape(1, 64, 1, UBlockShape(32, 1)), graphlib::UBlockOrder::R)),
        Pipe(
            TileLayout(GridShape(1, 1), BlockShape(1, 1, 1, UBlockShape(1, 1)), graphlib::UBlockOrder::R),
            2,
            {broadcast(3, 2048, false)},
            TileLayout(GridShape(1, 1), BlockShape(1, 1, 2048, UBlockShape(1, 1)), graphlib::UBlockOrder::C)),

        Pipe(
            TileLayout(GridShape(1, 2), BlockShape(16, 6, 1, UBlockShape(2, 1)), graphlib::UBlockOrder::R),
            32,
            {hstack(16)},
            TileLayout(GridShape(1, 1), BlockShape(1, 2, 8, UBlockShape(6, 4)), graphlib::UBlockOrder::R)),

        // Padding
        Pipe(
            TileLayout(
                GridShape(1, 1), BlockShape(1, 10, 5, UBlockShape(1, 2)), graphlib::UBlockOrder::R, Padding(2, 2)),
            2,
            {buda_unpad(2, 2, 256, 256)},
            TileLayout(GridShape(1, 1), BlockShape(1, 8, 2, UBlockShape(1, 4)), graphlib::UBlockOrder::R)),
        Pipe(
            TileLayout(
                GridShape(1, 1), BlockShape(1, 10, 5, UBlockShape(1, 2)), graphlib::UBlockOrder::R, Padding(9, 9)),
            2,
            {buda_unpad(9, 9, 12, 12)},
            TileLayout(GridShape(1, 1), BlockShape(1, 1, 1, UBlockShape(1, 1)), graphlib::UBlockOrder::R))
        // Pipe(
        //     TileLayout(GridShape(1, 1), BlockShape(1, 1, 1, UBlockShape(1, 1)), graphlib::UBlockOrder::R),
        //     2,
        //     {buda_pad(9, 9, 12, 12)},
        //     TileLayout(
        //         GridShape(1, 1), BlockShape(1, 10, 5, UBlockShape(1, 2)), graphlib::UBlockOrder::R, Padding(9, 9)))
        ));

}  // namespace tt::test
