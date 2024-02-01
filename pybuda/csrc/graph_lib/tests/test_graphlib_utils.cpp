// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "graph_lib/node_types.hpp"
#include "graph_lib/utils.hpp"
#include "gtest/gtest.h"

using namespace tt::graphlib;

class GraphlibTest : public ::testing::Test
{
   public:
    void SetUp() override {}
    void TearDown() override {}
};

template <typename... Attrs>
OpType tm(std::string const& type, Attrs... attrs)
{
    return OpType(type, {attrs...});
}

TEST_F(GraphlibTest, kernel_broadcast_single_tile)
{
    Shape shape = Shape::create_buda({1, 1, 32, 32});
    std::vector<OpType> tms = {
        tm("broadcast", -1, 4),
        tm("broadcast", -2, 4),
    };

    for (int ublock_ct = 1; ublock_ct <= 2; ++ublock_ct)
    {
        EXPECT_TRUE(tms_support_kernel_broadcast(shape, tms, UBlockOrder::R, ublock_ct));
        EXPECT_TRUE(tms_support_kernel_broadcast(shape, tms, UBlockOrder::C, ublock_ct));
    }
}

TEST_F(GraphlibTest, kernel_broadcast_z)
{
    Shape shape = Shape::create_buda({1, 2, 32, 32});
    std::vector<OpType> tms = {
        tm("broadcast", -1, 4),
        tm("broadcast", -2, 4),
    };

    for (int ublock_ct = 1; ublock_ct <= 2; ++ublock_ct)
    {
        EXPECT_FALSE(tms_support_kernel_broadcast(shape, tms, UBlockOrder::R, ublock_ct));
        EXPECT_FALSE(tms_support_kernel_broadcast(shape, tms, UBlockOrder::C, ublock_ct));
    }
}

TEST_F(GraphlibTest, kernel_broadcast_bcast)
{
    for (int dim : std::vector{-1, -2})
    {
        Shape shape = (dim == -1) ? Shape::create_buda({1, 1, 64, 32}) : Shape::create_buda({1, 1, 32, 64});
        std::vector<OpType> tms = {
            tm("broadcast", dim, 4),
        };

        if (dim == -1)
        {
            int ublock_ct = 1;
            EXPECT_FALSE(tms_support_kernel_broadcast(shape, tms, UBlockOrder::R, ublock_ct));
            EXPECT_TRUE(tms_support_kernel_broadcast(shape, tms, UBlockOrder::C, ublock_ct));
            ublock_ct = 2;
            EXPECT_FALSE(tms_support_kernel_broadcast(shape, tms, UBlockOrder::R, ublock_ct));
            EXPECT_FALSE(tms_support_kernel_broadcast(shape, tms, UBlockOrder::C, ublock_ct));
        }
        else
        {
            int ublock_ct = 1;
            EXPECT_TRUE(tms_support_kernel_broadcast(shape, tms, UBlockOrder::R, ublock_ct));
            EXPECT_FALSE(tms_support_kernel_broadcast(shape, tms, UBlockOrder::C, ublock_ct));
            ublock_ct = 2;
            EXPECT_TRUE(tms_support_kernel_broadcast(shape, tms, UBlockOrder::R, ublock_ct));
            EXPECT_FALSE(tms_support_kernel_broadcast(shape, tms, UBlockOrder::C, ublock_ct));
        }
    }
}

TEST_F(GraphlibTest, kernel_broadcast_slice)
{
    for (auto slice : std::vector{"vslice", "hslice"})
    {
        bool v = slice[0] == 'v';
        Shape shape = v ? Shape::create_buda({1, 1, 32, 64}) : Shape::create_buda({1, 1, 64, 32});
        std::vector<OpType> tms = {
            tm("broadcast", v ? -2 : -1, 4),
            tm(slice, 2),
        };

        if (v)
        {
            int ublock_ct = 1;
            EXPECT_TRUE(tms_support_kernel_broadcast(shape, tms, UBlockOrder::R, ublock_ct));
            EXPECT_FALSE(tms_support_kernel_broadcast(shape, tms, UBlockOrder::C, ublock_ct));
            ublock_ct = 2;
            EXPECT_TRUE(tms_support_kernel_broadcast(shape, tms, UBlockOrder::R, ublock_ct));
            EXPECT_FALSE(tms_support_kernel_broadcast(shape, tms, UBlockOrder::C, ublock_ct));
        }
        else
        {
            int ublock_ct = 1;
            EXPECT_FALSE(tms_support_kernel_broadcast(shape, tms, UBlockOrder::R, ublock_ct));
            EXPECT_TRUE(tms_support_kernel_broadcast(shape, tms, UBlockOrder::C, ublock_ct));
            ublock_ct = 2;
            EXPECT_FALSE(tms_support_kernel_broadcast(shape, tms, UBlockOrder::R, ublock_ct));
            EXPECT_FALSE(tms_support_kernel_broadcast(shape, tms, UBlockOrder::C, ublock_ct));
        }
    }
}

TEST_F(GraphlibTest, kernel_broadcast_stack)
{
    for (auto stack : std::vector{"vstack", "hstack"})
    {
        bool v = stack[0] == 'v';
        Shape shape = Shape::create_buda({1, 2, 32, 32});
        std::vector<OpType> tms = {
            tm(stack, 2),
            tm("broadcast", v ? -1 : -2, 4),
        };

        if (v)
        {
            int ublock_ct = 1;
            EXPECT_FALSE(tms_support_kernel_broadcast(shape, tms, UBlockOrder::R, ublock_ct));
            EXPECT_FALSE(tms_support_kernel_broadcast(shape, tms, UBlockOrder::C, ublock_ct));
            ublock_ct = 2;
            EXPECT_FALSE(tms_support_kernel_broadcast(shape, tms, UBlockOrder::R, ublock_ct));
            EXPECT_FALSE(tms_support_kernel_broadcast(shape, tms, UBlockOrder::C, ublock_ct));
        }
        else
        {
            int ublock_ct = 1;
            EXPECT_FALSE(tms_support_kernel_broadcast(shape, tms, UBlockOrder::R, ublock_ct));
            EXPECT_FALSE(tms_support_kernel_broadcast(shape, tms, UBlockOrder::C, ublock_ct));
            ublock_ct = 2;
            EXPECT_FALSE(tms_support_kernel_broadcast(shape, tms, UBlockOrder::R, ublock_ct));
            EXPECT_FALSE(tms_support_kernel_broadcast(shape, tms, UBlockOrder::C, ublock_ct));
        }
    }
}
