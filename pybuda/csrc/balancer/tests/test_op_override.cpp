// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "balancer/balancer_config.hpp"
#include "balancer/types.hpp"
#include "gtest/gtest.h"

namespace tt::test
{
using OpOverride = tt::balancer::OpOverride;

TEST(TestOpOverride, no_override1)
{
    // Test that default override doesn't change anything
    tt::balancer::FactorizedShape shape = std::make_pair(10, 16);
    bool force_dram_parameters_out = false;
    std::vector<balancer::TStreamDir> t_stream_dirs = {};
    tt::balancer::FactorizedShape streaming_pars = std::make_pair(8, 12);
    bool enable_t_streaming = false;

    OpOverride blank;
    blank.apply(shape, force_dram_parameters_out, t_stream_dirs, streaming_pars, enable_t_streaming, "foo");

    EXPECT_EQ(shape, tt::balancer::FactorizedShape(10, 16));
    EXPECT_EQ(streaming_pars, tt::balancer::FactorizedShape(8, 12));
    EXPECT_EQ(force_dram_parameters_out, false);
    EXPECT_EQ(t_stream_dirs.size(), 0);
    EXPECT_EQ(enable_t_streaming, false);
}

TEST(TestOpOverride, no_override2)
{
    // Test that default override doesn't change anything
    tt::balancer::FactorizedShape shape = std::make_pair(2, 4);
    bool force_dram_parameters_out = true;
    std::vector<balancer::TStreamDir> t_stream_dirs = {tt::balancer::TStreamDir::R};
    tt::balancer::FactorizedShape streaming_pars = std::make_pair(6, 1);
    bool enable_t_streaming = true;

    OpOverride blank;
    blank.apply(shape, force_dram_parameters_out, t_stream_dirs, streaming_pars, enable_t_streaming, "foo");

    EXPECT_EQ(shape, tt::balancer::FactorizedShape(2, 4));
    EXPECT_EQ(streaming_pars, tt::balancer::FactorizedShape(6, 1));
    EXPECT_EQ(force_dram_parameters_out, true);
    EXPECT_EQ(t_stream_dirs.size(), 1);
    EXPECT_EQ(t_stream_dirs[0], tt::balancer::TStreamDir::R);
    EXPECT_EQ(enable_t_streaming, true);
}

}
