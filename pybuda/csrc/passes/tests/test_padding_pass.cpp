// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "graph_lib/shape.hpp"
#include "gtest/gtest.h"
#include "passes/padding_pass_placer.hpp"

using namespace tt;

struct PaddingFunctionTests : testing::Test
{
};

TEST_F(PaddingFunctionTests, test_biggest_prime_factor_10_increment)
{
    std::uint32_t shape_r_size = 13 * tt::graphlib::Shape::BUDA_TILE_DIM;  // 13 tiles

    // Per BIGGEST_FACTOR_PRIME_10_INCREMENT padding function 13 tiles should be irregular shape.
    ASSERT_EQ(true, tt::padding_placer::is_irregular(shape_r_size, tt::padding_placer::PaddingCriterion::BIGGEST_FACTOR_PRIME_10_INCREMENT));

    // Per BIGGEST_FACTOR_PRIME_10_INCREMENT padding function should add one tile of padding.
    ASSERT_EQ(
        tt::graphlib::Shape::BUDA_TILE_DIM,
        tt::padding_placer::compute_pad(shape_r_size, tt::padding_placer::PaddingCriterion::BIGGEST_FACTOR_PRIME_10_INCREMENT));
}
