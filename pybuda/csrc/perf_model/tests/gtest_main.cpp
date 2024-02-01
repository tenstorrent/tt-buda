// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include <pybind11/embed.h>

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    pybind11::scoped_interpreter guard{};
    return RUN_ALL_TESTS();
}
