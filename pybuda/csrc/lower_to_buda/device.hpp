// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <ostream>

namespace tt {

struct BudaDevice {
    int id;

    BudaDevice(int id) : id(id) {}
};

std::ostream &operator<<(std::ostream &os, BudaDevice const &d);

} // namespace tt
