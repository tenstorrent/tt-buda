// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "lower_to_buda/device.hpp"

namespace tt {

std::ostream &operator<<(std::ostream &os, BudaDevice const &d) {
    return os << d.id;
}


}
