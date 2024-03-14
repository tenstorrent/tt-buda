// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string>
#include "device/tt_arch_types.h"

namespace tt {
    std::string to_string_arch(ARCH ar);
    std::string to_string_arch_lower(ARCH arch);
    ARCH to_arch_type(const std::string& arch_string);
}


