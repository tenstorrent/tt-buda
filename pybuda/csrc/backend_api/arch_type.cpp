// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include <stdexcept>
#include "arch_type.hpp"
#include "shared_utils/string_extension.hpp"

namespace tt{
    std::string to_string_arch(ARCH arch)
    {
        switch (arch)
        {
            case ARCH::GRAYSKULL:
                return "GRAYSKULL";
            case ARCH::WORMHOLE_B0:
                return "WORMHOLE_B0";
            case ARCH::BLACKHOLE:
                return "BLACKHOLE";
            default:
                throw std::runtime_error("Unsupported ARCH enum: " + std::to_string(static_cast<int>(arch)));
        }
    }

    std::string to_string_arch_lower(ARCH arch)
    {
        return tt::utils::to_lower_string(to_string_arch(arch));
    }

    ARCH to_arch_type(const std::string& arch_string)
    {
        std::string arch_string_lower = tt::utils::to_upper_string(arch_string);
        if (arch_string_lower == "GRAYSKULL")
        {
            return ARCH::GRAYSKULL;
        }
        else if (arch_string_lower == "WORMHOLE_B0")
        {
            return ARCH::WORMHOLE_B0;
        }
        else if (arch_string_lower == "BLACKHOLE")
        {
            return ARCH::BLACKHOLE;
        }
        else
        {
            throw std::runtime_error("Unsuported tt::ARCH string: " + arch_string_lower);
        }
    }
}