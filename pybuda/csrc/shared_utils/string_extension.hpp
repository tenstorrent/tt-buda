// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string>

namespace tt::utils{
    // Convert a string to lower case
    std::string to_lower_string(const std::string& str);
    
    // Convert a string to upper case
    std::string to_upper_string(const std::string &str);
}