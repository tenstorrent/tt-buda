// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "string_extension.hpp"
#include <algorithm>
namespace tt::utils {
    // Convert a string to lower case
    std::string to_lower_string(const std::string& str) {
        std::string lower_str = str;
        std::transform(lower_str.begin(), lower_str.end(), lower_str.begin(), [](unsigned char c) { return std::tolower(c); });
        return lower_str;
    }

    // Convert a string to upper case
    std::string to_upper_string(const std::string &str) {
        std::string upper_str = str;
        std::transform(upper_str.begin(), upper_str.end(), upper_str.begin(), [](unsigned char c) { return std::toupper(c); });
        return upper_str;
    }
}