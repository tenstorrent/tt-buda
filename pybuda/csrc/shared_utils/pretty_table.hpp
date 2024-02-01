// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string>
#include <vector>

namespace tt::utils {

class PrettyTable {
   public:
    enum class Format {
        Pretty,
        CSV,
    };

    std::vector<std::vector<std::string>> table_;
    int horizontal_line_row = 1;
    int vertical_line_col = 1;
    int padding_between_cells = 2;

    void add_row(std::vector<std::string> incoming_new_row);
    void add_divider();
    bool is_divider_row(int row_index) const;
    std::string generate_table_string(Format format = Format::Pretty);
};

}  // namespace tt::utils
