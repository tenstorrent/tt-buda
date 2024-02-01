// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "pretty_table.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace tt::utils {

std::string spaces(int num) {
    std::string spaces_string;
    for (int i = 0; i < num; i++) {
        spaces_string.append(" ");
    }
    return spaces_string;
}

std::string lines(int num) {
    std::string spaces_string;
    for (int i = 0; i < num; i++) {
        spaces_string.append("-");
    }
    return spaces_string;
}

void PrettyTable::add_row(std::vector<std::string> incoming_new_row) {
    std::vector<std::string> new_row;

    for (std::uint32_t i = 0; i < incoming_new_row.size(); i++) {
        new_row.push_back(incoming_new_row[i]);
    }
    table_.push_back(new_row);

    // std::cout << table_.size() << std::endl;
    // std::cout << table_[0].size() << std::endl;
}

void PrettyTable::add_divider() { table_.push_back({"__divider__"}); }

bool PrettyTable::is_divider_row(int row_index) const {
    return table_[row_index].size() == 1 and table_[row_index][0] == "__divider__";
}

std::string PrettyTable::generate_table_string(Format format) {
    // Validate - all rows must be equal in size
    std::string column_delimeter = (format == Format::CSV) ? "," : "|  ";

    std::uint32_t row_size = table_[0].size();
    for (std::uint32_t row_index = 0; row_index < table_.size(); row_index++) {
        if (is_divider_row(row_index)) {
            continue;
        }
        if (row_size != table_[row_index].size()) {
            throw std::runtime_error("ERROR: all table rows must be the same size");
        }
    }

    // Pad cells per column
    if (format == Format::Pretty) {
        for (std::uint32_t col_index = 0; col_index < row_size; col_index++) {
            int max_cell_size = 0;

            // Determine max cell size in this column
            for (std::uint32_t row_index = 0; row_index < table_.size(); row_index++) {
                if (is_divider_row(row_index)) {
                    continue;
                }
                int cell_size = table_[row_index][col_index].size();
                if (max_cell_size < cell_size) {
                    max_cell_size = cell_size;
                }
            }

            // Pad cells to max cell size
            for (std::uint32_t row_index = 0; row_index < table_.size(); row_index++) {
                if (is_divider_row(row_index)) {
                    continue;
                }
                int cell_size = table_[row_index][col_index].size();
                std::string cell_padding = spaces(max_cell_size - cell_size + padding_between_cells);
                table_[row_index][col_index].append(cell_padding);
            }
        }
    }

    // Calculate horizontal line size
    int string_length_of_row = 0;
    int string_length_of_vertical_lines = (row_size + 1) * 3 - 2;
    for (std::uint32_t col_index = 0; col_index < row_size; col_index++) {
        string_length_of_row += table_[0][col_index].size();
    }
    int horizontal_line_string_size = string_length_of_row + string_length_of_vertical_lines;

    // Create string
    std::string table_string;

    if (format == Format::Pretty) {
        table_string.append(lines(horizontal_line_string_size));
        table_string.append("\n");
    }

    for (std::uint32_t row_index = 0; row_index < table_.size(); row_index++) {
        int row_size = table_[row_index].size();

        if ((int)row_index == horizontal_line_row and format == Format::Pretty) {
            table_string.append(lines(horizontal_line_string_size));
            table_string.append("\n");
        }

        if (is_divider_row(row_index)) {
            table_string.append(lines(horizontal_line_string_size));
            table_string.append("\n");
        } else {
            for (int col_index = 0; col_index < row_size; col_index++) {
                table_string.append(column_delimeter);
                table_string.append(table_[row_index][col_index]);
            }
            table_string.append(column_delimeter);
            table_string.append("\n");
        }
    }

    if (format == Format::Pretty) {
        table_string.append(lines(horizontal_line_string_size));
        table_string.append("\n");
    }

    return table_string;
}

}  // namespace tt::utils
