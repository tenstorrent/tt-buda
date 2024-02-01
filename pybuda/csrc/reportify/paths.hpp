// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <string>

namespace tt {
namespace reportify {

std::string build_report_path(std::string const& base_path, std::string const& test_name, std::string report_name);
bool initalize_reportify_directory(const std::string& reportify_dir, const std::string& test_name);
std::string get_default_reportify_path(const std::string& test_name);
std::string get_pass_reports_relative_directory();
std::string get_router_report_relative_directory();
std::string get_memory_report_relative_directory();
std::string get_epoch_type_report_relative_directory();
std::string get_epoch_id_report_relative_directory();
inline std::string get_constraint_reports_relative_directory() { return "/constraint_reports/"; }

} // namespace reportify
} // namespace tt

