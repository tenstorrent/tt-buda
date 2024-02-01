// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "reportify/paths.hpp"

#include <experimental/filesystem>
#include <fstream>
#include <iostream>

#include "utils/env.hpp"

namespace tt {

namespace reportify {

static std::string get_variant(std::string const& test_name)
{
    std::string variant;

    if (auto maybe = env_as_optional<std::string>("PYBUDA_REPORTIFY_POSTFIX"); maybe and not test_name.empty())
    {
        variant = "." + *maybe;
    }
    return test_name + variant;
}

std::string get_default_reportify_path(const std::string& test_name) {
  const char* home_dir = getenv("HOME");
  std::string home_dir_str;
  if (home_dir == NULL) {
    std::cerr << "HOME not set, can't find testify dir. Will write out results "
                 "in current dir."
              << std::endl;
    home_dir_str = ".";
  } else {
    home_dir_str = std::string(home_dir);
  }
  const std::string default_reportify_dir = home_dir_str + "/testify/ll-sw/" + get_variant(test_name);

  return default_reportify_dir;
}

std::string get_pass_reports_relative_directory() {
  std::string retstring("/buda_reports/Passes/");

  return retstring;
}

std::string get_router_report_relative_directory() {
  std::string retstring("/router_reports/EpochType/");

  return retstring;
}

std::string get_epoch_type_report_relative_directory() {
  std::string retstring("/epoch_reports/EpochType/");

  return retstring;
}


std::string get_epoch_id_report_relative_directory() {
  std::string retstring("/epoch_reports/EpochId/");

  return retstring;

}

std::string get_memory_report_relative_directory() {
  std::string retstring("/memory_reports/");

  return retstring;

}

std::string get_default_reportify_path_for_sage_reports(
    const std::string& test_name) {
  std::string base_dir = get_default_reportify_path(test_name);

  std::string sage_report_dir = base_dir + get_pass_reports_relative_directory();

  return sage_report_dir;
}

std::string get_pass_reports_path_from_reportify_dir(
    const std::string& reportify_dir, const std::string& test_name) {
    std::string sage_report_dir = reportify_dir + get_variant(test_name) + get_pass_reports_relative_directory();

    return sage_report_dir;
}

std::string build_report_path(std::string const& base_path, std::string const& test_name, std::string report_name)
{
    return base_path + get_variant(test_name) + "/" + report_name + "/";
}


bool initalize_reportify_directory(const std::string& reportify_dir,
                                   const std::string& test_name) {
    std::string dir = reportify_dir + "/" + get_variant(test_name);
    std::string summary_filename = dir + "/" + "summary.yaml";

    std::experimental::filesystem::create_directories(dir);

    std::ofstream ofs(summary_filename);

    if (!ofs.is_open())
    {
        return false;
  }

  ofs << "content:" << std::endl
      << "  name: ll-sw." << get_variant(test_name) << std::endl
      << "  output-dir: " << dir << std::endl;
  ofs << "type: summary" << std::endl;
  ofs.close();

  //std::cout << "Wrote dumps to " + summary_filename << std::endl;

  return true;
}

bool initalize_default_reportify_directory(const std::string& test_name) {
  std::string default_reportify_dir = get_default_reportify_path("");

  return initalize_reportify_directory(default_reportify_dir, test_name);
}


} // namespace reportify

} // namespace tt
