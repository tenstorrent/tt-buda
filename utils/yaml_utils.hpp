// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <string>

namespace tt
{

#define WRITE_YAML_LINE(out_stream, num_indents, content) (out_stream << yaml_indent(num_indents) << content << '\n')

#define YAML_KV_PAIR(key, value) key << ": " << value

template <int MAX_YAML_INDENT>
class YamlIndentLookup
{
   public:
    YamlIndentLookup()
    {
        for (unsigned int i = 0; i < yaml_indents_.size(); ++i)
        {
            yaml_indents_[i] = std::string(i, ' ');
        }
    }

    const std::string &get_yaml_indent(const int yaml_indent) const { return yaml_indents_.at(yaml_indent); }

   private:
    std::array<std::string, MAX_YAML_INDENT> yaml_indents_;
};

static const std::string &yaml_indent(const int num_spaces)
{
    const static YamlIndentLookup<20 /* MAX_YAML_INDENT */> yaml_indent_lookup;

    return yaml_indent_lookup.get_yaml_indent(num_spaces);
}

}  // namespace tt