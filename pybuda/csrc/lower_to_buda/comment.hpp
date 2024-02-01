// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>

namespace tt
{

struct Comment
{
    Comment() = default;
    Comment(char const *str) : str(str) {}
    Comment(std::string const& str) : str(str) {}
    Comment(std::stringstream const& str) : str(str.str()) {}

    operator bool() const { return not str.empty(); }

    std::string str;
};

inline std::ostream &operator<<(std::ostream &os, Comment const &comment)
{
    std::string::size_type begin = 0;
    std::string::size_type end = 0;
    while (end != comment.str.size())
    {
        end = comment.str.find('\n', begin);
        if (end == std::string::npos)
            end = comment.str.size();
        std::string::size_type size = end - begin;
        os << "# " << std::string_view(comment.str.data() + begin, size) << std::endl;
        begin = end + 1;
    }
    return os;
}

}  // namespace tt
