// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "graph_lib/defines.hpp"
#include "graph_lib/node.hpp"
#include "utils/assert.hpp"

#include <string>

namespace tt::graphlib
{

std::ostream& operator<<(std::ostream& out, const NodeEpochType& node_epoch_type)
{
    return out << node_epoch_type_to_string(node_epoch_type);
}

}  // namespace tt::graphlib
