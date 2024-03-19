// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <exception>
#include <string>
#include <unordered_set>
#include <variant>
#include <vector>

namespace tt::placer
{
struct FailToPlaceOnCurrentEpoch : public std::exception
{
    std::string message;
    FailToPlaceOnCurrentEpoch(std::string const& message) : message(message) {}
    virtual char const* what() const noexcept override { return message.c_str(); }
};

struct FailToSatisfyPlacementConstraint : public std::exception
{
    std::string message;
    FailToSatisfyPlacementConstraint(std::string const& message) : message(message) {}
    virtual char const* what() const noexcept override { return message.c_str(); }
};

struct FailToSatisfyConflictingConstraint : public std::exception
{
    std::string message;
    FailToSatisfyConflictingConstraint(std::string const& message) : message(message) {}
    virtual char const* what() const noexcept override { return message.c_str(); }
};

struct FailToAllocateQueues : public std::exception
{
    std::string message;
    FailToAllocateQueues(std::string const& message) : message(message) {}
    virtual char const* what() const noexcept override { return message.c_str(); }
};

}  // namespace tt::placer
