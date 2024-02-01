// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <utility>

namespace tt
{
template <typename T>
struct raw_ptr
{
    using element_type = T;

    element_type* ptr;

    raw_ptr() : ptr(nullptr) {}
    raw_ptr(element_type* ptr) : ptr(ptr) {}
    raw_ptr(const raw_ptr& other) : ptr(other.ptr) {}
    raw_ptr<element_type>& operator=(const raw_ptr& other) { ptr = other.ptr; }
    raw_ptr<element_type>& operator=(element_type* other) { ptr = other; }

    void swap(raw_ptr<element_type>& other) { std::swap(ptr, other.ptr); }
    element_type* get() const { return ptr; }
    element_type& operator*() const { return *ptr; }
    element_type* operator->() const { return ptr; }
    element_type& operator[](std::ptrdiff_t idx) const { return get()[idx]; }
    explicit operator bool() const { return get() != nullptr; }
};
}  // namespace tt
