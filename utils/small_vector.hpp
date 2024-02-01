// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <type_traits>
#include <vector>

#include "utils/assert.hpp"

namespace tt
{
template <typename T, std::size_t SmallSize = 1>
class SmallVector
{
    static_assert(std::is_standard_layout_v<T>);

    using Self = SmallVector<T, SmallSize>;
    static constexpr std::size_t SizeBytes = std::max(SmallSize * sizeof(T), sizeof(std::vector<T>));
    static constexpr std::size_t SizeElems = SizeBytes / sizeof(T);

   public:
    using Iterator = T*;
    using ConstIterator = T const*;

    // To match std::vector interface
    using iterator = Iterator;
    using const_iterator = ConstIterator;

   public:
    SmallVector() {}

    ~SmallVector()
    {
        if (curr_size > SizeElems)
        {
            as_vec().~vector();
        }
    }

    SmallVector(SmallVector const& other) { *this = other; }

    SmallVector<T, SmallSize>& operator=(SmallVector<T, SmallSize> const& other)
    {
        this->curr_size = other.size();
        if (other.size() <= SizeElems)
        {
            copy(as_arr(), other.as_arr(), other.size());
        }
        else
        {
            new (&as_vec()) std::vector<T>(other.as_vec());
        }
        return *this;
    }

    T const& operator[](int i) const
    {
        TT_ASSERT(i < (int)curr_size);
        return (curr_size <= SizeElems) ? as_arr()[i] : as_vec()[i];
    }

    T& operator[](int i)
    {
        TT_ASSERT(i < (int)curr_size);
        return (curr_size <= SizeElems) ? as_arr()[i] : as_vec()[i];
    }

    bool operator==(Self const& other) const
    {
        if (size() != other.size())
            return false;
        return std::memcmp(
                   static_cast<void const*>(data()), static_cast<void const*>(other.data()), size() * sizeof(T)) == 0;
    }

    bool operator!=(Self const& other) const { return !(*this == other); }

    void push_back(T const& t)
    {
        std::size_t i = curr_size++;
        if (i < SizeElems)
        {
            as_arr()[i] = t;
            return;
        }

        if (i == SizeElems)
        {
            from_arr_to_vec();
        }

        as_vec().push_back(t);
        TT_ASSERT(curr_size == as_vec().size());
    }

    template <typename... Args>
    void emplace_back(Args&&... args)
    {
        push_back(T{std::forward<Args>(args)...});
    }

    void pop_back()
    {
        TT_ASSERT(curr_size > 0);

        --curr_size;

        if (curr_size > SizeElems)
        {
            as_vec().pop_back();
            TT_ASSERT(curr_size == as_vec().size());
        }
        else if (curr_size == SizeElems)
        {
            from_vec_to_arr();
        }
    }

    void insert(std::size_t pos_i, T const& v)
    {
        push_back(v);
        for (std::size_t i = size() - 1; i > pos_i; --i) std::swap((*this)[i - 1], (*this)[i]);
    }

    void resize(std::size_t size, T const& v = {})
    {
        for (std::size_t i = 0; i < size; ++i) push_back(v);
    }

    void reserve(std::size_t)
    { /*nop, here to match the std::vector interface*/
    }

    void clear()
    {
        if (curr_size > SizeElems)
        {
            as_vec().~vector();
        }

        curr_size = 0;
    }

    bool empty() const { return curr_size == 0; }

    T& front() { return (*this)[0]; }
    T const& front() const { return (*this)[0]; }
    T& back() { return (*this)[curr_size - 1]; }
    T const& back() const { return (*this)[curr_size - 1]; }
    T* data() { return (curr_size <= SizeElems) ? as_arr() : as_vec().data(); }
    T const* data() const { return (curr_size <= SizeElems) ? as_arr() : as_vec().data(); }

    Iterator begin() { return data(); }
    Iterator end() { return data() + size(); }
    ConstIterator begin() const { return data(); }
    ConstIterator end() const { return data() + size(); }
    ConstIterator cbegin() { return data(); }
    ConstIterator cend() { return data() + size(); }
    ConstIterator cbegin() const { return data(); }
    ConstIterator cend() const { return data() + size(); }

    std::size_t size() const { return curr_size; }

    Iterator erase(ConstIterator const_pos)
    {
        Iterator pos = const_cast<Iterator>(const_pos);
        auto distance = std::distance(begin(), pos);
        for (Iterator from = pos + 1; from != cend(); ++from, ++pos)
        {
            *pos = *from;
        }

        --curr_size;

        if (curr_size > SizeElems)
        {
            as_vec().pop_back();
            TT_ASSERT(curr_size == as_vec().size());
        }
        else if (curr_size == SizeElems)
        {
            from_vec_to_arr();
        }

        return begin() + distance;
    }

   private:
    static void copy(T* to, T const* from, std::size_t num_elements)
    {
        if constexpr(std::is_trivially_copyable_v<T>)
        {
            std::memcpy(to, from, num_elements * sizeof(T));
        }
        else
        {
            T* to_iter = to;
            T const* from_iter = from;
            for (; to_iter != (to + num_elements); ++to_iter, ++from_iter) *to_iter = *from_iter;
        }
    }

    void from_arr_to_vec()
    {
        std::uint8_t arr[SizeBytes];
        copy(reinterpret_cast<T*>(arr), as_arr(), SizeElems);
        new (&as_vec()) std::vector<T>(reinterpret_cast<T*>(arr), reinterpret_cast<T*>(arr) + SizeElems);
    }

    void from_vec_to_arr()
    {
        std::uint8_t arr[SizeBytes];
        copy(reinterpret_cast<T*>(arr), as_vec().data(), SizeElems);
        as_vec().~vector();
        copy(as_arr(), reinterpret_cast<T const*>(arr), SizeElems);
    }

    T* as_arr() { return reinterpret_cast<T*>(storage); }
    T const* as_arr() const { return reinterpret_cast<T const*>(storage); }
    std::vector<T>& as_vec() { return *reinterpret_cast<std::vector<T>*>(storage); }
    std::vector<T> const& as_vec() const { return *reinterpret_cast<std::vector<T> const*>(storage); }

   private:
    std::size_t curr_size = 0;
    alignas(std::vector<T>) std::uint8_t storage[SizeBytes];
};
}  // namespace tt
