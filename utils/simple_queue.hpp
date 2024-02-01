// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string>
#include <queue>
#include <memory>

namespace tt
{

template <typename T>
struct SimpleQueue
{

private:
    const int MAX_QUEUE_SIZE = 1024;
    std::deque<T> data_queue;
    int max_size;
    std::string name;

    int rd_ptr;
    int shadow_rd_ptr;
    int offset;

public:

    SimpleQueue(int max_size_) : max_size(max_size_), rd_ptr(0), shadow_rd_ptr(0), offset(0)
    {
    }

    SimpleQueue(std::string _name = "") : name(_name), max_size(MAX_QUEUE_SIZE), rd_ptr(0), shadow_rd_ptr(0), offset(0)
    {
    }

    SimpleQueue(const SimpleQueue& rhs)
    {
        this->data_queue = rhs.data_queue;
        this->max_size = rhs.max_size;

        this->rd_ptr = 0;
        this->shadow_rd_ptr = 0;
        this->offset = 0;
    }

    std::string get_name() 
    {
        return this->name;
    }

    void set_name(const std::string &name)
    {
        this->name = name;
    }

    void assert_not_full() 
    {
        TT_ASSERT(
            not this->full(), 
            "Trying to push to a full SimpleQueue: " + this->name + ", current size = " + std::to_string(this->size())
        );
    }

    void assert_not_empty() 
    {
        TT_ASSERT(not this->empty(), "Trying to pop from an empty SimpleQueue: " + this->name);
    }

    void push_blocking(T value)
    {
        this->assert_not_full();   
        this->data_queue.push_back(value);
    }

    void pop_blocking_by_ref(T& value)
    {   
        this->assert_not_empty();
        value = this->data_queue.front();
        this->data_queue.pop_front();

        if (this->rd_ptr > 0) {
            this->rd_ptr--;
        }
        // Whenever we change the rd_ptr, we just reset the shadow_rd_ptr to the same value 
        // (this allows us to reset the shadow_rd_ptr in the forward pass from the backward pass).
        // This is needed for training
        this->shadow_rd_ptr = this->rd_ptr;

    }

    void clear(bool use_shadow_rd_ptr = false)
    {
        this->data_queue.clear();

        if (use_shadow_rd_ptr) {
            this->shadow_rd_ptr = 0;
        } else {
            this->rd_ptr = 0;
            this->shadow_rd_ptr = this->rd_ptr;
        }
    }

    void set_ptr(int ptr, bool use_shadow_rd_ptr = false)
    {
        if (not use_shadow_rd_ptr) {
            this->rd_ptr = ptr;
        }

        // This is done at all times
        this->shadow_rd_ptr = ptr;
    }

    int get_ptr(bool use_shadow_rd_ptr = false) const
    {
        if (use_shadow_rd_ptr) {
            return this->shadow_rd_ptr;
        }
        return this->rd_ptr;
    }

    int get_offset() const
    {
        return this->offset;
    }

    void reset_offset() 
    {
        this->offset = 0;
    }

    void increment_offset() 
    {
        this->offset++;
    }

    bool empty() const
    {
        return this->data_queue.empty();
    }

    bool full() const
    {
        return this->data_queue.size() == this->max_size;
    }

    size_t size() const
    {
        return this->data_queue.size();
    }

    const T& front() const
    {
        return this->data_queue.front();
    }

    const T& back() const
    {
        return this->data_queue.back();
    }

    const T& read(bool use_shadow_rd_ptr = false) 
    {
        int rd_ptr = (use_shadow_rd_ptr) ? this->shadow_rd_ptr: this->rd_ptr;
        TT_ASSERT(
            rd_ptr + this->offset < this->size() and 0 <= rd_ptr, 
            "Trying to read queue " + this->name + " with size " + std::to_string(this->size()) + " out-of-bounds with index " + std::to_string(rd_ptr)
        );

        return this->data_queue.at(rd_ptr + this->offset);
    }
};

} // end namespace tt
