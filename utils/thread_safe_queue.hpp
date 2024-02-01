// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string>
#include <queue>
#include <memory>
#include <mutex>
#include <condition_variable>

namespace tt
{


enum class ReadAction : int
{
    Peek    = 0,
    Pop     = 1,
};

template <typename T>
struct ThreadSafeQueue
{

private:
    mutable std::mutex mutex;
    std::condition_variable cond_var;
    std::deque<T> data_queue;
    int max_size;
    std::string name;

public:

    ThreadSafeQueue(int max_size_) : max_size(max_size_)
    {
    }

    ThreadSafeQueue(std::string _name = "") : name(_name), max_size(0)
    {
    }

    ThreadSafeQueue(const ThreadSafeQueue& rhs)
    {
        std::scoped_lock scoped_lock(rhs.mutex);
        this->data_queue = rhs.data_queue;
        this->max_size = rhs.max_size;
    }

    std::string get_name() {
        std::scoped_lock scoped_lock(this->mutex);
        return this->name;
    }

    void push_blocking(T value)
    {
        std::unique_lock<std::mutex> lock(this->mutex);
        this->cond_var.wait(lock, [this] {
            if (this->max_size == 0)
            {
                return true;
            }
            else
            {
                return this->data_queue.size() < this->max_size;
            }
        });
        this->data_queue.push_back(value);
        this->cond_var.notify_one();
    }

    void push_front_blocking(T value)
    {
        std::unique_lock<std::mutex> lock(this->mutex);
        this->cond_var.wait(lock, [this] {
            if (this->max_size == 0)
            {
                return true;
            }
            else
            {
                return this->data_queue.size() < this->max_size;
            }
        });
        this->data_queue.push_front(value);
        this->cond_var.notify_one();
    }

    void pop_blocking_by_ref(T& value)
    {
        std::unique_lock<std::mutex> lock(this->mutex);
        this->cond_var.wait(lock, [this] {
            return (not this->data_queue.empty());
        });
        value = this->data_queue.front();
        this->data_queue.pop_front();
    }

    std::shared_ptr<T> pop_blocking_return_shared()
    {
        std::unique_lock<std::mutex> lock(this->mutex);
        this->cond_var.wait(lock,[this] {
            return (not this->data_queue.empty());
        });
        std::shared_ptr<T> result(std::make_shared<T>(this->data_queue.front()));
        this->data_queue.pop_front();
        return result;
    }

    void clear()
    {
        std::scoped_lock scoped_lock(this->mutex);
        this->data_queue.clear();
    }

    bool empty() const
    {
        std::scoped_lock scoped_lock(this->mutex);
        return this->data_queue.empty();
    }

    size_t size() const
    {
        std::scoped_lock scoped_lock(this->mutex);
        return this->data_queue.size();
    }

    const T& front() const
    {
        std::scoped_lock scoped_lock(this->mutex);
        return this->data_queue.front();
    }

    const T& back() const
    {
        std::scoped_lock scoped_lock(this->mutex);
        return this->data_queue.back();
    }

    const T& at(size_t index) const
    {
        std::scoped_lock scoped_lock(this->mutex);
        return this->data_queue.at(index);
    }
};

} // end namespace tt
