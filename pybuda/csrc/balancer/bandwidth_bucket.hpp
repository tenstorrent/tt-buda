
// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <ostream>
#include <string>

namespace tt
{
namespace balancer
{

// Class that represents bandwidth range in the form of a pair of doubles.
class BandwidthBucket
{
   public:
    enum class BucketIndex
    {
        k0to4 = 0,
        k4to8,
        k8to12,
        k12to16,
        k16to20,
        k20to24,
        k24to28,
        k28to32
    };

    BandwidthBucket(int bucket_index) : bucket_index_(bucket_index) {}

    BandwidthBucket(double bandwidth) : bucket_index_(static_cast<int>(bandwidth / BUCKET_WIDTH)) {}

    BandwidthBucket(BucketIndex bucket_index) : bucket_index_(static_cast<int>(bucket_index)) {}

    void set_bucket(int bucket_index) { bucket_index_ = bucket_index; }

    std::pair<double, double> get_bucket() const
    {
        return {bucket_index_ * BUCKET_WIDTH, (bucket_index_ + 1) * BUCKET_WIDTH};
    }

    std::string to_string() const
    {
        auto bucket = get_bucket();
        return "BandwidthBucket(" + std::to_string(bucket.first) + ", " + std::to_string(bucket.second) + ")";
    }

    double get_bandwidth() const
    {
        auto bucket = get_bucket();
        return (bucket.second + bucket.first) / 2.0;
    }

    BucketIndex get_bucket_index() const { return static_cast<BucketIndex>(bucket_index_); }

    friend std::ostream& operator<<(std::ostream& os, const BandwidthBucket& bandwidth_bucket)
    {
        os << bandwidth_bucket.to_string();
        return os;
    }

   private:
    int bucket_index_;

    static constexpr double BUCKET_WIDTH = 4.0;
};

}  // namespace balancer
}  // namespace tt
