// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "balancer/bandwidth_estimator_impl.hpp"
#include "balancer/bandwidth_bucket.hpp"

namespace tt
{
namespace balancer
{

BandwidthBucket estimate_direct_connection(const int unpacker_buffer_size_bytes,
                                           const int kernel_clear_granularity,
                                           const int buf_space_available_ack_thr,
                                           const int epoch_tiles,
                                           const int tile_size,
                                           const int packer_buffer_size_bytes,
                                           const int packer_scatter_gather_num_tiles,
                                           const int packer_num_phases,
                                           const bool scatter_pack)
{
    if (scatter_pack <= 0.50)
    {
        if (buf_space_available_ack_thr <= 0.50)
        {
            if (unpacker_buffer_size_bytes <= 34560.00)
            {
                return BandwidthBucket(6);
            }
            else  // if unpacker_buffer_size_bytes > 34560.00
            {
                if (packer_scatter_gather_num_tiles <= 155.00)
                {
                    if (packer_scatter_gather_num_tiles <= 93.00)
                    {
                        return BandwidthBucket(7);
                    }
                    else  // if packer_scatter_gather_num_tiles > 93.00
                    {
                        if (unpacker_buffer_size_bytes <= 36640.00)
                        {
                            if (packer_buffer_size_bytes <= 318080.00)
                            {
                                if (kernel_clear_granularity <= 6.00)
                                {
                                    if (epoch_tiles <= 2656.00)
                                    {
                                        return BandwidthBucket(6);
                                    }
                                    else  // if epoch_tiles > 2656.00
                                    {
                                        return BandwidthBucket(7);
                                    }
                                }
                                else  // if kernel_clear_granularity > 6.00
                                {
                                    return BandwidthBucket(7);
                                }
                            }
                            else  // if packer_buffer_size_bytes > 318080.00
                            {
                                return BandwidthBucket(6);
                            }
                        }
                        else  // if unpacker_buffer_size_bytes > 36640.00
                        {
                            return BandwidthBucket(7);
                        }
                    }
                }
                else  // if packer_scatter_gather_num_tiles > 155.00
                {
                    if (unpacker_buffer_size_bytes <= 38080.00)
                    {
                        return BandwidthBucket(6);
                    }
                    else  // if unpacker_buffer_size_bytes > 38080.00
                    {
                        if (packer_scatter_gather_num_tiles <= 270.00)
                        {
                            return BandwidthBucket(7);
                        }
                        else  // if packer_scatter_gather_num_tiles > 270.00
                        {
                            return BandwidthBucket(6);
                        }
                    }
                }
            }
        }
        else  // if buf_space_available_ack_thr > 0.50
        {
            if (unpacker_buffer_size_bytes <= 40960.00)
            {
                if (buf_space_available_ack_thr <= 1.50)
                {
                    if (epoch_tiles <= 928.00)
                    {
                        if (unpacker_buffer_size_bytes <= 36640.00)
                        {
                            if (unpacker_buffer_size_bytes <= 34560.00)
                            {
                                if (packer_scatter_gather_num_tiles <= 7.50)
                                {
                                    return BandwidthBucket(4);
                                }
                                else  // if packer_scatter_gather_num_tiles > 7.50
                                {
                                    if (epoch_tiles <= 608.00)
                                    {
                                        return BandwidthBucket(4);
                                    }
                                    else  // if epoch_tiles > 608.00
                                    {
                                        return BandwidthBucket(4);
                                    }
                                }
                            }
                            else  // if unpacker_buffer_size_bytes > 34560.00
                            {
                                if (epoch_tiles <= 832.00)
                                {
                                    return BandwidthBucket(4);
                                }
                                else  // if epoch_tiles > 832.00
                                {
                                    return BandwidthBucket(5);
                                }
                            }
                        }
                        else  // if unpacker_buffer_size_bytes > 36640.00
                        {
                            return BandwidthBucket(5);
                        }
                    }
                    else  // if epoch_tiles > 928.00
                    {
                        if (packer_buffer_size_bytes <= 38080.00)
                        {
                            if (unpacker_buffer_size_bytes <= 38080.00)
                            {
                                return BandwidthBucket(4);
                            }
                            else  // if unpacker_buffer_size_bytes > 38080.00
                            {
                                return BandwidthBucket(5);
                            }
                        }
                        else  // if packer_buffer_size_bytes > 38080.00
                        {
                            if (epoch_tiles <= 1568.00)
                            {
                                if (unpacker_buffer_size_bytes <= 36640.00)
                                {
                                    if (epoch_tiles <= 992.00)
                                    {
                                        return BandwidthBucket(4);
                                    }
                                    else  // if epoch_tiles > 992.00
                                    {
                                        if (packer_buffer_size_bytes <= 42560.00)
                                        {
                                            return BandwidthBucket(4);
                                        }
                                        else  // if packer_buffer_size_bytes > 42560.00
                                        {
                                            if (packer_scatter_gather_num_tiles <= 38.00)
                                            {
                                                if (packer_scatter_gather_num_tiles <= 34.00)
                                                {
                                                    return BandwidthBucket(5);
                                                }
                                                else  // if packer_scatter_gather_num_tiles > 34.00
                                                {
                                                    return BandwidthBucket(4);
                                                }
                                            }
                                            else  // if packer_scatter_gather_num_tiles > 38.00
                                            {
                                                return BandwidthBucket(5);
                                            }
                                        }
                                    }
                                }
                                else  // if unpacker_buffer_size_bytes > 36640.00
                                {
                                    return BandwidthBucket(5);
                                }
                            }
                            else  // if epoch_tiles > 1568.00
                            {
                                if (packer_buffer_size_bytes <= 58240.00)
                                {
                                    return BandwidthBucket(5);
                                }
                                else  // if packer_buffer_size_bytes > 58240.00
                                {
                                    return BandwidthBucket(5);
                                }
                            }
                        }
                    }
                }
                else  // if buf_space_available_ack_thr > 1.50
                {
                    if (unpacker_buffer_size_bytes <= 38880.00)
                    {
                        if (packer_buffer_size_bytes <= 7520.00)
                        {
                            return BandwidthBucket(3);
                        }
                        else  // if packer_buffer_size_bytes > 7520.00
                        {
                            if (epoch_tiles <= 480.00)
                            {
                                if (unpacker_buffer_size_bytes <= 34560.00)
                                {
                                    if (epoch_tiles <= 288.00)
                                    {
                                        return BandwidthBucket(5);
                                    }
                                    else  // if epoch_tiles > 288.00
                                    {
                                        return BandwidthBucket(6);
                                    }
                                }
                                else  // if unpacker_buffer_size_bytes > 34560.00
                                {
                                    if (epoch_tiles <= 352.00)
                                    {
                                        return BandwidthBucket(6);
                                    }
                                    else  // if epoch_tiles > 352.00
                                    {
                                        return BandwidthBucket(6);
                                    }
                                }
                            }
                            else  // if epoch_tiles > 480.00
                            {
                                return BandwidthBucket(5);
                            }
                        }
                    }
                    else  // if unpacker_buffer_size_bytes > 38880.00
                    {
                        if (packer_buffer_size_bytes <= 10080.00)
                        {
                            return BandwidthBucket(4);
                        }
                        else  // if packer_buffer_size_bytes > 10080.00
                        {
                            return BandwidthBucket(6);
                        }
                    }
                }
            }
            else  // if unpacker_buffer_size_bytes > 40960.00
            {
                if (unpacker_buffer_size_bytes <= 48480.00)
                {
                    if (epoch_tiles <= 1120.00)
                    {
                        if (buf_space_available_ack_thr <= 1.50)
                        {
                            return BandwidthBucket(5);
                        }
                        else  // if buf_space_available_ack_thr > 1.50
                        {
                            return BandwidthBucket(6);
                        }
                    }
                    else  // if epoch_tiles > 1120.00
                    {
                        return BandwidthBucket(6);
                    }
                }
                else  // if unpacker_buffer_size_bytes > 48480.00
                {
                    if (packer_scatter_gather_num_tiles <= 13.00)
                    {
                        return BandwidthBucket(6);
                    }
                    else  // if packer_scatter_gather_num_tiles > 13.00
                    {
                        return BandwidthBucket(7);
                    }
                }
            }
        }
    }
    else  // if scatter_pack > 0.50
    {
        if (packer_buffer_size_bytes <= 96960.00)
        {
            if (packer_buffer_size_bytes <= 48480.00)
            {
                if (packer_buffer_size_bytes <= 15040.00)
                {
                    if (epoch_tiles <= 576.00)
                    {
                        return BandwidthBucket(0);
                    }
                    else  // if epoch_tiles > 576.00
                    {
                        return BandwidthBucket(1);
                    }
                }
                else  // if packer_buffer_size_bytes > 15040.00
                {
                    if (packer_buffer_size_bytes <= 40960.00)
                    {
                        return BandwidthBucket(1);
                    }
                    else  // if packer_buffer_size_bytes > 40960.00
                    {
                        if (packer_num_phases <= 11.00)
                        {
                            return BandwidthBucket(2);
                        }
                        else  // if packer_num_phases > 11.00
                        {
                            return BandwidthBucket(1);
                        }
                    }
                }
            }
            else  // if packer_buffer_size_bytes > 48480.00
            {
                if (packer_buffer_size_bytes <= 57120.00)
                {
                    if (packer_num_phases <= 17.50)
                    {
                        if (buf_space_available_ack_thr <= 0.50)
                        {
                            return BandwidthBucket(3);
                        }
                        else  // if buf_space_available_ack_thr > 0.50
                        {
                            return BandwidthBucket(2);
                        }
                    }
                    else  // if packer_num_phases > 17.50
                    {
                        if (packer_num_phases <= 21.50)
                        {
                            if (epoch_tiles <= 1152.00)
                            {
                                return BandwidthBucket(1);
                            }
                            else  // if epoch_tiles > 1152.00
                            {
                                return BandwidthBucket(2);
                            }
                        }
                        else  // if packer_num_phases > 21.50
                        {
                            if (epoch_tiles <= 1568.00)
                            {
                                return BandwidthBucket(1);
                            }
                            else  // if epoch_tiles > 1568.00
                            {
                                return BandwidthBucket(1);
                            }
                        }
                    }
                }
                else  // if packer_buffer_size_bytes > 57120.00
                {
                    if (packer_scatter_gather_num_tiles <= 3.50)
                    {
                        return BandwidthBucket(2);
                    }
                    else  // if packer_scatter_gather_num_tiles > 3.50
                    {
                        if (packer_buffer_size_bytes <= 80640.00)
                        {
                            return BandwidthBucket(2);
                        }
                        else  // if packer_buffer_size_bytes > 80640.00
                        {
                            return BandwidthBucket(3);
                        }
                    }
                }
            }
        }
        else  // if packer_buffer_size_bytes > 96960.00
        {
            if (packer_buffer_size_bytes <= 193920.00)
            {
                if (packer_num_phases <= 40.50)
                {
                    if (packer_buffer_size_bytes <= 118720.00)
                    {
                        if (packer_scatter_gather_num_tiles <= 1.50)
                        {
                            if (packer_num_phases <= 17.50)
                            {
                                return BandwidthBucket(3);
                            }
                            else  // if packer_num_phases > 17.50
                            {
                                if (epoch_tiles <= 1664.00)
                                {
                                    return BandwidthBucket(2);
                                }
                                else  // if epoch_tiles > 1664.00
                                {
                                    if (packer_num_phases <= 28.50)
                                    {
                                        return BandwidthBucket(3);
                                    }
                                    else  // if packer_num_phases > 28.50
                                    {
                                        return BandwidthBucket(2);
                                    }
                                }
                            }
                        }
                        else  // if packer_scatter_gather_num_tiles > 1.50
                        {
                            if (epoch_tiles <= 1152.00)
                            {
                                return BandwidthBucket(2);
                            }
                            else  // if epoch_tiles > 1152.00
                            {
                                return BandwidthBucket(3);
                            }
                        }
                    }
                    else  // if packer_buffer_size_bytes > 118720.00
                    {
                        if (packer_buffer_size_bytes <= 170560.00)
                        {
                            if (packer_buffer_size_bytes <= 125120.00)
                            {
                                if (epoch_tiles <= 3648.00)
                                {
                                    return BandwidthBucket(3);
                                }
                                else  // if epoch_tiles > 3648.00
                                {
                                    return BandwidthBucket(4);
                                }
                            }
                            else  // if packer_buffer_size_bytes > 125120.00
                            {
                                return BandwidthBucket(3);
                            }
                        }
                        else  // if packer_buffer_size_bytes > 170560.00
                        {
                            if (packer_scatter_gather_num_tiles <= 3.50)
                            {
                                return BandwidthBucket(3);
                            }
                            else  // if packer_scatter_gather_num_tiles > 3.50
                            {
                                if (kernel_clear_granularity <= 4.50)
                                {
                                    return BandwidthBucket(4);
                                }
                                else  // if kernel_clear_granularity > 4.50
                                {
                                    return BandwidthBucket(3);
                                }
                            }
                        }
                    }
                }
                else  // if packer_num_phases > 40.50
                {
                    if (tile_size <= 1600.00)
                    {
                        if (packer_buffer_size_bytes <= 159040.00)
                        {
                            return BandwidthBucket(2);
                        }
                        else  // if packer_buffer_size_bytes > 159040.00
                        {
                            if (packer_num_phases <= 62.50)
                            {
                                return BandwidthBucket(3);
                            }
                            else  // if packer_num_phases > 62.50
                            {
                                return BandwidthBucket(2);
                            }
                        }
                    }
                    else  // if tile_size > 1600.00
                    {
                        return BandwidthBucket(3);
                    }
                }
            }
            else  // if packer_buffer_size_bytes > 193920.00
            {
                if (packer_scatter_gather_num_tiles <= 1.50)
                {
                    if (packer_buffer_size_bytes <= 288960.00)
                    {
                        if (packer_num_phases <= 83.00)
                        {
                            if (packer_num_phases <= 36.50)
                            {
                                if (packer_buffer_size_bytes <= 220480.00)
                                {
                                    return BandwidthBucket(3);
                                }
                                else  // if packer_buffer_size_bytes > 220480.00
                                {
                                    return BandwidthBucket(4);
                                }
                            }
                            else  // if packer_num_phases > 36.50
                            {
                                return BandwidthBucket(3);
                            }
                        }
                        else  // if packer_num_phases > 83.00
                        {
                            if (packer_buffer_size_bytes <= 259840.00)
                            {
                                if (kernel_clear_granularity <= 2.50)
                                {
                                    return BandwidthBucket(3);
                                }
                                else  // if kernel_clear_granularity > 2.50
                                {
                                    return BandwidthBucket(2);
                                }
                            }
                            else  // if packer_buffer_size_bytes > 259840.00
                            {
                                return BandwidthBucket(3);
                            }
                        }
                    }
                    else  // if packer_buffer_size_bytes > 288960.00
                    {
                        if (packer_num_phases <= 63.50)
                        {
                            if (packer_buffer_size_bytes <= 305760.00)
                            {
                                if (packer_num_phases <= 43.00)
                                {
                                    return BandwidthBucket(4);
                                }
                                else  // if packer_num_phases > 43.00
                                {
                                    return BandwidthBucket(3);
                                }
                            }
                            else  // if packer_buffer_size_bytes > 305760.00
                            {
                                return BandwidthBucket(4);
                            }
                        }
                        else  // if packer_num_phases > 63.50
                        {
                            if (packer_buffer_size_bytes <= 577920.00)
                            {
                                if (epoch_tiles <= 9920.00)
                                {
                                    if (packer_buffer_size_bytes <= 484800.00)
                                    {
                                        if (kernel_clear_granularity <= 3.50)
                                        {
                                            if (packer_buffer_size_bytes <= 366400.00)
                                            {
                                                return BandwidthBucket(3);
                                            }
                                            else  // if packer_buffer_size_bytes > 366400.00
                                            {
                                                if (packer_num_phases <= 78.50)
                                                {
                                                    return BandwidthBucket(4);
                                                }
                                                else  // if packer_num_phases > 78.50
                                                {
                                                    return BandwidthBucket(3);
                                                }
                                            }
                                        }
                                        else  // if kernel_clear_granularity > 3.50
                                        {
                                            return BandwidthBucket(3);
                                        }
                                    }
                                    else  // if packer_buffer_size_bytes > 484800.00
                                    {
                                        if (packer_num_phases <= 114.00)
                                        {
                                            return BandwidthBucket(4);
                                        }
                                        else  // if packer_num_phases > 114.00
                                        {
                                            if (kernel_clear_granularity <= 1.50)
                                            {
                                                if (tile_size <= 1600.00)
                                                {
                                                    return BandwidthBucket(3);
                                                }
                                                else  // if tile_size > 1600.00
                                                {
                                                    return BandwidthBucket(4);
                                                }
                                            }
                                            else  // if kernel_clear_granularity > 1.50
                                            {
                                                return BandwidthBucket(3);
                                            }
                                        }
                                    }
                                }
                                else  // if epoch_tiles > 9920.00
                                {
                                    if (tile_size <= 1600.00)
                                    {
                                        return BandwidthBucket(3);
                                    }
                                    else  // if tile_size > 1600.00
                                    {
                                        return BandwidthBucket(4);
                                    }
                                }
                            }
                            else  // if packer_buffer_size_bytes > 577920.00
                            {
                                if (packer_num_phases <= 217.00)
                                {
                                    if (kernel_clear_granularity <= 3.50)
                                    {
                                        return BandwidthBucket(4);
                                    }
                                    else  // if kernel_clear_granularity > 3.50
                                    {
                                        if (packer_buffer_size_bytes <= 625600.00)
                                        {
                                            return BandwidthBucket(3);
                                        }
                                        else  // if packer_buffer_size_bytes > 625600.00
                                        {
                                            return BandwidthBucket(4);
                                        }
                                    }
                                }
                                else  // if packer_num_phases > 217.00
                                {
                                    return BandwidthBucket(3);
                                }
                            }
                        }
                    }
                }
                else  // if packer_scatter_gather_num_tiles > 1.50
                {
                    if (packer_buffer_size_bytes <= 293120.00)
                    {
                        if (packer_num_phases <= 40.00)
                        {
                            if (packer_buffer_size_bytes <= 228800.00)
                            {
                                if (packer_num_phases <= 17.50)
                                {
                                    return BandwidthBucket(4);
                                }
                                else  // if packer_num_phases > 17.50
                                {
                                    if (kernel_clear_granularity <= 2.50)
                                    {
                                        return BandwidthBucket(4);
                                    }
                                    else  // if kernel_clear_granularity > 2.50
                                    {
                                        if (packer_scatter_gather_num_tiles <= 3.50)
                                        {
                                            if (packer_buffer_size_bytes <= 208320.00)
                                            {
                                                return BandwidthBucket(3);
                                            }
                                            else  // if packer_buffer_size_bytes > 208320.00
                                            {
                                                return BandwidthBucket(3);
                                            }
                                        }
                                        else  // if packer_scatter_gather_num_tiles > 3.50
                                        {
                                            return BandwidthBucket(4);
                                        }
                                    }
                                }
                            }
                            else  // if packer_buffer_size_bytes > 228800.00
                            {
                                return BandwidthBucket(4);
                            }
                        }
                        else  // if packer_num_phases > 40.00
                        {
                            if (unpacker_buffer_size_bytes <= 34560.00)
                            {
                                return BandwidthBucket(4);
                            }
                            else  // if unpacker_buffer_size_bytes > 34560.00
                            {
                                return BandwidthBucket(3);
                            }
                        }
                    }
                    else  // if packer_buffer_size_bytes > 293120.00
                    {
                        if (packer_scatter_gather_num_tiles <= 2.50)
                        {
                            if (tile_size <= 1600.00)
                            {
                                if (packer_buffer_size_bytes <= 380800.00)
                                {
                                    if (packer_num_phases <= 64.00)
                                    {
                                        return BandwidthBucket(4);
                                    }
                                    else  // if packer_num_phases > 64.00
                                    {
                                        return BandwidthBucket(3);
                                    }
                                }
                                else  // if packer_buffer_size_bytes > 380800.00
                                {
                                    return BandwidthBucket(4);
                                }
                            }
                            else  // if tile_size > 1600.00
                            {
                                if (packer_buffer_size_bytes <= 773760.00)
                                {
                                    return BandwidthBucket(4);
                                }
                                else  // if packer_buffer_size_bytes > 773760.00
                                {
                                    return BandwidthBucket(4);
                                }
                            }
                        }
                        else  // if packer_scatter_gather_num_tiles > 2.50
                        {
                            if (epoch_tiles <= 7936.00)
                            {
                                if (packer_buffer_size_bytes <= 433440.00)
                                {
                                    return BandwidthBucket(4);
                                }
                                else  // if packer_buffer_size_bytes > 433440.00
                                {
                                    if (packer_buffer_size_bytes <= 560000.00)
                                    {
                                        if (packer_num_phases <= 33.50)
                                        {
                                            if (kernel_clear_granularity <= 3.50)
                                            {
                                                return BandwidthBucket(5);
                                            }
                                            else  // if kernel_clear_granularity > 3.50
                                            {
                                                return BandwidthBucket(4);
                                            }
                                        }
                                        else  // if packer_num_phases > 33.50
                                        {
                                            return BandwidthBucket(4);
                                        }
                                    }
                                    else  // if packer_buffer_size_bytes > 560000.00
                                    {
                                        return BandwidthBucket(5);
                                    }
                                }
                            }
                            else  // if epoch_tiles > 7936.00
                            {
                                if (tile_size <= 1600.00)
                                {
                                    return BandwidthBucket(4);
                                }
                                else  // if tile_size > 1600.00
                                {
                                    if (packer_buffer_size_bytes <= 474240.00)
                                    {
                                        return BandwidthBucket(5);
                                    }
                                    else  // if packer_buffer_size_bytes > 474240.00
                                    {
                                        if (kernel_clear_granularity <= 2.50)
                                        {
                                            return BandwidthBucket(5);
                                        }
                                        else  // if kernel_clear_granularity > 2.50
                                        {
                                            return BandwidthBucket(4);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}


BandwidthBucket estimate_gather_connection(const int epoch_tiles,
                                           const int tile_size,
                                           const int packer_scatter_gather_num_tiles,
                                           const int consumer_fanin)
{
    if (epoch_tiles <= 3936.00)
    {
        if (tile_size <= 1600.00)
        {
            if (epoch_tiles <= 1984.00)
            {
                if (packer_scatter_gather_num_tiles <= 4.50)
                {
                    if (epoch_tiles <= 1312.00)
                    {
                        if (packer_scatter_gather_num_tiles <= 2.50)
                        {
                            if (epoch_tiles <= 992.00)
                            {
                                if (packer_scatter_gather_num_tiles <= 1.50)
                                {
                                    if (epoch_tiles <= 704.00)
                                    {
                                        return BandwidthBucket(0);
                                    }
                                    else  // if epoch_tiles > 704.00
                                    {
                                        if (consumer_fanin <= 3.50)
                                        {
                                            return BandwidthBucket(1);
                                        }
                                        else  // if consumer_fanin > 3.50
                                        {
                                            return BandwidthBucket(0);
                                        }
                                    }
                                }
                                else  // if packer_scatter_gather_num_tiles > 1.50
                                {
                                    return BandwidthBucket(1);
                                }
                            }
                            else  // if epoch_tiles > 992.00
                            {
                                return BandwidthBucket(1);
                            }
                        }
                        else  // if packer_scatter_gather_num_tiles > 2.50
                        {
                            if (consumer_fanin <= 3.50)
                            {
                                return BandwidthBucket(2);
                            }
                            else  // if consumer_fanin > 3.50
                            {
                                return BandwidthBucket(1);
                            }
                        }
                    }
                    else  // if epoch_tiles > 1312.00
                    {
                        if (consumer_fanin <= 3.50)
                        {
                            return BandwidthBucket(2);
                        }
                        else  // if consumer_fanin > 3.50
                        {
                            if (packer_scatter_gather_num_tiles <= 2.50)
                            {
                                return BandwidthBucket(1);
                            }
                            else  // if packer_scatter_gather_num_tiles > 2.50
                            {
                                return BandwidthBucket(1);
                            }
                        }
                    }
                }
                else  // if packer_scatter_gather_num_tiles > 4.50
                {
                    if (consumer_fanin <= 3.50)
                    {
                        return BandwidthBucket(2);
                    }
                    else  // if consumer_fanin > 3.50
                    {
                        if (epoch_tiles <= 1696.00)
                        {
                            return BandwidthBucket(1);
                        }
                        else  // if epoch_tiles > 1696.00
                        {
                            return BandwidthBucket(2);
                        }
                    }
                }
            }
            else  // if epoch_tiles > 1984.00
            {
                if (consumer_fanin <= 3.50)
                {
                    if (epoch_tiles <= 2784.00)
                    {
                        if (packer_scatter_gather_num_tiles <= 8.00)
                        {
                            return BandwidthBucket(2);
                        }
                        else  // if packer_scatter_gather_num_tiles > 8.00
                        {
                            return BandwidthBucket(3);
                        }
                    }
                    else  // if epoch_tiles > 2784.00
                    {
                        if (packer_scatter_gather_num_tiles <= 2.50)
                        {
                            return BandwidthBucket(3);
                        }
                        else  // if packer_scatter_gather_num_tiles > 2.50
                        {
                            return BandwidthBucket(3);
                        }
                    }
                }
                else  // if consumer_fanin > 3.50
                {
                    if (epoch_tiles <= 2976.00)
                    {
                        if (packer_scatter_gather_num_tiles <= 2.50)
                        {
                            if (epoch_tiles <= 2272.00)
                            {
                                return BandwidthBucket(2);
                            }
                            else  // if epoch_tiles > 2272.00
                            {
                                return BandwidthBucket(2);
                            }
                        }
                        else  // if packer_scatter_gather_num_tiles > 2.50
                        {
                            return BandwidthBucket(2);
                        }
                    }
                    else  // if epoch_tiles > 2976.00
                    {
                        if (packer_scatter_gather_num_tiles <= 1.50)
                        {
                            return BandwidthBucket(2);
                        }
                        else  // if packer_scatter_gather_num_tiles > 1.50
                        {
                            if (consumer_fanin <= 4.50)
                            {
                                return BandwidthBucket(2);
                            }
                            else  // if consumer_fanin > 4.50
                            {
                                return BandwidthBucket(2);
                            }
                        }
                    }
                }
            }
        }
        else  // if tile_size > 1600.00
        {
            if (epoch_tiles <= 1440.00)
            {
                if (packer_scatter_gather_num_tiles <= 4.50)
                {
                    if (packer_scatter_gather_num_tiles <= 3.50)
                    {
                        if (epoch_tiles <= 992.00)
                        {
                            if (packer_scatter_gather_num_tiles <= 1.50)
                            {
                                if (epoch_tiles <= 704.00)
                                {
                                    return BandwidthBucket(1);
                                }
                                else  // if epoch_tiles > 704.00
                                {
                                    return BandwidthBucket(1);
                                }
                            }
                            else  // if packer_scatter_gather_num_tiles > 1.50
                            {
                                return BandwidthBucket(2);
                            }
                        }
                        else  // if epoch_tiles > 992.00
                        {
                            return BandwidthBucket(2);
                        }
                    }
                    else  // if packer_scatter_gather_num_tiles > 3.50
                    {
                        return BandwidthBucket(2);
                    }
                }
                else  // if packer_scatter_gather_num_tiles > 4.50
                {
                    if (epoch_tiles <= 1088.00)
                    {
                        return BandwidthBucket(2);
                    }
                    else  // if epoch_tiles > 1088.00
                    {
                        return BandwidthBucket(3);
                    }
                }
            }
            else  // if epoch_tiles > 1440.00
            {
                if (epoch_tiles <= 2272.00)
                {
                    if (packer_scatter_gather_num_tiles <= 6.50)
                    {
                        if (epoch_tiles <= 1856.00)
                        {
                            if (consumer_fanin <= 3.50)
                            {
                                return BandwidthBucket(3);
                            }
                            else  // if consumer_fanin > 3.50
                            {
                                return BandwidthBucket(2);
                            }
                        }
                        else  // if epoch_tiles > 1856.00
                        {
                            if (packer_scatter_gather_num_tiles <= 1.50)
                            {
                                return BandwidthBucket(3);
                            }
                            else  // if packer_scatter_gather_num_tiles > 1.50
                            {
                                return BandwidthBucket(3);
                            }
                        }
                    }
                    else  // if packer_scatter_gather_num_tiles > 6.50
                    {
                        return BandwidthBucket(3);
                    }
                }
                else  // if epoch_tiles > 2272.00
                {
                    if (consumer_fanin <= 3.50)
                    {
                        if (consumer_fanin <= 2.50)
                        {
                            return BandwidthBucket(3);
                        }
                        else  // if consumer_fanin > 2.50
                        {
                            if (epoch_tiles <= 2784.00)
                            {
                                return BandwidthBucket(3);
                            }
                            else  // if epoch_tiles > 2784.00
                            {
                                if (packer_scatter_gather_num_tiles <= 1.50)
                                {
                                    return BandwidthBucket(3);
                                }
                                else  // if packer_scatter_gather_num_tiles > 1.50
                                {
                                    return BandwidthBucket(4);
                                }
                            }
                        }
                    }
                    else  // if consumer_fanin > 3.50
                    {
                        if (packer_scatter_gather_num_tiles <= 1.50)
                        {
                            if (epoch_tiles <= 2432.00)
                            {
                                return BandwidthBucket(3);
                            }
                            else  // if epoch_tiles > 2432.00
                            {
                                return BandwidthBucket(3);
                            }
                        }
                        else  // if packer_scatter_gather_num_tiles > 1.50
                        {
                            if (consumer_fanin <= 4.50)
                            {
                                return BandwidthBucket(3);
                            }
                            else  // if consumer_fanin > 4.50
                            {
                                return BandwidthBucket(3);
                            }
                        }
                    }
                }
            }
        }
    }
    else  // if epoch_tiles > 3936.00
    {
        if (tile_size <= 1600.00)
        {
            if (epoch_tiles <= 5952.00)
            {
                if (consumer_fanin <= 5.50)
                {
                    if (packer_scatter_gather_num_tiles <= 1.50)
                    {
                        if (consumer_fanin <= 3.50)
                        {
                            return BandwidthBucket(3);
                        }
                        else  // if consumer_fanin > 3.50
                        {
                            return BandwidthBucket(2);
                        }
                    }
                    else  // if packer_scatter_gather_num_tiles > 1.50
                    {
                        if (consumer_fanin <= 3.50)
                        {
                            return BandwidthBucket(3);
                        }
                        else  // if consumer_fanin > 3.50
                        {
                            return BandwidthBucket(3);
                        }
                    }
                }
                else  // if consumer_fanin > 5.50
                {
                    if (packer_scatter_gather_num_tiles <= 13.00)
                    {
                        return BandwidthBucket(2);
                    }
                    else  // if packer_scatter_gather_num_tiles > 13.00
                    {
                        return BandwidthBucket(3);
                    }
                }
            }
            else  // if epoch_tiles > 5952.00
            {
                if (packer_scatter_gather_num_tiles <= 34.00)
                {
                    if (packer_scatter_gather_num_tiles <= 1.50)
                    {
                        if (epoch_tiles <= 7424.00)
                        {
                            return BandwidthBucket(3);
                        }
                        else  // if epoch_tiles > 7424.00
                        {
                            if (consumer_fanin <= 7.50)
                            {
                                return BandwidthBucket(3);
                            }
                            else  // if consumer_fanin > 7.50
                            {
                                return BandwidthBucket(3);
                            }
                        }
                    }
                    else  // if packer_scatter_gather_num_tiles > 1.50
                    {
                        if (consumer_fanin <= 3.50)
                        {
                            return BandwidthBucket(3);
                        }
                        else  // if consumer_fanin > 3.50
                        {
                            if (epoch_tiles <= 9920.00)
                            {
                                return BandwidthBucket(3);
                            }
                            else  // if epoch_tiles > 9920.00
                            {
                                return BandwidthBucket(3);
                            }
                        }
                    }
                }
                else  // if packer_scatter_gather_num_tiles > 34.00
                {
                    if (consumer_fanin <= 5.50)
                    {
                        return BandwidthBucket(4);
                    }
                    else  // if consumer_fanin > 5.50
                    {
                        return BandwidthBucket(3);
                    }
                }
            }
        }
        else  // if tile_size > 1600.00
        {
            if (epoch_tiles <= 6336.00)
            {
                if (epoch_tiles <= 4704.00)
                {
                    if (consumer_fanin <= 4.50)
                    {
                        if (consumer_fanin <= 2.50)
                        {
                            return BandwidthBucket(3);
                        }
                        else  // if consumer_fanin > 2.50
                        {
                            if (consumer_fanin <= 3.50)
                            {
                                if (packer_scatter_gather_num_tiles <= 1.50)
                                {
                                    return BandwidthBucket(3);
                                }
                                else  // if packer_scatter_gather_num_tiles > 1.50
                                {
                                    return BandwidthBucket(4);
                                }
                            }
                            else  // if consumer_fanin > 3.50
                            {
                                return BandwidthBucket(3);
                            }
                        }
                    }
                    else  // if consumer_fanin > 4.50
                    {
                        return BandwidthBucket(3);
                    }
                }
                else  // if epoch_tiles > 4704.00
                {
                    if (packer_scatter_gather_num_tiles <= 1.50)
                    {
                        if (consumer_fanin <= 2.50)
                        {
                            return BandwidthBucket(3);
                        }
                        else  // if consumer_fanin > 2.50
                        {
                            return BandwidthBucket(3);
                        }
                    }
                    else  // if packer_scatter_gather_num_tiles > 1.50
                    {
                        if (consumer_fanin <= 2.50)
                        {
                            return BandwidthBucket(3);
                        }
                        else  // if consumer_fanin > 2.50
                        {
                            if (consumer_fanin <= 6.50)
                            {
                                return BandwidthBucket(4);
                            }
                            else  // if consumer_fanin > 6.50
                            {
                                if (epoch_tiles <= 5760.00)
                                {
                                    return BandwidthBucket(3);
                                }
                                else  // if epoch_tiles > 5760.00
                                {
                                    return BandwidthBucket(4);
                                }
                            }
                        }
                    }
                }
            }
            else  // if epoch_tiles > 6336.00
            {
                if (packer_scatter_gather_num_tiles <= 1.50)
                {
                    if (epoch_tiles <= 9920.00)
                    {
                        if (consumer_fanin <= 4.50)
                        {
                            return BandwidthBucket(4);
                        }
                        else  // if consumer_fanin > 4.50
                        {
                            return BandwidthBucket(4);
                        }
                    }
                    else  // if epoch_tiles > 9920.00
                    {
                        return BandwidthBucket(4);
                    }
                }
                else  // if packer_scatter_gather_num_tiles > 1.50
                {
                    if (packer_scatter_gather_num_tiles <= 34.00)
                    {
                        if (epoch_tiles <= 11904.00)
                        {
                            return BandwidthBucket(4);
                        }
                        else  // if epoch_tiles > 11904.00
                        {
                            return BandwidthBucket(4);
                        }
                    }
                    else  // if packer_scatter_gather_num_tiles > 34.00
                    {
                        if (epoch_tiles <= 12864.00)
                        {
                            return BandwidthBucket(4);
                        }
                        else  // if epoch_tiles > 12864.00
                        {
                            return BandwidthBucket(5);
                        }
                    }
                }
            }
        }
    }
}

BandwidthBucket estimate_forked_connection(const int epoch_tiles,
                                           const int tile_size,
                                           const int packer_buffer_size_bytes,
                                           const int packer_scatter_gather_num_tiles,
                                           const int producer_fanout)
{
    if (producer_fanout <= 4.50)
    {
        if (epoch_tiles <= 832.00)
        {
            if (producer_fanout <= 2.50)
            {
                if (epoch_tiles <= 352.00)
                {
                    return BandwidthBucket(0);
                }
                else  // if epoch_tiles > 352.00
                {
                    if (tile_size <= 1600.00)
                    {
                        if (epoch_tiles <= 608.00)
                        {
                            return BandwidthBucket(0);
                        }
                        else  // if epoch_tiles > 608.00
                        {
                            if (packer_scatter_gather_num_tiles <= 1.50)
                            {
                                return BandwidthBucket(0);
                            }
                            else  // if packer_scatter_gather_num_tiles > 1.50
                            {
                                return BandwidthBucket(1);
                            }
                        }
                    }
                    else  // if tile_size > 1600.00
                    {
                        return BandwidthBucket(1);
                    }
                }
            }
            else  // if producer_fanout > 2.50
            {
                if (epoch_tiles <= 608.00)
                {
                    if (packer_buffer_size_bytes <= 332800.00)
                    {
                        if (tile_size <= 1600.00)
                        {
                            return BandwidthBucket(0);
                        }
                        else  // if tile_size > 1600.00
                        {
                            if (epoch_tiles <= 480.00)
                            {
                                return BandwidthBucket(0);
                            }
                            else  // if epoch_tiles > 480.00
                            {
                                return BandwidthBucket(0);
                            }
                        }
                    }
                    else  // if packer_buffer_size_bytes > 332800.00
                    {
                        return BandwidthBucket(1);
                    }
                }
                else  // if epoch_tiles > 608.00
                {
                    if (tile_size <= 1600.00)
                    {
                        return BandwidthBucket(0);
                    }
                    else  // if tile_size > 1600.00
                    {
                        if (packer_buffer_size_bytes <= 79040.00)
                        {
                            if (producer_fanout <= 3.50)
                            {
                                return BandwidthBucket(1);
                            }
                            else  // if producer_fanout > 3.50
                            {
                                return BandwidthBucket(0);
                            }
                        }
                        else  // if packer_buffer_size_bytes > 79040.00
                        {
                            return BandwidthBucket(0);
                        }
                    }
                }
            }
        }
        else  // if epoch_tiles > 832.00
        {
            if (producer_fanout <= 2.50)
            {
                if (epoch_tiles <= 1216.00)
                {
                    if (epoch_tiles <= 1088.00)
                    {
                        return BandwidthBucket(1);
                    }
                    else  // if epoch_tiles > 1088.00
                    {
                        return BandwidthBucket(1);
                    }
                }
                else  // if epoch_tiles > 1216.00
                {
                    if (tile_size <= 1600.00)
                    {
                        if (epoch_tiles <= 2176.00)
                        {
                            return BandwidthBucket(1);
                        }
                        else  // if epoch_tiles > 2176.00
                        {
                            if (packer_buffer_size_bytes <= 67200.00)
                            {
                                return BandwidthBucket(2);
                            }
                            else  // if packer_buffer_size_bytes > 67200.00
                            {
                                return BandwidthBucket(1);
                            }
                        }
                    }
                    else  // if tile_size > 1600.00
                    {
                        if (packer_buffer_size_bytes <= 79040.00)
                        {
                            return BandwidthBucket(2);
                        }
                        else  // if packer_buffer_size_bytes > 79040.00
                        {
                            if (epoch_tiles <= 2496.00)
                            {
                                return BandwidthBucket(1);
                            }
                            else  // if epoch_tiles > 2496.00
                            {
                                if (packer_buffer_size_bytes <= 274560.00)
                                {
                                    return BandwidthBucket(2);
                                }
                                else  // if packer_buffer_size_bytes > 274560.00
                                {
                                    return BandwidthBucket(1);
                                }
                            }
                        }
                    }
                }
            }
            else  // if producer_fanout > 2.50
            {
                if (tile_size <= 1600.00)
                {
                    if (epoch_tiles <= 1664.00)
                    {
                        if (producer_fanout <= 3.50)
                        {
                            if (epoch_tiles <= 1440.00)
                            {
                                if (packer_scatter_gather_num_tiles <= 17.00)
                                {
                                    return BandwidthBucket(0);
                                }
                                else  // if packer_scatter_gather_num_tiles > 17.00
                                {
                                    return BandwidthBucket(1);
                                }
                            }
                            else  // if epoch_tiles > 1440.00
                            {
                                if (packer_buffer_size_bytes <= 120960.00)
                                {
                                    return BandwidthBucket(1);
                                }
                                else  // if packer_buffer_size_bytes > 120960.00
                                {
                                    return BandwidthBucket(0);
                                }
                            }
                        }
                        else  // if producer_fanout > 3.50
                        {
                            if (epoch_tiles <= 1312.00)
                            {
                                return BandwidthBucket(0);
                            }
                            else  // if epoch_tiles > 1312.00
                            {
                                if (packer_scatter_gather_num_tiles <= 2.50)
                                {
                                    return BandwidthBucket(0);
                                }
                                else  // if packer_scatter_gather_num_tiles > 2.50
                                {
                                    return BandwidthBucket(1);
                                }
                            }
                        }
                    }
                    else  // if epoch_tiles > 1664.00
                    {
                        if (packer_buffer_size_bytes <= 132160.00)
                        {
                            return BandwidthBucket(1);
                        }
                        else  // if packer_buffer_size_bytes > 132160.00
                        {
                            if (epoch_tiles <= 2496.00)
                            {
                                return BandwidthBucket(0);
                            }
                            else  // if epoch_tiles > 2496.00
                            {
                                return BandwidthBucket(1);
                            }
                        }
                    }
                }
                else  // if tile_size > 1600.00
                {
                    if (packer_buffer_size_bytes <= 108160.00)
                    {
                        return BandwidthBucket(1);
                    }
                    else  // if packer_buffer_size_bytes > 108160.00
                    {
                        if (epoch_tiles <= 1664.00)
                        {
                            if (producer_fanout <= 3.50)
                            {
                                if (epoch_tiles <= 1216.00)
                                {
                                    return BandwidthBucket(0);
                                }
                                else  // if epoch_tiles > 1216.00
                                {
                                    return BandwidthBucket(1);
                                }
                            }
                            else  // if producer_fanout > 3.50
                            {
                                if (epoch_tiles <= 1088.00)
                                {
                                    return BandwidthBucket(0);
                                }
                                else  // if epoch_tiles > 1088.00
                                {
                                    return BandwidthBucket(0);
                                }
                            }
                        }
                        else  // if epoch_tiles > 1664.00
                        {
                            return BandwidthBucket(1);
                        }
                    }
                }
            }
        }
    }
    else  // if producer_fanout > 4.50
    {
        if (producer_fanout <= 5.50)
        {
            if (epoch_tiles <= 1440.00)
            {
                if (tile_size <= 1600.00)
                {
                    return BandwidthBucket(0);
                }
                else  // if tile_size > 1600.00
                {
                    if (epoch_tiles <= 928.00)
                    {
                        return BandwidthBucket(0);
                    }
                    else  // if epoch_tiles > 928.00
                    {
                        return BandwidthBucket(0);
                    }
                }
            }
            else  // if epoch_tiles > 1440.00
            {
                if (tile_size <= 1600.00)
                {
                    if (epoch_tiles <= 2176.00)
                    {
                        return BandwidthBucket(0);
                    }
                    else  // if epoch_tiles > 2176.00
                    {
                        if (packer_buffer_size_bytes <= 67200.00)
                        {
                            return BandwidthBucket(1);
                        }
                        else  // if packer_buffer_size_bytes > 67200.00
                        {
                            return BandwidthBucket(0);
                        }
                    }
                }
                else  // if tile_size > 1600.00
                {
                    if (packer_buffer_size_bytes <= 93600.00)
                    {
                        if (packer_scatter_gather_num_tiles <= 1.50)
                        {
                            return BandwidthBucket(1);
                        }
                        else  // if packer_scatter_gather_num_tiles > 1.50
                        {
                            return BandwidthBucket(1);
                        }
                    }
                    else  // if packer_buffer_size_bytes > 93600.00
                    {
                        if (epoch_tiles <= 2176.00)
                        {
                            return BandwidthBucket(0);
                        }
                        else  // if epoch_tiles > 2176.00
                        {
                            return BandwidthBucket(1);
                        }
                    }
                }
            }
        }
        else  // if producer_fanout > 5.50
        {
            if (epoch_tiles <= 1856.00)
            {
                if (packer_scatter_gather_num_tiles <= 27.00)
                {
                    return BandwidthBucket(0);
                }
                else  // if packer_scatter_gather_num_tiles > 27.00
                {
                    return BandwidthBucket(0);
                }
            }
            else  // if epoch_tiles > 1856.00
            {
                if (producer_fanout <= 6.50)
                {
                    if (tile_size <= 1600.00)
                    {
                        return BandwidthBucket(0);
                    }
                    else  // if tile_size > 1600.00
                    {
                        if (packer_buffer_size_bytes <= 180960.00)
                        {
                            if (epoch_tiles <= 2496.00)
                            {
                                return BandwidthBucket(0);
                            }
                            else  // if epoch_tiles > 2496.00
                            {
                                if (packer_buffer_size_bytes <= 93600.00)
                                {
                                    return BandwidthBucket(1);
                                }
                                else  // if packer_buffer_size_bytes > 93600.00
                                {
                                    return BandwidthBucket(0);
                                }
                            }
                        }
                        else  // if packer_buffer_size_bytes > 180960.00
                        {
                            return BandwidthBucket(0);
                        }
                    }
                }
                else  // if producer_fanout > 6.50
                {
                    return BandwidthBucket(0);
                }
            }
        }
    }
}

} // namespace balancer
} // namespace tt
