
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/

//
#pragma once

#include <map>
#include <memory>
#include <mutex>
#include "status.h"

namespace hercules::core {

    // This is a singleton class responsible for maintaining CUDA memory pool
    // used by the inference server. CUDA memory allocations and deallocations
    // must be requested via functions provided by this class.
    class cuda_memory_manager {
    public:
        // Options to configure CUDA memory manager.
        struct Options {
            Options(double cc = 6.0, const std::map<int, uint64_t> &s = {})
                    : min_supported_compute_capability_(cc), memory_pool_byte_size_(s) {
            }

            // The minimum compute capability of the supported devices.
            double min_supported_compute_capability_;

            // The size of CUDA memory reserved for the specified devices.
            // The memory size will be rounded up to align with
            // the default granularity (512 bytes).
            // No memory will be reserved for devices that is not listed.
            std::map<int, uint64_t> memory_pool_byte_size_;
        };

        ~cuda_memory_manager();

        // Create the memory manager based on 'options' specified.
        // Return Status object indicating success or failure.
        static Status Create(const Options &options);

        // Allocate CUDA memory on GPU 'device_id' with
        // the requested 'size' and return the pointer in 'ptr'.
        // Return Status object indicating success or failure.
        static Status Alloc(void **ptr, uint64_t size, int64_t device_id);

        // Free the memory allocated by the memory manager on 'device_id'.
        // Return Status object indicating success or failure.
        static Status Free(void *ptr, int64_t device_id);

    protected:
        // Provide explicit control on the lifecycle of the CUDA memory manager,
        // for testing only.
        static void Reset();

    private:
        cuda_memory_manager(bool has_allocation) : has_allocation_(has_allocation) {}

        bool has_allocation_;
        static std::unique_ptr<cuda_memory_manager> instance_;
        static std::mutex instance_mu_;
    };

}  // namespace hercules::core
