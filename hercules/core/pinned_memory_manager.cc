
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/

//

#include "pinned_memory_manager.h"

#include <sstream>
#include "numa_utils.h"
#include "hercules/common/logging.h"

#ifdef HERCULES_ENABLE_GPU
#include <cuda_runtime_api.h>
#endif  // HERCULES_ENABLE_GPU

namespace hercules::core {

    namespace {

        std::string
        PointerToString(void *ptr) {
            std::stringstream ss;
            ss << ptr;
            return ss.str();
        }

        Status
        ParseIntOption(const std::string &msg, const std::string &arg, int *value) {
            try {
                *value = std::stoi(arg);
            }
            catch (const std::invalid_argument &ia) {
                return Status(
                        Status::Code::INVALID_ARG,
                        msg + ": Can't parse '" + arg + "' to integer");
            }
            return Status::Success;
        }

    }  // namespace

    std::unique_ptr<PinnedMemoryManager> PinnedMemoryManager::instance_;
    uint64_t PinnedMemoryManager::pinned_memory_byte_size_;

    PinnedMemoryManager::PinnedMemory::PinnedMemory(
            void *pinned_memory_buffer, uint64_t size)
            : pinned_memory_buffer_(pinned_memory_buffer) {
        if (pinned_memory_buffer_ != nullptr) {
            managed_pinned_memory_ = boost::interprocess::managed_external_buffer(
                    boost::interprocess::create_only_t{}, pinned_memory_buffer_, size);
        }
    }


    PinnedMemoryManager::PinnedMemory::~PinnedMemory() {
#ifdef HERCULES_ENABLE_GPU
        if (pinned_memory_buffer_ != nullptr) {
          cudaFreeHost(pinned_memory_buffer_);
        }
#endif  // HERCULES_ENABLE_GPU
    }

    PinnedMemoryManager::~PinnedMemoryManager() {
        // Clean up
        for (const auto &memory_info : memory_info_) {
            const auto &is_pinned = memory_info.second.first;
            if (!is_pinned) {
                free(memory_info.first);
            }
        }
    }

    void
    PinnedMemoryManager::AddPinnedMemoryBuffer(
            const std::shared_ptr<PinnedMemory> &pinned_memory_buffer,
            unsigned long node_mask) {
        pinned_memory_buffers_[node_mask] = pinned_memory_buffer;
    }

    Status
    PinnedMemoryManager::AllocInternal(
            void **ptr, uint64_t size, TRITONSERVER_MemoryType *allocated_type,
            bool allow_nonpinned_fallback, PinnedMemory *pinned_memory_buffer) {
        auto status = Status::Success;
        if (pinned_memory_buffer->pinned_memory_buffer_ != nullptr) {
            std::lock_guard<std::mutex> lk(pinned_memory_buffer->buffer_mtx_);
            *ptr = pinned_memory_buffer->managed_pinned_memory_.allocate(
                    size, std::nothrow_t{});
            *allocated_type = TRITONSERVER_MEMORY_CPU_PINNED;
            if (*ptr == nullptr) {
                status = Status(
                        Status::Code::INTERNAL, "failed to allocate pinned system memory");
            }
        } else {
            status = Status(
                    Status::Code::INTERNAL,
                    "failed to allocate pinned system memory: no pinned memory pool");
        }

        bool is_pinned = true;
        if ((!status.IsOk()) && allow_nonpinned_fallback) {
            static bool warning_logged = false;
            if (!warning_logged) {
                FLARE_LOG(WARNING) << status.Message()
                            << ", falling back to non-pinned system memory";
                warning_logged = true;
            }
            *ptr = malloc(size);
            *allocated_type = TRITONSERVER_MEMORY_CPU;
            is_pinned = false;
            if (*ptr == nullptr) {
                status = Status(
                        Status::Code::INTERNAL,
                        "failed to allocate non-pinned system memory");
            } else {
                status = Status::Success;
            }
        }

        // keep track of allocated buffer or clean up
        {
            std::lock_guard<std::mutex> lk(info_mtx_);
            if (status.IsOk()) {
                auto res = memory_info_.emplace(
                        *ptr, std::make_pair(is_pinned, pinned_memory_buffer));
                if (!res.second) {
                    status = Status(
                            Status::Code::INTERNAL, "unexpected memory address collision, '" +
                                                    PointerToString(*ptr) +
                                                    "' has been managed");
                }
                FLARE_LOG(DEBUG) << (is_pinned ? "" : "non-")
                               << "pinned memory allocation: "
                               << "size " << size << ", addr " << *ptr;
            }
        }

        if ((!status.IsOk()) && (*ptr != nullptr)) {
            if (is_pinned) {
                std::lock_guard<std::mutex> lk(pinned_memory_buffer->buffer_mtx_);
                pinned_memory_buffer->managed_pinned_memory_.deallocate(*ptr);
            } else {
                free(*ptr);
            }
        }

        return status;
    }

    Status
    PinnedMemoryManager::FreeInternal(void *ptr) {
        bool is_pinned = true;
        PinnedMemory *pinned_memory_buffer = nullptr;
        {
            std::lock_guard<std::mutex> lk(info_mtx_);
            auto it = memory_info_.find(ptr);
            if (it != memory_info_.end()) {
                is_pinned = it->second.first;
                pinned_memory_buffer = it->second.second;
                FLARE_LOG(DEBUG) << (is_pinned ? "" : "non-")
                               << "pinned memory deallocation: "
                               << "addr " << ptr;
                memory_info_.erase(it);
            } else {
                return Status(
                        Status::Code::INTERNAL, "unexpected memory address '" +
                                                PointerToString(ptr) +
                                                "' is not being managed");
            }
        }

        if (is_pinned) {
            std::lock_guard<std::mutex> lk(pinned_memory_buffer->buffer_mtx_);
            pinned_memory_buffer->managed_pinned_memory_.deallocate(ptr);
        } else {
            free(ptr);
        }
        return Status::Success;
    }

    void
    PinnedMemoryManager::Reset() {
        instance_.reset();
    }

    Status
    PinnedMemoryManager::Create(const Options &options) {
        if (instance_ != nullptr) {
            FLARE_LOG(WARNING) << "New pinned memory pool of size "
                        << options.pinned_memory_pool_byte_size_
                        << " could not be created since one already exists"
                        << " of size " << pinned_memory_byte_size_;
            return Status::Success;
        }

        instance_.reset(new PinnedMemoryManager());
        if (options.host_policy_map_.empty()) {
            void *buffer = nullptr;
#ifdef HERCULES_ENABLE_GPU
            auto err = cudaHostAlloc(
                &buffer, options.pinned_memory_pool_byte_size_, cudaHostAllocPortable);
            if (err != cudaSuccess) {
              buffer = nullptr;
              FLARE_LOG(WARNING) << "Unable to allocate pinned system memory, pinned memory "
                             "pool will not be available: "
                          << std::string(cudaGetErrorString(err));
            } else if (options.pinned_memory_pool_byte_size_ != 0) {
              FLARE_LOG(INFO) << "Pinned memory pool is created at '"
                       << PointerToString(buffer) << "' with size "
                       << options.pinned_memory_pool_byte_size_;
            } else {
              FLARE_LOG(INFO) << "Pinned memory pool disabled";
            }
#endif  // HERCULES_ENABLE_GPU
            instance_->AddPinnedMemoryBuffer(
                    std::shared_ptr<PinnedMemory>(
                            new PinnedMemory(buffer, options.pinned_memory_pool_byte_size_)),
                    0);
        } else {
            // Create only one buffer / manager should be created for one node,
            // and all associated devices should request memory from the shared manager
            std::map<int32_t, std::string> numa_map;
            for (const auto &host_policy : options.host_policy_map_) {
                const auto numa_it = host_policy.second.find("numa-node");
                if (numa_it != host_policy.second.end()) {
                    int32_t numa_id;
                    if (ParseIntOption("Parsing NUMA node", numa_it->second, &numa_id)
                            .IsOk()) {
                        numa_map.emplace(numa_id, host_policy.first);
                    }
                }
            }
            for (const auto &node_policy : numa_map) {
                auto status =
                        SetNumaMemoryPolicy(options.host_policy_map_.at(node_policy.second));
                if (!status.IsOk()) {
                    FLARE_LOG(WARNING) << "Unable to allocate pinned system memory for NUMA node "
                                << node_policy.first << ": " << status.AsString();
                    continue;
                }
                unsigned long node_mask;
                status = GetNumaMemoryPolicyNodeMask(&node_mask);
                if (!status.IsOk()) {
                    FLARE_LOG(WARNING) << "Unable to get NUMA node set for current thread: "
                                << status.AsString();
                    continue;
                }
                void *buffer = nullptr;
#ifdef HERCULES_ENABLE_GPU
                auto err = cudaHostAlloc(
                    &buffer, options.pinned_memory_pool_byte_size_,
                    cudaHostAllocPortable);
                if (err != cudaSuccess) {
                  buffer = nullptr;
                  FLARE_LOG(WARNING) << "Unable to allocate pinned system memory, pinned memory "
                                 "pool will not be available: "
                              << std::string(cudaGetErrorString(err));
                } else if (options.pinned_memory_pool_byte_size_ != 0) {
                  FLARE_LOG(INFO) << "Pinned memory pool is created at '"
                           << PointerToString(buffer) << "' with size "
                           << options.pinned_memory_pool_byte_size_;
                } else {
                  FLARE_LOG(INFO) << "Pinned memory pool disabled";
                }
#endif  // HERCULES_ENABLE_GPU
                ResetNumaMemoryPolicy();
                instance_->AddPinnedMemoryBuffer(
                        std::shared_ptr<PinnedMemory>(
                                new PinnedMemory(buffer, options.pinned_memory_pool_byte_size_)),
                        node_mask);
            }
            // If no pinned memory is allocated, add an empty entry where all allocation
            // will be on noraml system memory
            if (instance_->pinned_memory_buffers_.empty()) {
                instance_->AddPinnedMemoryBuffer(
                        std::shared_ptr<PinnedMemory>(
                                new PinnedMemory(nullptr, options.pinned_memory_pool_byte_size_)),
                        0);
            }
        }
        pinned_memory_byte_size_ = options.pinned_memory_pool_byte_size_;
        return Status::Success;
    }

    Status
    PinnedMemoryManager::Alloc(
            void **ptr, uint64_t size, TRITONSERVER_MemoryType *allocated_type,
            bool allow_nonpinned_fallback) {
        if (instance_ == nullptr) {
            return Status(
                    Status::Code::UNAVAILABLE, "PinnedMemoryManager has not been created");
        }

        auto pinned_memory_buffer =
                instance_->pinned_memory_buffers_.begin()->second.get();
        if (instance_->pinned_memory_buffers_.size() > 1) {
            unsigned long node_mask;
            if (GetNumaMemoryPolicyNodeMask(&node_mask).IsOk()) {
                auto it = instance_->pinned_memory_buffers_.find(node_mask);
                if (it != instance_->pinned_memory_buffers_.end()) {
                    pinned_memory_buffer = it->second.get();
                }
            }
        }

        return instance_->AllocInternal(
                ptr, size, allocated_type, allow_nonpinned_fallback,
                pinned_memory_buffer);
    }

    Status
    PinnedMemoryManager::Free(void *ptr) {
        if (instance_ == nullptr) {
            return Status(
                    Status::Code::UNAVAILABLE, "PinnedMemoryManager has not been created");
        }

        return instance_->FreeInternal(ptr);
    }

}  // namespace hercules::core
