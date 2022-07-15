
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/

//
#pragma once

#include <boost/interprocess/managed_external_buffer.hpp>
#include <map>
#include <memory>
#include <mutex>
#include "status.h"
#include "hercules/common/model_config.h"

namespace hercules::core {

// This is a singleton class responsible for maintaining pinned memory pool
// used by the inference server. Pinned memory allocations and deallocations
// must be requested via functions provided by this class.
class PinnedMemoryManager {
 public:
  // Options to configure pinned memeory manager.
  struct Options {
    Options(
        uint64_t b = 0,
        const hercules::common::HostPolicyCmdlineConfigMap& host_policy_map = {})
        : pinned_memory_pool_byte_size_(b), host_policy_map_(host_policy_map)
    {
    }

    uint64_t pinned_memory_pool_byte_size_;
    hercules::common::HostPolicyCmdlineConfigMap host_policy_map_;
  };

  ~PinnedMemoryManager();

  // Create the pinned memory manager based on 'options' specified.
  // Return Status object indicating success or failure.
  static Status Create(const Options& options);

  // Allocate pinned memory with the requested 'size' and return the pointer
  // in 'ptr'. If 'allow_nonpinned_fallback' is true, regular system memory
  // will be allocated as fallback in the case where pinned memory fails to
  // be allocated.
  // Return Status object indicating success or failure.
  static Status Alloc(
      void** ptr, uint64_t size, TRITONSERVER_MemoryType* allocated_type,
      bool allow_nonpinned_fallback);

  // Free the memory allocated by the pinned memory manager.
  // Return Status object indicating success or failure.
  static Status Free(void* ptr);

 protected:
  // Provide explicit control on the lifecycle of the CUDA memory manager,
  // for testing only.
  static void Reset();

 private:
  class PinnedMemory {
   public:
    PinnedMemory(void* pinned_memory_buffer, uint64_t size);
    ~PinnedMemory();
    void* pinned_memory_buffer_;
    std::mutex buffer_mtx_;
    boost::interprocess::managed_external_buffer managed_pinned_memory_;
  };

  PinnedMemoryManager() = default;

  Status AllocInternal(
      void** ptr, uint64_t size, TRITONSERVER_MemoryType* allocated_type,
      bool allow_nonpinned_fallback, PinnedMemory* pinned_memory_buffer);
  Status FreeInternal(void* ptr);
  void AddPinnedMemoryBuffer(
      const std::shared_ptr<PinnedMemory>& pinned_memory_buffer,
      unsigned long node_mask);

  static std::unique_ptr<PinnedMemoryManager> instance_;
  static uint64_t pinned_memory_byte_size_;

  std::mutex info_mtx_;
  std::map<void*, std::pair<bool, PinnedMemory*>> memory_info_;
  std::map<unsigned long, std::shared_ptr<PinnedMemory>> pinned_memory_buffers_;
};

}  // namespace hercules::core
