
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/


#include "memory.h"

#include "pinned_memory_manager.h"
#include "hercules/common/logging.h"

#ifdef HERCULES_ENABLE_GPU
#include <cuda_runtime_api.h>
#include "cuda_memory_manager.h"
#endif  // HERCULES_ENABLE_GPU

namespace hercules::core {

//
// memory_reference
//
memory_reference::memory_reference() : memory_base() {}

const char*
memory_reference::buffer_at(
    size_t idx, size_t* byte_size, TRITONSERVER_MemoryType* memory_type,
    int64_t* memory_type_id) const
{
  if (idx >= buffer_.size()) {
    *byte_size = 0;
    *memory_type = TRITONSERVER_MEMORY_CPU;
    *memory_type_id = 0;
    return nullptr;
  }
  *memory_type = buffer_[idx].buffer_attributes_.MemoryType();
  *memory_type_id = buffer_[idx].buffer_attributes_.MemoryTypeId();
  *byte_size = buffer_[idx].buffer_attributes_.ByteSize();
  return buffer_[idx].buffer_;
}

const char*
memory_reference::buffer_at(size_t idx, buffer_attributes** attr)
{
  if (idx >= buffer_.size()) {
    *attr = nullptr;
    return nullptr;
  }

  *attr = &(buffer_[idx].buffer_attributes_);
  return buffer_[idx].buffer_;
}

size_t
memory_reference::add_buffer(
    const char* buffer, size_t byte_size, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id)
{
  total_byte_size_ += byte_size;
  buffer_count_++;
  buffer_.emplace_back(buffer, byte_size, memory_type, memory_type_id);
  return buffer_.size() - 1;
}

size_t
memory_reference::add_buffer(
    const char* buffer, buffer_attributes* attr)
{
  total_byte_size_ += attr->ByteSize();
  buffer_count_++;
  buffer_.emplace_back(buffer, attr);
  return buffer_.size() - 1;
}

size_t
memory_reference::add_buffer_front(
    const char* buffer, size_t byte_size, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id)
{
  total_byte_size_ += byte_size;
  buffer_count_++;
  buffer_.emplace(
      buffer_.begin(), buffer, byte_size, memory_type, memory_type_id);
  return buffer_.size() - 1;
}

//
// mutable_memory
//
mutable_memory::mutable_memory(
    char* buffer, size_t byte_size, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id)
    : memory_base(), buffer_(buffer),
      buffer_attributes_(
          buffer_attributes(byte_size, memory_type, memory_type_id, nullptr))
{
  total_byte_size_ = byte_size;
  buffer_count_ = (byte_size == 0) ? 0 : 1;
}

const char*
mutable_memory::buffer_at(
    size_t idx, size_t* byte_size, TRITONSERVER_MemoryType* memory_type,
    int64_t* memory_type_id) const
{
  if (idx != 0) {
    *byte_size = 0;
    *memory_type = TRITONSERVER_MEMORY_CPU;
    *memory_type_id = 0;
    return nullptr;
  }
  *byte_size = total_byte_size_;
  *memory_type = buffer_attributes_.MemoryType();
  *memory_type_id = buffer_attributes_.MemoryTypeId();
  return buffer_;
}

const char*
mutable_memory::buffer_at(size_t idx, buffer_attributes** attr)
{
  if (idx != 0) {
    *attr = nullptr;
    return nullptr;
  }

  *attr = &buffer_attributes_;
  return buffer_;
}

char*
mutable_memory::mutable_buffer(
    TRITONSERVER_MemoryType* memory_type, int64_t* memory_type_id)
{
  if (memory_type != nullptr) {
    *memory_type = buffer_attributes_.MemoryType();
  }
  if (memory_type_id != nullptr) {
    *memory_type_id = buffer_attributes_.MemoryTypeId();
  }

  return buffer_;
}

//
// allocated_memory
//
allocated_memory::allocated_memory(
    size_t byte_size, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id)
    : mutable_memory(nullptr, byte_size, memory_type, memory_type_id)
{
  if (total_byte_size_ != 0) {
    // Allocate memory with the following fallback policy:
    // CUDA memory -> pinned system memory -> non-pinned system memory
    switch (buffer_attributes_.MemoryType()) {
#ifdef HERCULES_ENABLE_GPU
      case TRITONSERVER_MEMORY_GPU: {
        auto status = cuda_memory_manager::Alloc(
            (void**)&buffer_, total_byte_size_,
            buffer_attributes_.MemoryTypeId());
        if (!status.IsOk()) {
          static bool warning_logged = false;
          if (!warning_logged) {
            FLARE_LOG(WARNING) << status.Message()
                        << ", falling back to pinned system memory";
            warning_logged = true;
          }

          goto pinned_memory_allocation;
        }
        break;
      }
      pinned_memory_allocation:
#endif  // HERCULES_ENABLE_GPU
      default: {
        TRITONSERVER_MemoryType memory_type = buffer_attributes_.MemoryType();
        auto status = PinnedMemoryManager::Alloc(
            (void**)&buffer_, total_byte_size_, &memory_type, true);
        buffer_attributes_.SetMemoryType(memory_type);
        if (!status.IsOk()) {
          FLARE_LOG(ERROR) << status.Message();
          buffer_ = nullptr;
        }
        break;
      }
    }
  }
  total_byte_size_ = (buffer_ == nullptr) ? 0 : total_byte_size_;
}

allocated_memory::~allocated_memory()
{
  if (buffer_ != nullptr) {
    switch (buffer_attributes_.MemoryType()) {
      case TRITONSERVER_MEMORY_GPU: {
#ifdef HERCULES_ENABLE_GPU
        auto status =
            cuda_memory_manager::Free(buffer_, buffer_attributes_.MemoryTypeId());
        if (!status.IsOk()) {
          FLARE_LOG(ERROR) << status.Message();
        }
#endif  // HERCULES_ENABLE_GPU
        break;
      }

      default: {
        auto status = PinnedMemoryManager::Free(buffer_);
        if (!status.IsOk()) {
          FLARE_LOG(ERROR) << status.Message();
          buffer_ = nullptr;
        }
        break;
      }
    }
    buffer_ = nullptr;
  }
}

}  // namespace hercules::core
