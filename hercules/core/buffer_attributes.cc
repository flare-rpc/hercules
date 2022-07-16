
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/


#include "hercules/core/buffer_attributes.h"

#include <cstring>
#include "constants.h"

namespace hercules::core {
void
buffer_attributes::SetByteSize(const size_t& byte_size)
{
  byte_size_ = byte_size;
}

void
buffer_attributes::SetMemoryType(const TRITONSERVER_MemoryType& memory_type)
{
  memory_type_ = memory_type;
}

void
buffer_attributes::SetMemoryTypeId(const int64_t& memory_type_id)
{
  memory_type_id_ = memory_type_id;
}

void
buffer_attributes::SetCudaIpcHandle(void* cuda_ipc_handle)
{
  char* lcuda_ipc_handle = reinterpret_cast<char*>(cuda_ipc_handle);
  cuda_ipc_handle_.clear();
  std::copy(
      lcuda_ipc_handle, lcuda_ipc_handle + CUDA_IPC_STRUCT_SIZE,
      std::back_inserter(cuda_ipc_handle_));
}

void*
buffer_attributes::CudaIpcHandle()
{
  if (cuda_ipc_handle_.empty()) {
    return nullptr;
  } else {
    return reinterpret_cast<void*>(cuda_ipc_handle_.data());
  }
}

size_t
buffer_attributes::ByteSize() const
{
  return byte_size_;
}

TRITONSERVER_MemoryType
buffer_attributes::MemoryType() const
{
  return memory_type_;
}

int64_t
buffer_attributes::MemoryTypeId() const
{
  return memory_type_id_;
}

buffer_attributes::buffer_attributes(
    size_t byte_size, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id, char* cuda_ipc_handle)
    : byte_size_(byte_size), memory_type_(memory_type),
      memory_type_id_(memory_type_id)
{
  // cuda ipc handle size
  cuda_ipc_handle_.reserve(CUDA_IPC_STRUCT_SIZE);

  if (cuda_ipc_handle != nullptr) {
    std::copy(
        cuda_ipc_handle, cuda_ipc_handle + CUDA_IPC_STRUCT_SIZE,
        std::back_inserter(cuda_ipc_handle_));
  }
}
}  // namespace hercules::core
