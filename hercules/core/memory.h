
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/

#pragma once

#include <vector>
#include "buffer_attributes.h"
#include "constants.h"
#include "status.h"

namespace hercules::core {

//
// Memory used to access data in inference requests
//
class Memory {
 public:
  // Get the 'idx'-th data block in the buffer. Using index to avoid
  // maintaining internal state such that one buffer can be shared
  // across multiple providers.
  // 'idx' zero base index. Valid indices are continuous.
  // 'byte_size' returns the byte size of the chunk of bytes.
  // 'memory_type' returns the memory type of the chunk of bytes.
  // 'memory_type_id' returns the memory type id of the chunk of bytes.
  // Return the pointer to the data block. Returns nullptr if 'idx' is
  // out of range
  virtual const char* BufferAt(
      size_t idx, size_t* byte_size, TRITONSERVER_MemoryType* memory_type,
      int64_t* memory_type_id) const = 0;

  // Similar to the above BufferAt but with BufferAttributes.
  virtual const char* BufferAt(
      size_t idx, BufferAttributes** buffer_attributes) = 0;

  // Get the number of contiguous buffers composing the memory.
  size_t BufferCount() const { return buffer_count_; }

  // Return the total byte size of the data buffer
  size_t TotalByteSize() const { return total_byte_size_; }

 protected:
  Memory() : total_byte_size_(0), buffer_count_(0) {}
  size_t total_byte_size_;
  size_t buffer_count_;
};

//
// MemoryReference
//
class MemoryReference : public Memory {
 public:
  // Create a read-only data buffer as a reference to other data buffer
  MemoryReference();

  //\see Memory::BufferAt()
  const char* BufferAt(
      size_t idx, size_t* byte_size, TRITONSERVER_MemoryType* memory_type,
      int64_t* memory_type_id) const override;

  const char* BufferAt(
      size_t idx, BufferAttributes** buffer_attributes) override;

  // Add a 'buffer' with 'byte_size' as part of this data buffer
  // Return the index of the buffer
  size_t AddBuffer(
      const char* buffer, size_t byte_size, TRITONSERVER_MemoryType memory_type,
      int64_t memory_type_id);

  size_t AddBuffer(const char* buffer, BufferAttributes* buffer_attributes);

  // Add a 'buffer' with 'byte_size' as part of this data buffer in the front
  // Return the index of the buffer
  size_t AddBufferFront(
      const char* buffer, size_t byte_size, TRITONSERVER_MemoryType memory_type,
      int64_t memory_type_id);

 private:
  struct Block {
    Block(
        const char* buffer, size_t byte_size,
        TRITONSERVER_MemoryType memory_type, int64_t memory_type_id)
        : buffer_(buffer), buffer_attributes_(BufferAttributes(
                               byte_size, memory_type, memory_type_id, nullptr))
    {
    }

    Block(const char* buffer, BufferAttributes* buffer_attributes)
        : buffer_(buffer), buffer_attributes_(*buffer_attributes)
    {
    }
    const char* buffer_;
    BufferAttributes buffer_attributes_;
  };
  std::vector<Block> buffer_;
};

//
// MutableMemory
//
class MutableMemory : public Memory {
 public:
  // Create a mutable data buffer referencing to other data buffer.
  MutableMemory(
      char* buffer, size_t byte_size, TRITONSERVER_MemoryType memory_type,
      int64_t memory_type_id);

  virtual ~MutableMemory() {}

  //\see Memory::BufferAt()
  const char* BufferAt(
      size_t idx, size_t* byte_size, TRITONSERVER_MemoryType* memory_type,
      int64_t* memory_type_id) const override;

  //\see Memory::BufferAt()
  const char* BufferAt(
      size_t idx, BufferAttributes** buffer_attributes) override;

  // Return a pointer to the base address of the mutable buffer. If
  // non-null 'memory_type' returns the memory type of the chunk of
  // bytes. If non-null 'memory_type_id' returns the memory type id of
  // the chunk of bytes.
  char* MutableBuffer(
      TRITONSERVER_MemoryType* memory_type = nullptr,
      int64_t* memory_type_id = nullptr);

  DISALLOW_COPY_AND_ASSIGN(MutableMemory);

 protected:
  MutableMemory() : Memory() {}

  char* buffer_;
  BufferAttributes buffer_attributes_;
};

//
// AllocatedMemory
//
class AllocatedMemory : public MutableMemory {
 public:
  // Create a continuous data buffer with 'byte_size', 'memory_type' and
  // 'memory_type_id'. Note that the buffer may be created on different memeory
  // type and memory type id if the original request type and id can not be
  // satisfied, thus the function caller should always check the actual memory
  // type and memory type id before use.
  AllocatedMemory(
      size_t byte_size, TRITONSERVER_MemoryType memory_type,
      int64_t memory_type_id);

  ~AllocatedMemory() override;
};

}  // namespace hercules::core
