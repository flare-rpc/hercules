
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/

#pragma once

#include <set>
#include "status.h"
#include "hercules/common/sync_queue.h"

#ifdef HERCULES_ENABLE_GPU
#include <cuda_runtime_api.h>
#endif  // HERCULES_ENABLE_GPU

namespace hercules::core {

#ifdef HERCULES_ENABLE_GPU
#define RETURN_IF_CUDA_ERR(X, MSG)                                           \
  do {                                                                       \
    cudaError_t err__ = (X);                                                 \
    if (err__ != cudaSuccess) {                                              \
      return Status(                                                         \
          Status::Code::INTERNAL, (MSG) + ": " + cudaGetErrorString(err__)); \
    }                                                                        \
  } while (false)
#endif  // HERCULES_ENABLE_GPU

#ifndef HERCULES_ENABLE_GPU
using cudaStream_t = void*;
#endif  // !HERCULES_ENABLE_GPU

/// Enable peer access for all GPU device pairs
/// \param min_compute_capability The minimum support CUDA compute
/// capability.
/// \return The error status. A non-OK status means not all pairs are enabled
Status EnablePeerAccess(const double min_compute_capability);

/// Copy buffer from 'src' to 'dst' for given 'byte_size'. The buffer location
/// is identified by the memory type and id, and the corresponding copy will be
/// initiated.
/// \param msg The message to be prepended in error message.
/// \param src_memory_type The memory type CPU/GPU of the source.
/// \param src_memory_type_id The device id of the source.
/// \param dst_memory_type The memory type CPU/GPU of the destination.
/// \param dst_memory_type_id The device id of the destination.
/// \param byte_size The size in bytes to me copied from source to destination.
/// \param src The buffer start address of the source.
/// \param dst The buffer start address of the destination.
/// \param cuda_stream The stream to be associated with, and 0 can be
/// passed for default stream.
/// \param cuda_used returns whether a CUDA memory copy is initiated. If true,
/// the caller should synchronize on the given 'cuda_stream' to ensure data copy
/// is completed.
/// \param copy_on_stream whether the memory copies should be performed in cuda
/// host functions on the 'cuda_stream'.
/// \return The error status. A non-ok status indicates failure to copy the
/// buffer.
Status CopyBuffer(
    const std::string& msg, const TRITONSERVER_MemoryType src_memory_type,
    const int64_t src_memory_type_id,
    const TRITONSERVER_MemoryType dst_memory_type,
    const int64_t dst_memory_type_id, const size_t byte_size, const void* src,
    void* dst, cudaStream_t cuda_stream, bool* cuda_used,
    bool copy_on_stream = false);

#ifdef HERCULES_ENABLE_GPU
/// Validates the compute capability of the GPU indexed
/// \param gpu_id The index of the target GPU.
/// \param min_compute_capability The minimum support CUDA compute
/// capability.
/// \return The error status. A non-OK status means the target GPU is
/// not supported.
Status CheckGPUCompatibility(
    const int gpu_id, const double min_compute_capability);

/// Obtains a set of gpu ids that is supported by triton.
/// \param supported_gpus Returns the set of integers which is
///  populated by ids of supported GPUS
/// \param min_compute_capability The minimum support CUDA compute
/// capability.
/// \return The error status. A non-ok status means there were
/// errors encountered while querying GPU devices.
Status GetSupportedGPUs(
    std::set<int>* supported_gpus, const double min_compute_capability);

/// Checks if the GPU specified is an integrated GPU and supports Zero-copy.
/// \param gpu_id The index of the target GPU.
/// \param zero_copy_support If true, Zero-copy is supported by this GPU.
/// \return The error status. A non-OK status means the target GPU is
/// not supported.
Status SupportsIntegratedZeroCopy(const int gpu_id, bool* zero_copy_support);
#endif

// Helper around CopyBuffer that updates the completion queue with the returned
// status and cuda_used flag.
void CopyBufferHandler(
    const std::string& msg, const TRITONSERVER_MemoryType src_memory_type,
    const int64_t src_memory_type_id,
    const TRITONSERVER_MemoryType dst_memory_type,
    const int64_t dst_memory_type_id, const size_t byte_size, const void* src,
    void* dst, cudaStream_t cuda_stream, void* response_ptr,
    hercules::common::SyncQueue<std::tuple<Status, bool, void*>>*
        completion_queue);

struct CopyParams {
  CopyParams(void* dst, const void* src, const size_t byte_size)
      : dst_(dst), src_(src), byte_size_(byte_size)
  {
  }

  void* dst_;
  const void* src_;
  const size_t byte_size_;
};

}  // namespace hercules::core
