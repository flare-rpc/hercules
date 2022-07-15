
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/


#include "kernel.h"

#include <cuda.h>

#define THREADBLOCK_SIZE 512
__launch_bounds__(THREADBLOCK_SIZE) __global__ void TritonGatherKernel(
    const int8_t** __restrict input_ptr_buffer,
    const size_t* __restrict byte_size_buffer,
    const size_t* __restrict byte_size_offset_buffer,
    int8_t* __restrict output_buffer)
{
  int request_idx = blockIdx.x;
  int lane_id = threadIdx.x;
  const int8_t* request_input_buffer = input_ptr_buffer[request_idx];
  int byte_size = byte_size_buffer[request_idx];
  int byte_size_offset = byte_size_offset_buffer[request_idx];

  int8_t* output_buffer_with_offset = output_buffer + byte_size_offset;
  if (((byte_size % 4) == 0) && (((uint64_t)request_input_buffer % 4) == 0) &&
      (((uint64_t)output_buffer_with_offset % 4) == 0)) {
    int32_t* input_4 = (int32_t*)request_input_buffer;
    int32_t* output_4 = (int32_t*)output_buffer_with_offset;
    int element_count = byte_size / 4;
    for (int elem_id = lane_id; elem_id < element_count;
         elem_id += THREADBLOCK_SIZE) {
      output_4[elem_id] = input_4[elem_id];
    }
  } else {
    for (int elem_id = lane_id; elem_id < byte_size;
         elem_id += THREADBLOCK_SIZE) {
      output_buffer_with_offset[elem_id] =
          __ldg(request_input_buffer + elem_id);
    }
  }
}

#ifdef __cplusplus
extern "C" {
#endif

cudaError_t
RunGatherKernel(
    const int8_t** input_ptr_buffer, const size_t* byte_size_buffer,
    const size_t* byte_size_offset_buffer, int8_t* output_buffer,
    size_t request_count, cudaStream_t stream)
{
  TritonGatherKernel<<<request_count, THREADBLOCK_SIZE, 0, stream>>>(
      input_ptr_buffer, byte_size_buffer, byte_size_offset_buffer,
      output_buffer);
  return cudaGetLastError();
}

#ifdef __cplusplus
}
#endif
