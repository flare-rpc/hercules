
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/


#pragma once
#include <cuda_runtime_api.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

cudaError_t RunGatherKernel(
    const int8_t** input_ptr_buffer, const size_t* byte_size_buffer,
    const size_t* byte_size_offset_buffer, int8_t* output_buffer,
    size_t request_count, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
