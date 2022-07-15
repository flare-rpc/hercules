
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/


#ifdef HERCULES_ENABLE_GPU
#include "model_config_cuda.h"

#include <cuda_runtime_api.h>

namespace hercules::core {

int
GetCudaStreamPriority(
    hercules::proto::ModelOptimizationPolicy::ModelPriority priority)
{
  // Default priority is 0
  int cuda_stream_priority = 0;

  int min, max;
  cudaError_t cuerr = cudaDeviceGetStreamPriorityRange(&min, &max);
  if ((cuerr != cudaErrorNoDevice) && (cuerr != cudaSuccess)) {
    return 0;
  }

  switch (priority) {
    case hercules::proto::ModelOptimizationPolicy::PRIORITY_MAX:
      cuda_stream_priority = max;
      break;
    case hercules::proto::ModelOptimizationPolicy::PRIORITY_MIN:
      cuda_stream_priority = min;
      break;
    default:
      cuda_stream_priority = 0;
      break;
  }

  return cuda_stream_priority;
}

}  // namespace hercules::core
#endif  // HERCULES_ENABLE_GPU