
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/

#pragma once

#include <stdint.h>
#include "hercules/proto/model_config.pb.h"

namespace hercules::core {

/// Get the CUDA stream priority for a given ModelPriority
/// \param priority The hercules::proto::ModelOptimizationPolicy::ModelPriority
/// priority. \param cuda_stream_priority Returns the CUDA stream priority.
/// \return The error status.
int GetCudaStreamPriority(
    hercules::proto::ModelOptimizationPolicy::ModelPriority priority);

}  // namespace hercules::core
