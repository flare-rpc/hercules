
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/

#pragma once

#include <NvInfer.h>
#include <memory>
#include <string>
#include <vector>
#include "triton/core/tritonserver.h"

namespace triton { namespace backend { namespace tensorrt {

/// Load a TensorRT plan from a serialized plan file and return the
/// corresponding runtime and engine. It is the caller's
/// responsibility to destroy any returned runtime or engine object
/// even if an error is returned.
///
/// \param plan_path The path to the model plan file.
/// \param dla_core_id The DLA core to use for this runtime. Does not
/// use DLA when set to -1.
/// \param runtime Returns the IRuntime object, or nullptr if failed
/// to create.
/// \param engine Returns the ICudaEngine object, or nullptr if failed
/// to create.
/// \return Error status.
TRITONSERVER_Error* LoadPlan(
    const std::string& plan_path, const int64_t dla_core_id,
    std::shared_ptr<nvinfer1::IRuntime>* runtime,
    std::shared_ptr<nvinfer1::ICudaEngine>* engine);

}}}  // namespace triton::backend::tensorrt
