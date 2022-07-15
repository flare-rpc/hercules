
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/


#include "loader.h"

#include <NvInferPlugin.h>
#include <memory>
#include <mutex>
#include "logging.h"
#include "triton/backend/backend_common.h"

namespace triton { namespace backend { namespace tensorrt {

TRITONSERVER_Error*
LoadPlan(
    const std::string& plan_path, const int64_t dla_core_id,
    std::shared_ptr<nvinfer1::IRuntime>* runtime,
    std::shared_ptr<nvinfer1::ICudaEngine>* engine)
{
  // Create runtime only if it is not provided
  if (*runtime == nullptr) {
    runtime->reset(nvinfer1::createInferRuntime(tensorrt_logger));
    if (*runtime == nullptr) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL, "unable to create TensorRT runtime");
    }
  }

  // Report error if 'dla_core_id' >= number of DLA cores
  if (dla_core_id != -1) {
    auto dla_core_count = (*runtime)->getNbDLACores();
    if (dla_core_id < dla_core_count) {
      (*runtime)->setDLACore(dla_core_id);
    } else {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("unable to create TensorRT runtime with DLA Core ID: ") +
           std::to_string(dla_core_id) +
           ", available number of DLA cores: " + std::to_string(dla_core_count))
              .c_str());
    }
  }

  std::string model_data_str;
  RETURN_IF_ERROR(ReadTextFile(plan_path, &model_data_str));
  std::vector<char> model_data(model_data_str.begin(), model_data_str.end());

  engine->reset(
      (*runtime)->deserializeCudaEngine(&model_data[0], model_data.size()));
  if (*engine == nullptr) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL, "unable to create TensorRT engine");
  }

  return nullptr;
}

}}}  // namespace triton::backend::tensorrt
