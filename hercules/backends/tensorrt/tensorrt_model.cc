
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/


#include "tensorrt_model.h"

namespace hercules::backend { namespace tensorrt {

TensorRTModel::Priority
ParsePriority(const std::string& priority)
{
  TensorRTModel::Priority trt_priority = TensorRTModel::Priority::DEFAULT;

  if (priority.compare("PRIORITY_MAX") == 0) {
    trt_priority = TensorRTModel::Priority::MAX;
  } else if (priority.compare("PRIORITY_MIN") == 0) {
    trt_priority = TensorRTModel::Priority::MIN;
  } else if (priority.compare("PRIORITY_DEFAULT") != 0) {
    LOG_MESSAGE(
        TRITONSERVER_LOG_WARN,
        (std::string(
             "TRT backend does not support the provided stream priority '") +
         priority + "', using 'PRIORITY_DEFAULT'.")
            .c_str());
  }

  return trt_priority;
}

TensorRTModel::TensorRTModel(TRITONBACKEND_Model* triton_model)
    : BackendModel(triton_model), priority_(Priority::DEFAULT),
      use_cuda_graphs_(false), gather_kernel_buffer_threshold_(0),
      separate_output_stream_(false), eager_batching_(false),
      busy_wait_events_(false)
{
  ParseModelConfig();
}

TRITONSERVER_Error*
TensorRTModel::SetTensorRTModelConfig()
{
  RETURN_IF_ERROR(SetModelConfig());
  ParseModelConfig();

  return nullptr;
}

void
TensorRTModel::ParseModelConfig()
{
  hercules::common::json_parser::Value optimization;
  if (model_config_.Find("optimization", &optimization)) {
    optimization.MemberAsUInt(
        "gather_kernel_buffer_threshold", &gather_kernel_buffer_threshold_);
    optimization.MemberAsBool("eager_batching", &eager_batching_);
    std::string priority;
    optimization.MemberAsString("priority", &priority);
    priority_ = ParsePriority(priority);
    hercules::common::json_parser::Value cuda;
    if (optimization.Find("cuda", &cuda)) {
      cuda.MemberAsBool("graphs", &use_cuda_graphs_);
      cuda.MemberAsBool("busy_wait_events", &busy_wait_events_);
      cuda.MemberAsArray("graph_spec", &graph_specs_);
      cuda.MemberAsBool("output_copy_stream", &separate_output_stream_);
    }
  }
}

}}}  // namespace hercules::backend::tensorrt
