
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/

#pragma once

#include "triton/backend/backend_model.h"

namespace triton { namespace backend { namespace tensorrt {

class TensorRTModel : public BackendModel {
 public:
  TensorRTModel(TRITONBACKEND_Model* triton_model);
  virtual ~TensorRTModel() = default;

  TRITONSERVER_Error* SetTensorRTModelConfig();

  void ParseModelConfig();

  // The model configuration.
  common::TritonJson::Value& GraphSpecs() { return graph_specs_; }

  enum Priority { DEFAULT = 0, MIN = 1, MAX = 2 };
  Priority ModelPriority() { return priority_; }
  bool UseCudaGraphs() { return use_cuda_graphs_; }
  size_t GatherKernelBufferThreshold()
  {
    return gather_kernel_buffer_threshold_;
  }
  bool SeparateOutputStream() { return separate_output_stream_; }
  bool EagerBatching() { return eager_batching_; }
  bool BusyWaitEvents() { return busy_wait_events_; }

 protected:
  common::TritonJson::Value graph_specs_;
  Priority priority_;
  bool use_cuda_graphs_;
  size_t gather_kernel_buffer_threshold_;
  bool separate_output_stream_;
  bool eager_batching_;
  bool busy_wait_events_;
};

}}}  // namespace triton::backend::tensorrt
