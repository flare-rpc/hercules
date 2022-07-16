
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/

#pragma once

#include "tensorrt_model.h"
#include "hercules/backend/backend_model_instance.h"

#include <set>

namespace hercules::backend { namespace tensorrt {

class TensorRTModelInstance : public BackendModelInstance {
 public:
  TensorRTModelInstance(
      TensorRTModel* tensorrt_model,
      TRITONBACKEND_ModelInstance* triton_model_instance);
  virtual ~TensorRTModelInstance() = default;

  void Initialize();
  TensorRTModel* Model() { return tensorrt_model_; }
  std::set<std::string>& ProfileNames() { return profile_names_; }
  int64_t DLACoreId() { return dla_core_id_; }

 protected:
  TensorRTModel* tensorrt_model_;
  std::set<std::string> profile_names_;
  int64_t dla_core_id_;
};

}}}  // namespace hercules::backend::tensorrt
