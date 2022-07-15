
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/


#include "tensorrt_model_instance.h"

namespace triton { namespace backend { namespace tensorrt {

TensorRTModelInstance::TensorRTModelInstance(
    TensorRTModel* tensorrt_model,
    TRITONBACKEND_ModelInstance* triton_model_instance)
    : BackendModelInstance(tensorrt_model, triton_model_instance),
      tensorrt_model_(tensorrt_model)
{
  uint32_t profile_count;
  THROW_IF_BACKEND_INSTANCE_ERROR(TRITONBACKEND_ModelInstanceProfileCount(
      triton_model_instance, &profile_count));
  for (uint32_t index = 0; index < profile_count; index++) {
    const char* profile_name;
    THROW_IF_BACKEND_INSTANCE_ERROR(TRITONBACKEND_ModelInstanceProfileName(
        triton_model_instance, index, &profile_name));
    profile_names_.insert(profile_name);
  }
  uint32_t secondary_device_count;
  THROW_IF_BACKEND_INSTANCE_ERROR(
      TRITONBACKEND_ModelInstanceSecondaryDeviceCount(
          triton_model_instance, &secondary_device_count));
  if (secondary_device_count != 0) {
    if (secondary_device_count != 1) {
      THROW_IF_BACKEND_INSTANCE_ERROR(TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (Name() + " of model " + tensorrt_model->Name() +
           " must have either zero or or one secondary devices")
              .c_str()));
    }
    const char* secondary_device_kind;
    int64_t secondary_device_id;
    THROW_IF_BACKEND_INSTANCE_ERROR(
        TRITONBACKEND_ModelInstanceSecondaryDeviceProperties(
            triton_model_instance, 0 /* index */, &secondary_device_kind,
            &secondary_device_id));

    if (strcmp(secondary_device_kind, "KIND_NVDLA") != 0) {
      THROW_IF_BACKEND_INSTANCE_ERROR(TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("secondary device for ") + Name() + " of model " +
           tensorrt_model->Name() + " must be KIND_NVDLA")
              .c_str()));
    } else {
      dla_core_id_ = secondary_device_id;
    }
  } else {
    dla_core_id_ = -1;
  }
}

}}}  // namespace triton::backend::tensorrt
