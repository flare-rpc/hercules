
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/


#include "hercules/backend/backend_model.h"

#include "hercules/backend/backend_common.h"

namespace hercules::backend {

//
// BackendModel
//
BackendModel::BackendModel(
    TRITONBACKEND_Model* triton_model, const bool allow_optional)
    : triton_model_(triton_model), allow_optional_(allow_optional)
{
  const char* model_name;
  THROW_IF_BACKEND_MODEL_ERROR(
      TRITONBACKEND_ModelName(triton_model, &model_name));
  name_ = model_name;

  THROW_IF_BACKEND_MODEL_ERROR(
      TRITONBACKEND_ModelVersion(triton_model, &version_));

  const char* repository_path = nullptr;
  TRITONBACKEND_ArtifactType repository_artifact_type;
  THROW_IF_BACKEND_MODEL_ERROR(TRITONBACKEND_ModelRepository(
      triton_model, &repository_artifact_type, &repository_path));
  if (repository_artifact_type != TRITONBACKEND_ARTIFACT_FILESYSTEM) {
    throw BackendModelException(TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        (std::string("unsupported repository artifact type for model '") +
         model_name + "'")
            .c_str()));
  }
  repository_path_ = repository_path;

  THROW_IF_BACKEND_MODEL_ERROR(
      TRITONBACKEND_ModelServer(triton_model, &triton_server_));
  TRITONBACKEND_Backend* backend;
  THROW_IF_BACKEND_MODEL_ERROR(
      TRITONBACKEND_ModelBackend(triton_model, &backend));
  THROW_IF_BACKEND_MODEL_ERROR(
      TRITONBACKEND_BackendMemoryManager(backend, &triton_memory_manager_));

  THROW_IF_BACKEND_MODEL_ERROR(ParseModelConfig());
}

TRITONSERVER_Error*
BackendModel::ParseModelConfig()
{
  TRITONSERVER_Message* config_message;
  RETURN_IF_ERROR(TRITONBACKEND_ModelConfig(
      triton_model_, 1 /* config_version */, &config_message));

  // Get the model configuration as a json string from
  // config_message. We use json_parser, which is a wrapper that
  // returns nice errors (currently the underlying implementation is
  // rapidjson... but others could be added).
  const char* buffer;
  size_t byte_size;
  RETURN_IF_ERROR(
      TRITONSERVER_MessageSerializeToJson(config_message, &buffer, &byte_size));

  TRITONSERVER_Error* err = model_config_.Parse(buffer, byte_size);
  RETURN_IF_ERROR(TRITONSERVER_MessageDelete(config_message));
  RETURN_IF_ERROR(err);

  int64_t mbs = 0;
  RETURN_IF_ERROR(model_config_.MemberAsInt("max_batch_size", &mbs));
  max_batch_size_ = mbs;

  enable_pinned_input_ = false;
  enable_pinned_output_ = false;
  {
    common::json_parser::Value optimization;
    if (model_config_.Find("optimization", &optimization)) {
      common::json_parser::Value pinned_memory;
      if (optimization.Find("input_pinned_memory", &pinned_memory)) {
        RETURN_IF_ERROR(
            pinned_memory.MemberAsBool("enable", &enable_pinned_input_));
      }
      if (optimization.Find("output_pinned_memory", &pinned_memory)) {
        RETURN_IF_ERROR(
            pinned_memory.MemberAsBool("enable", &enable_pinned_output_));
      }
    }
  }

  RETURN_IF_ERROR(
      BatchInput::ParseFromModelConfig(model_config_, &batch_inputs_));
  RETURN_IF_ERROR(
      BatchOutput::ParseFromModelConfig(model_config_, &batch_outputs_));
  for (const auto& batch_output : batch_outputs_) {
    for (const auto& name : batch_output.TargetNames()) {
      batch_output_map_.emplace(name, &batch_output);
    }
  }
  hercules::common::json_parser::Value config_inputs;
  RETURN_IF_ERROR(model_config_.MemberAsArray("input", &config_inputs));
  for (size_t i = 0; i < config_inputs.ArraySize(); i++) {
    hercules::common::json_parser::Value io;
    RETURN_IF_ERROR(config_inputs.IndexAsObject(i, &io));
    std::string io_name;
    RETURN_IF_ERROR(io.MemberAsString("name", &io_name));
    hercules::common::json_parser::Value input_property_json;
    bool allow_ragged_batch = false;
    if (io.Find("allow_ragged_batch", &input_property_json)) {
      RETURN_IF_ERROR(input_property_json.AsBool(&allow_ragged_batch));
    }
    if (allow_ragged_batch) {
      ragged_inputs_.emplace(io_name);
    }
    bool optional = false;
    if (io.Find("optional", &input_property_json)) {
      RETURN_IF_ERROR(input_property_json.AsBool(&optional));
    }
    if (optional) {
      if (allow_optional_) {
        optional_inputs_.emplace(io_name);
      } else {
        RETURN_IF_ERROR(TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string("'optional' is set to true for input '") + io_name +
             "' while the backend model doesn't support optional input")
                .c_str()));
      }
    }
  }

  return nullptr;
}

TRITONSERVER_Error*
BackendModel::SetModelConfig()
{
  hercules::common::json_parser::WriteBuffer json_buffer;
  RETURN_IF_ERROR(ModelConfig().Write(&json_buffer));

  TRITONSERVER_Message* message;
  RETURN_IF_ERROR(TRITONSERVER_MessageNewFromSerializedJson(
      &message, json_buffer.Base(), json_buffer.Size()));
  RETURN_IF_ERROR(TRITONBACKEND_ModelSetConfig(
      triton_model_, 1 /* config_version */, message));

  // Triton core can normalize the missing config settings
  // in the above call. We must retrieve the updated model
  // configration from the core.
  RETURN_IF_ERROR(ParseModelConfig());

  return nullptr;
}

TRITONSERVER_Error*
BackendModel::SupportsFirstDimBatching(bool* supports)
{
  *supports = max_batch_size_ > 0;
  return nullptr;
}

const BatchOutput*
BackendModel::FindBatchOutput(const std::string& output_name) const
{
  const auto it = batch_output_map_.find(output_name);
  return ((it == batch_output_map_.end()) ? nullptr : it->second);
}

}   // namespace hercules::backend
