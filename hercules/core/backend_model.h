
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/

#pragma once

#include <memory>
#include <string>
#include "backend_manager.h"
#include "filesystem.h"
#include "infer_request.h"
#include "model.h"
#include "hercules/proto/model_config.pb.h"
#include "status.h"
#include "flare/base/result_status.h"

namespace hercules::core {

class inference_server;
class TritonModelInstance;

//
// Represents a model.
//
// Inheriting from Model to implement backend APIs
//
class TritonModel : public Model {
 public:
  static Status Create(
      inference_server* server, const std::string& model_path,
      const hercules::common::BackendCmdlineConfigMap& backend_cmdline_config_map,
      const hercules::common::HostPolicyCmdlineConfigMap& host_policy_map,
      const std::string& model_name, const int64_t version,
      const hercules::proto::ModelConfig& model_config,
      std::unique_ptr<TritonModel>* model);
  ~TritonModel();

  const std::string& LocalizedModelPath() const
  {
    return localized_model_dir_->Path();
  }
  inference_server* Server() { return server_; }
  bool AutoCompleteConfig() const { return auto_complete_config_; }
  Status UpdateModelConfig(
      const uint32_t config_version,
      TRITONSERVER_Message* updated_config_message);
  const std::shared_ptr<hercules_backend>& Backend() const { return backend_; }
  const std::vector<std::unique_ptr<TritonModelInstance>>& Instances() const
  {
    return instances_;
  }
  void* State() { return state_; }
  void SetState(void* state) { state_ = state; }
  Status AddInstance(
      std::unique_ptr<TritonModelInstance>&& instance, const bool passive);

 private:
  FLARE_DISALLOW_COPY_AND_ASSIGN(TritonModel);

  TritonModel(
      inference_server* server,
      const std::shared_ptr<LocalizedDirectory>& localized_model_dir,
      const std::shared_ptr<hercules_backend>& backend,
      const double min_compute_capability, const int64_t version,
      const hercules::proto::ModelConfig& config, const bool auto_complete_config);

  // Set the scheduler based on the model configuration. The scheduler
  // can only be set once for a backend.
  Status SetConfiguredScheduler();

  // Merges the global backend configs with the specific
  // backend configs.
  static Status ResolveBackendConfigs(
      const hercules::common::BackendCmdlineConfigMap& backend_cmdline_config_map,
      const std::string& backend_name,
      hercules::common::BackendCmdlineConfig& config);

  // Sets defaults for some backend configurations when none are specified on
  // the command line.
  static Status SetBackendConfigDefaults(
      hercules::common::BackendCmdlineConfig& config);

  Status Initialize();
  Status WarmUp();

  // The server object that owns this model. The model holds this as a
  // raw pointer because the lifetime of the server is guaranteed to
  // be longer than the lifetime of a model owned by the server.
  inference_server* server_;

  // The minimum supported compute capability on device.
  const double min_compute_capability_;

  // Whether the backend should attempt to auto-complete the model config.
  const bool auto_complete_config_;

  // The localized repo directory holding the model. If localization
  // required creation of a temporary local copy then that copy will
  // persist as along as this object is retained by this model.
  std::shared_ptr<LocalizedDirectory> localized_model_dir_;

  // Backend used by this model.
  std::shared_ptr<hercules_backend> backend_;

  // The model instances for this model.
  std::vector<std::unique_ptr<TritonModelInstance>> instances_;
  std::vector<std::unique_ptr<TritonModelInstance>> passive_instances_;

  // Opaque state associated with this model.
  void* state_;
};

}  // namespace hercules::core
