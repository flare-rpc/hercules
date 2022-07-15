
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/

#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include "constants.h"
#include "server_message.h"
#include "status.h"
#include "hercules/common/model_config.h"
#include "tritonserver_apis.h"

namespace hercules::core {

//
// Proxy to a backend shared library.
//
class TritonBackend {
 public:
  typedef TRITONSERVER_Error* (*TritonModelInitFn_t)(
      TRITONBACKEND_Model* model);
  typedef TRITONSERVER_Error* (*TritonModelFiniFn_t)(
      TRITONBACKEND_Model* model);
  typedef TRITONSERVER_Error* (*TritonModelInstanceInitFn_t)(
      TRITONBACKEND_ModelInstance* instance);
  typedef TRITONSERVER_Error* (*TritonModelInstanceFiniFn_t)(
      TRITONBACKEND_ModelInstance* instance);
  typedef TRITONSERVER_Error* (*TritonModelInstanceExecFn_t)(
      TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests,
      const uint32_t request_cnt);

  static Status Create(
      const std::string& name, const std::string& dir,
      const std::string& libpath,
      const triton::common::BackendCmdlineConfig& backend_cmdline_config,
      std::shared_ptr<TritonBackend>* backend);
  ~TritonBackend();

  const std::string& Name() const { return name_; }
  const std::string& Directory() const { return dir_; }
  const TritonServerMessage& BackendConfig() const { return backend_config_; }

  TRITONBACKEND_ExecutionPolicy ExecutionPolicy() const { return exec_policy_; }
  void SetExecutionPolicy(const TRITONBACKEND_ExecutionPolicy policy)
  {
    exec_policy_ = policy;
  }

  void* State() { return state_; }
  void SetState(void* state) { state_ = state; }

  TritonModelInitFn_t ModelInitFn() const { return model_init_fn_; }
  TritonModelFiniFn_t ModelFiniFn() const { return model_fini_fn_; }
  TritonModelInstanceInitFn_t ModelInstanceInitFn() const
  {
    return inst_init_fn_;
  }
  TritonModelInstanceFiniFn_t ModelInstanceFiniFn() const
  {
    return inst_fini_fn_;
  }
  TritonModelInstanceExecFn_t ModelInstanceExecFn() const
  {
    return inst_exec_fn_;
  }

 private:
  typedef TRITONSERVER_Error* (*TritonBackendInitFn_t)(
      TRITONBACKEND_Backend* backend);
  typedef TRITONSERVER_Error* (*TritonBackendFiniFn_t)(
      TRITONBACKEND_Backend* backend);

  TritonBackend(
      const std::string& name, const std::string& dir,
      const std::string& libpath, const TritonServerMessage& backend_config);

  void ClearHandles();
  Status LoadBackendLibrary();

  // The name of the backend.
  const std::string name_;

  // Full path to the directory holding backend shared library and
  // other artifacts.
  const std::string dir_;

  // Full path to the backend shared library.
  const std::string libpath_;

  // Backend configuration as JSON
  TritonServerMessage backend_config_;

  // dlopen / dlsym handles
  void* dlhandle_;
  TritonBackendInitFn_t backend_init_fn_;
  TritonBackendFiniFn_t backend_fini_fn_;
  TritonModelInitFn_t model_init_fn_;
  TritonModelFiniFn_t model_fini_fn_;
  TritonModelInstanceInitFn_t inst_init_fn_;
  TritonModelInstanceFiniFn_t inst_fini_fn_;
  TritonModelInstanceExecFn_t inst_exec_fn_;

  // Execution policy
  TRITONBACKEND_ExecutionPolicy exec_policy_;

  // Opaque state associated with the backend.
  void* state_;
};

//
// Manage communication with Triton backends and their lifecycle.
//
class TritonBackendManager {
 public:
  static Status Create(std::shared_ptr<TritonBackendManager>* manager);

  Status CreateBackend(
      const std::string& name, const std::string& dir,
      const std::string& libpath,
      const triton::common::BackendCmdlineConfig& backend_cmdline_config,
      std::shared_ptr<TritonBackend>* backend);

  Status BackendState(
      std::unique_ptr<
          std::unordered_map<std::string, std::vector<std::string>>>*
          backend_state);

 private:
  DISALLOW_COPY_AND_ASSIGN(TritonBackendManager);
  TritonBackendManager() = default;
  std::unordered_map<std::string, std::shared_ptr<TritonBackend>> backend_map_;
};

}  // namespace hercules::core
