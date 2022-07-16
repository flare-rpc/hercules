
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
    class hercules_backend {
    public:
        typedef TRITONSERVER_Error *(*hercules_model_init_func)(
                TRITONBACKEND_Model *model);

        typedef TRITONSERVER_Error *(*hercules_model_fini_func)(
                TRITONBACKEND_Model *model);

        typedef TRITONSERVER_Error *(*hercules_model_instance_init_func)(
                TRITONBACKEND_ModelInstance *instance);

        typedef TRITONSERVER_Error *(*hercules_model_instance_fini_func)(
                TRITONBACKEND_ModelInstance *instance);

        typedef TRITONSERVER_Error *(*hercules_model_instance_exec_func)(
                TRITONBACKEND_ModelInstance *instance, TRITONBACKEND_Request **requests,
                const uint32_t request_cnt);

        static Status Create(
                const std::string &name, const std::string &dir,
                const std::string &libpath,
                const hercules::common::BackendCmdlineConfig &backend_cmdline_config,
                std::shared_ptr<hercules_backend> *backend);

        ~hercules_backend();

        const std::string &Name() const { return name_; }

        const std::string &Directory() const { return dir_; }

        const TritonServerMessage &BackendConfig() const { return backend_config_; }

        TRITONBACKEND_ExecutionPolicy ExecutionPolicy() const { return exec_policy_; }

        void SetExecutionPolicy(const TRITONBACKEND_ExecutionPolicy policy) {
            exec_policy_ = policy;
        }

        void *State() { return state_; }

        void SetState(void *state) { state_ = state; }

        hercules_model_init_func ModelInitFn() const { return model_init_fn_; }

        hercules_model_fini_func ModelFiniFn() const { return model_fini_fn_; }

        hercules_model_instance_init_func ModelInstanceInitFn() const {
            return inst_init_fn_;
        }

        hercules_model_instance_fini_func ModelInstanceFiniFn() const {
            return inst_fini_fn_;
        }

        hercules_model_instance_exec_func ModelInstanceExecFn() const {
            return inst_exec_fn_;
        }

    private:
        typedef TRITONSERVER_Error *(*hercules_backend_init_func)(
                TRITONBACKEND_Backend *backend);

        typedef TRITONSERVER_Error *(*hercules_backend_fini_func)(
                TRITONBACKEND_Backend *backend);

        hercules_backend(
                const std::string &name, const std::string &dir,
                const std::string &libpath, const TritonServerMessage &backend_config);

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
        void *dlhandle_;
        hercules_backend_init_func backend_init_fn_;
        hercules_backend_fini_func backend_fini_fn_;
        hercules_model_init_func model_init_fn_;
        hercules_model_fini_func model_fini_fn_;
        hercules_model_instance_init_func inst_init_fn_;
        hercules_model_instance_fini_func inst_fini_fn_;
        hercules_model_instance_exec_func inst_exec_fn_;

        // Execution policy
        TRITONBACKEND_ExecutionPolicy exec_policy_;

        // Opaque state associated with the backend.
        void *state_;
    };

    //
    // Manage communication with Triton backends and their lifecycle.
    //
    class TritonBackendManager {
    public:
        static Status Create(std::shared_ptr<TritonBackendManager> *manager);

        Status CreateBackend(
                const std::string &name, const std::string &dir,
                const std::string &libpath,
                const hercules::common::BackendCmdlineConfig &backend_cmdline_config,
                std::shared_ptr<hercules_backend> *backend);

        Status BackendState(
                std::unique_ptr<
                        std::unordered_map<std::string, std::vector<std::string>>> *
                backend_state);

    private:
        FLARE_DISALLOW_COPY_AND_ASSIGN(TritonBackendManager);

        TritonBackendManager() = default;

        std::unordered_map<std::string, std::shared_ptr<hercules_backend>> backend_map_;
    };

}  // namespace hercules::core
