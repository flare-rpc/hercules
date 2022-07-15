
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/


#include "backend_manager.h"

#include "backend_memory_manager.h"
#include "server_message.h"
#include "shared_library.h"
#include "hercules/common/logging.h"

// For unknown reason, windows will not export the TRITONBACKEND_*
// functions declared with dllexport in tritonbackend.h. To get those
// functions exported it is (also?) necessary to mark the definitions
// in this file with dllexport as well.
#if defined(_MSC_VER)
#define TRITONAPI_DECLSPEC __declspec(dllexport)
#elif defined(__GNUC__)
#define TRITONAPI_DECLSPEC __attribute__((__visibility__("default")))
#else
#define TRITONAPI_DECLSPEC
#endif

namespace hercules::core {

//
// TritonBackend
//
Status
TritonBackend::Create(
    const std::string& name, const std::string& dir, const std::string& libpath,
    const triton::common::BackendCmdlineConfig& backend_cmdline_config,
    std::shared_ptr<TritonBackend>* backend)
{
  // Create the JSON representation of the backend configuration.
  triton::common::TritonJson::Value backend_config_json(
      triton::common::TritonJson::ValueType::OBJECT);
  if (!backend_cmdline_config.empty()) {
    triton::common::TritonJson::Value cmdline_json(
        backend_config_json, triton::common::TritonJson::ValueType::OBJECT);
    for (const auto& pr : backend_cmdline_config) {
      RETURN_IF_ERROR(cmdline_json.AddString(pr.first.c_str(), pr.second));
    }

    RETURN_IF_ERROR(
        backend_config_json.Add("cmdline", std::move(cmdline_json)));
  }

  TritonServerMessage backend_config(backend_config_json);

  auto local_backend = std::shared_ptr<TritonBackend>(
      new TritonBackend(name, dir, libpath, backend_config));

  // Load the library and initialize all the entrypoints
  RETURN_IF_ERROR(local_backend->LoadBackendLibrary());

  // Backend initialization is optional... The TRITONBACKEND_Backend
  // object is this TritonBackend object. We must set set shared
  // library path to point to the backend directory in case the
  // backend library attempts to load additional shared libaries.
  if (local_backend->backend_init_fn_ != nullptr) {
    std::unique_ptr<SharedLibrary> slib;
    RETURN_IF_ERROR(SharedLibrary::Acquire(&slib));
    RETURN_IF_ERROR(slib->SetLibraryDirectory(local_backend->dir_));

    TRITONSERVER_Error* err = local_backend->backend_init_fn_(
        reinterpret_cast<TRITONBACKEND_Backend*>(local_backend.get()));

    RETURN_IF_ERROR(slib->ResetLibraryDirectory());
    RETURN_IF_TRITONSERVER_ERROR(err);
  }

  *backend = std::move(local_backend);
  return Status::Success;
}

TritonBackend::TritonBackend(
    const std::string& name, const std::string& dir, const std::string& libpath,
    const TritonServerMessage& backend_config)
    : name_(name), dir_(dir), libpath_(libpath),
      backend_config_(backend_config),
      exec_policy_(TRITONBACKEND_EXECUTION_BLOCKING), state_(nullptr)
{
  ClearHandles();
}

TritonBackend::~TritonBackend()
{
  LOG_VERBOSE(1) << "unloading backend '" << name_ << "'";

  // Backend finalization is optional... The TRITONBACKEND_Backend
  // object is this TritonBackend object.
  if (backend_fini_fn_ != nullptr) {
    LOG_TRITONSERVER_ERROR(
        backend_fini_fn_(reinterpret_cast<TRITONBACKEND_Backend*>(this)),
        "failed finalizing backend");
  }

  ClearHandles();
}

void
TritonBackend::ClearHandles()
{
  dlhandle_ = nullptr;
  backend_init_fn_ = nullptr;
  backend_fini_fn_ = nullptr;
  model_init_fn_ = nullptr;
  model_fini_fn_ = nullptr;
  inst_init_fn_ = nullptr;
  inst_fini_fn_ = nullptr;
  inst_exec_fn_ = nullptr;
}

Status
TritonBackend::LoadBackendLibrary()
{
  TritonBackendInitFn_t bifn;
  TritonBackendFiniFn_t bffn;
  TritonModelInitFn_t mifn;
  TritonModelFiniFn_t mffn;
  TritonModelInstanceInitFn_t iifn;
  TritonModelInstanceFiniFn_t iffn;
  TritonModelInstanceExecFn_t iefn;

  {
    std::unique_ptr<SharedLibrary> slib;
    RETURN_IF_ERROR(SharedLibrary::Acquire(&slib));

    RETURN_IF_ERROR(slib->OpenLibraryHandle(libpath_, &dlhandle_));

    // Backend initialize and finalize functions, optional
    RETURN_IF_ERROR(slib->GetEntrypoint(
        dlhandle_, "TRITONBACKEND_Initialize", true /* optional */,
        reinterpret_cast<void**>(&bifn)));
    RETURN_IF_ERROR(slib->GetEntrypoint(
        dlhandle_, "TRITONBACKEND_Finalize", true /* optional */,
        reinterpret_cast<void**>(&bffn)));

    // Model initialize and finalize functions, optional
    RETURN_IF_ERROR(slib->GetEntrypoint(
        dlhandle_, "TRITONBACKEND_ModelInitialize", true /* optional */,
        reinterpret_cast<void**>(&mifn)));
    RETURN_IF_ERROR(slib->GetEntrypoint(
        dlhandle_, "TRITONBACKEND_ModelFinalize", true /* optional */,
        reinterpret_cast<void**>(&mffn)));

    // Model instance initialize and finalize functions, optional
    RETURN_IF_ERROR(slib->GetEntrypoint(
        dlhandle_, "TRITONBACKEND_ModelInstanceInitialize", true /* optional */,
        reinterpret_cast<void**>(&iifn)));
    RETURN_IF_ERROR(slib->GetEntrypoint(
        dlhandle_, "TRITONBACKEND_ModelInstanceFinalize", true /* optional */,
        reinterpret_cast<void**>(&iffn)));

    // Model instance execute function, required
    RETURN_IF_ERROR(slib->GetEntrypoint(
        dlhandle_, "TRITONBACKEND_ModelInstanceExecute", false /* optional */,
        reinterpret_cast<void**>(&iefn)));
  }

  backend_init_fn_ = bifn;
  backend_fini_fn_ = bffn;
  model_init_fn_ = mifn;
  model_fini_fn_ = mffn;
  inst_init_fn_ = iifn;
  inst_fini_fn_ = iffn;
  inst_exec_fn_ = iefn;

  return Status::Success;
}

extern "C" {

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_ApiVersion(uint32_t* major, uint32_t* minor)
{
  *major = TRITONBACKEND_API_VERSION_MAJOR;
  *minor = TRITONBACKEND_API_VERSION_MINOR;
  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_BackendName(TRITONBACKEND_Backend* backend, const char** name)
{
  TritonBackend* tb = reinterpret_cast<TritonBackend*>(backend);
  *name = tb->Name().c_str();
  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_BackendConfig(
    TRITONBACKEND_Backend* backend, TRITONSERVER_Message** backend_config)
{
  TritonBackend* tb = reinterpret_cast<TritonBackend*>(backend);
  *backend_config = const_cast<TRITONSERVER_Message*>(
      reinterpret_cast<const TRITONSERVER_Message*>(&tb->BackendConfig()));
  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_BackendExecutionPolicy(
    TRITONBACKEND_Backend* backend, TRITONBACKEND_ExecutionPolicy* policy)
{
  TritonBackend* tb = reinterpret_cast<TritonBackend*>(backend);
  *policy = tb->ExecutionPolicy();
  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_BackendSetExecutionPolicy(
    TRITONBACKEND_Backend* backend, TRITONBACKEND_ExecutionPolicy policy)
{
  TritonBackend* tb = reinterpret_cast<TritonBackend*>(backend);
  tb->SetExecutionPolicy(policy);
  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_BackendArtifacts(
    TRITONBACKEND_Backend* backend, TRITONBACKEND_ArtifactType* artifact_type,
    const char** location)
{
  TritonBackend* tb = reinterpret_cast<TritonBackend*>(backend);
  *artifact_type = TRITONBACKEND_ARTIFACT_FILESYSTEM;
  *location = tb->Directory().c_str();
  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_BackendMemoryManager(
    TRITONBACKEND_Backend* backend, TRITONBACKEND_MemoryManager** manager)
{
  static TritonMemoryManager gMemoryManager;
  *manager = reinterpret_cast<TRITONBACKEND_MemoryManager*>(&gMemoryManager);
  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_BackendState(TRITONBACKEND_Backend* backend, void** state)
{
  TritonBackend* tb = reinterpret_cast<TritonBackend*>(backend);
  *state = tb->State();
  return nullptr;  // success
}

TRITONAPI_DECLSPEC TRITONSERVER_Error*
TRITONBACKEND_BackendSetState(TRITONBACKEND_Backend* backend, void* state)
{
  TritonBackend* tb = reinterpret_cast<TritonBackend*>(backend);
  tb->SetState(state);
  return nullptr;  // success
}

}  // extern C

//
// TritonBackendManager
//

static std::weak_ptr<TritonBackendManager> backend_manager_;
static std::mutex mu_;

Status
TritonBackendManager::Create(std::shared_ptr<TritonBackendManager>* manager)
{
  std::lock_guard<std::mutex> lock(mu_);

  // If there is already a manager then we just use it...
  *manager = backend_manager_.lock();
  if (*manager != nullptr) {
    return Status::Success;
  }

  manager->reset(new TritonBackendManager());
  backend_manager_ = *manager;

  return Status::Success;
}

Status
TritonBackendManager::CreateBackend(
    const std::string& name, const std::string& dir, const std::string& libpath,
    const triton::common::BackendCmdlineConfig& backend_cmdline_config,
    std::shared_ptr<TritonBackend>* backend)
{
  std::lock_guard<std::mutex> lock(mu_);

  const auto& itr = backend_map_.find(libpath);
  if (itr != backend_map_.end()) {
    *backend = itr->second;
    return Status::Success;
  }

  RETURN_IF_ERROR(TritonBackend::Create(
      name, dir, libpath, backend_cmdline_config, backend));
  backend_map_.insert({libpath, *backend});

  return Status::Success;
}

Status
TritonBackendManager::BackendState(
    std::unique_ptr<std::unordered_map<std::string, std::vector<std::string>>>*
        backend_state)
{
  std::lock_guard<std::mutex> lock(mu_);

  std::unique_ptr<std::unordered_map<std::string, std::vector<std::string>>>
      backend_state_map(
          new std::unordered_map<std::string, std::vector<std::string>>);
  for (const auto& backend_pair : backend_map_) {
    auto& libpath = backend_pair.first;
    auto backend = backend_pair.second;

    const char* backend_config;
    size_t backend_config_size;
    backend->BackendConfig().Serialize(&backend_config, &backend_config_size);
    backend_state_map->insert(
        {backend->Name(), std::vector<std::string>{libpath, backend_config}});
  }

  *backend_state = std::move(backend_state_map);

  return Status::Success;
}

}  // namespace hercules::core
