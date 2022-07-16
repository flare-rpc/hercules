
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/


#include "hercules/core/tritonrepoagent.h"
#include "hercules/core/tritonserver.h"

#include <cctype>
#include <cstring>
#include <iomanip>
#include <stdexcept>
#include <string>

//
// Relocation Repository Agent that is for test only.
//

namespace triton { namespace repoagent { namespace relocation {

namespace {
//
// ErrorException
//
// Exception thrown if error occurs while running RelocationRepoAgent
//
struct ErrorException {
  ErrorException(TRITONSERVER_Error* err) : err_(err) {}
  TRITONSERVER_Error* err_;
};

#define THROW_IF_TRITON_ERROR(X)                                      \
  do {                                                                \
    TRITONSERVER_Error* tie_err__ = (X);                              \
    if (tie_err__ != nullptr) {                                       \
      throw triton::repoagent::relocation::ErrorException(tie_err__); \
    }                                                                 \
  } while (false)


#define THROW_TRITON_ERROR(CODE, MSG)                                 \
  do {                                                                \
    TRITONSERVER_Error* tie_err__ = TRITONSERVER_ErrorNew(CODE, MSG); \
    throw triton::repoagent::relocation::ErrorException(tie_err__);   \
  } while (false)


#define RETURN_IF_ERROR(X)               \
  do {                                   \
    TRITONSERVER_Error* rie_err__ = (X); \
    if (rie_err__ != nullptr) {          \
      return rie_err__;                  \
    }                                    \
  } while (false)

}  // namespace

/////////////

extern "C" {

TRITONSERVER_Error*
TRITONREPOAGENT_ModelFinalize(
    TRITONREPOAGENT_Agent* agent, TRITONREPOAGENT_AgentModel* model)
{
  const char* location;
  RETURN_IF_ERROR(TRITONREPOAGENT_ModelState(model, (void**)&location));
  RETURN_IF_ERROR(
      TRITONREPOAGENT_ModelRepositoryLocationRelease(agent, model, location));
  return nullptr;
}

TRITONSERVER_Error*
TRITONREPOAGENT_ModelAction(
    TRITONREPOAGENT_Agent* agent, TRITONREPOAGENT_AgentModel* model,
    const TRITONREPOAGENT_ActionType action_type)
{
  // Return success (nullptr) if the agent does not handle the action
  if (action_type != TRITONREPOAGENT_ACTION_LOAD) {
    return nullptr;
  }

  // Check the agent parameters for the relocation configuration of the model
  uint32_t parameter_count = 0;
  RETURN_IF_ERROR(
      TRITONREPOAGENT_ModelParameterCount(agent, model, &parameter_count));
  if (parameter_count != 1) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        "Relocation repoagent expects 1 parameter for relocation agent");
  }
  const char* key = nullptr;
  const char* value = nullptr;
  RETURN_IF_ERROR(
      TRITONREPOAGENT_ModelParameter(agent, model, 0, &key, &value));
  if (std::string(key) != "empty_config") {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        "Relocation repoagent expects parameter with key 'empty_config' for "
        "relocation agent");
  } else if (
      (std::string(value) != "true") && (std::string(value) != "false")) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        "Relocation repoagent expects 'empty_config' parameter with value "
        "'true' or 'false' for relocation agent");
  }
  TRITONSERVER_Message* model_config;
  RETURN_IF_ERROR(TRITONREPOAGENT_ModelConfig(agent, model, 1, &model_config));
  const char* base;
  size_t byte_size;
  auto err =
      TRITONSERVER_MessageSerializeToJson(model_config, &base, &byte_size);
  if (err == nullptr) {
    // hack to check if proper model config is passed by knowing that only
    // the original config will contain 'model_repository_agents' setting
    auto pos = std::string(base, byte_size).find("model_repository_agents");
    if ((std::string(value) == "true") && (pos != std::string::npos)) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          "Relocation repoagent expects config does not contain "
          "'model_repository_agents' field when 'empty_config' has value "
          "'true' for relocation agent");
    } else if ((std::string(value) == "false") && (pos == std::string::npos)) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          "Relocation repoagent expects config contains "
          "'model_repository_agents' field when 'empty_config' has value "
          "'false' for relocation agent");
    }
  }
  RETURN_IF_ERROR(TRITONSERVER_MessageDelete(model_config));
  RETURN_IF_ERROR(err);

  // Point to a new model repository
  const char* location;
  RETURN_IF_ERROR(TRITONREPOAGENT_ModelRepositoryLocationAcquire(
      agent, model, TRITONREPOAGENT_ARTIFACT_FILESYSTEM, &location));
  RETURN_IF_ERROR(TRITONREPOAGENT_ModelRepositoryUpdate(
      agent, model, TRITONREPOAGENT_ARTIFACT_FILESYSTEM, location));
  RETURN_IF_ERROR(TRITONREPOAGENT_ModelSetState(model, (void*)location));

  return nullptr;  // success
}

}  // extern "C"

}}}  // namespace triton::repoagent::relocation