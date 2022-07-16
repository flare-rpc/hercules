
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/


#include "common.h"

#include "hercules/core/tritonserver.h"

namespace triton { namespace server {

TRITONSERVER_Error*
GetModelVersionFromString(const std::string& version_string, int64_t* version)
{
  if (version_string.empty()) {
    *version = -1;
    return nullptr;  // success
  }

  try {
    *version = std::stol(version_string);
  }
  catch (std::exception& e) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        std::string(
            "failed to get model version from specified version string '" +
            version_string + "' (details: " + e.what() +
            "), version should be an integral value > 0")
            .c_str());
  }

  if (*version < 0) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        std::string(
            "invalid model version specified '" + version_string +
            "' , version should be an integral value > 0")
            .c_str());
  }

  return nullptr;  // success
}

std::string
GetEnvironmentVariableOrDefault(
    const std::string& variable_name, const std::string& default_value)
{
  const char* value = getenv(variable_name.c_str());
  return value ? value : default_value;
}

int64_t
GetElementCount(const std::vector<int64_t>& dims)
{
  bool first = true;
  int64_t cnt = 0;
  for (auto dim : dims) {
    if (dim == WILDCARD_DIM) {
      return -1;
    }

    if (first) {
      cnt = dim;
      first = false;
    } else {
      cnt *= dim;
    }
  }

  return cnt;
}

}}  // namespace triton::server
