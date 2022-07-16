
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/

#pragma once

#include <iostream>
#include <string>
#include <vector>
#include "hercules/core/tritonserver.h"

namespace triton { namespace server {

constexpr char kInferHeaderContentLengthHTTPHeader[] =
    "Inference-Header-Content-Length";
constexpr char kAcceptEncodingHTTPHeader[] = "Accept-Encoding";
constexpr char kContentEncodingHTTPHeader[] = "Content-Encoding";
constexpr char kContentTypeHeader[] = "Content-Type";
constexpr char kContentLengthHeader[] = "Content-Length";

constexpr int MAX_GRPC_MESSAGE_SIZE = INT32_MAX;

/// The value for a dimension in a shape that indicates that that
/// dimension can take on any size.
constexpr int WILDCARD_DIM = -1;

#define RETURN_IF_ERR(X)             \
  do {                               \
    TRITONSERVER_Error* err__ = (X); \
    if (err__ != nullptr) {          \
      return err__;                  \
    }                                \
  } while (false)

#define RETURN_MSG_IF_ERR(X, MSG)                                      \
  do {                                                                 \
    TRITONSERVER_Error* err__ = (X);                                   \
    if (err__ != nullptr) {                                            \
      return TRITONSERVER_ErrorNew(                                    \
          TRITONSERVER_ErrorCode(err__),                               \
          (std::string(MSG) + ": " + TRITONSERVER_ErrorMessage(err__)) \
              .c_str());                                               \
    }                                                                  \
  } while (false)

#define GOTO_IF_ERR(X, T)            \
  do {                               \
    TRITONSERVER_Error* err__ = (X); \
    if (err__ != nullptr) {          \
      goto T;                        \
    }                                \
  } while (false)

#define FAIL(MSG)                                 \
  do {                                            \
    std::cerr << "error: " << (MSG) << std::endl; \
    exit(1);                                      \
  } while (false)

#define FAIL_IF_ERR(X, MSG)                                       \
  do {                                                            \
    TRITONSERVER_Error* err__ = (X);                              \
    if (err__ != nullptr) {                                       \
      std::cerr << "error: " << (MSG) << ": "                     \
                << TRITONSERVER_ErrorCodeString(err__) << " - "   \
                << TRITONSERVER_ErrorMessage(err__) << std::endl; \
      TRITONSERVER_ErrorDelete(err__);                            \
      exit(1);                                                    \
    }                                                             \
  } while (false)

#define IGNORE_ERR(X)                  \
  do {                                 \
    TRITONSERVER_Error* err__ = (X);   \
    if (err__ != nullptr) {            \
      TRITONSERVER_ErrorDelete(err__); \
    }                                  \
  } while (false)

#ifdef HERCULES_ENABLE_GPU
#define FAIL_IF_CUDA_ERR(X, MSG)                                           \
  do {                                                                     \
    cudaError_t err__ = (X);                                               \
    if (err__ != cudaSuccess) {                                            \
      std::cerr << "error: " << (MSG) << ": " << cudaGetErrorString(err__) \
                << std::endl;                                              \
      exit(1);                                                             \
    }                                                                      \
  } while (false)
#endif  // HERCULES_ENABLE_GPU

/// Get the integral version from a string, or fail if string does not
/// represent a valid version.
///
/// \param version_string The string version.
/// \param version Returns the integral version.
/// \return The error status. Failure if 'version_string' doesn't
/// convert to valid version.
TRITONSERVER_Error* GetModelVersionFromString(
    const std::string& version_string, int64_t* version);

/// Get the value of the environment variable, or default value if not set
///
/// \param variable_name The name of the environment variable.
/// \param default_value The default value.
/// \return The environment variable or the default value if not set.
std::string GetEnvironmentVariableOrDefault(
    const std::string& variable_name, const std::string& default_value);

/// Get the number of elements in a shape.
/// \param dims The shape.
/// \return The number of elements, or -1 if the number of elements
/// cannot be determined because the shape contains one or more
/// wilcard dimensions.
int64_t GetElementCount(const std::vector<int64_t>& dims);

}}  // namespace triton::server
