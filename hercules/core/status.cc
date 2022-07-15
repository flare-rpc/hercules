
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/


#include "status.h"

namespace hercules::core {

const Status Status::Success(Status::Code::SUCCESS);

Status::Code
TritonCodeToStatusCode(TRITONSERVER_Error_Code code)
{
  switch (code) {
    case TRITONSERVER_ERROR_UNKNOWN:
      return Status::Code::UNKNOWN;
    case TRITONSERVER_ERROR_INTERNAL:
      return Status::Code::INTERNAL;
    case TRITONSERVER_ERROR_NOT_FOUND:
      return Status::Code::NOT_FOUND;
    case TRITONSERVER_ERROR_INVALID_ARG:
      return Status::Code::INVALID_ARG;
    case TRITONSERVER_ERROR_UNAVAILABLE:
      return Status::Code::UNAVAILABLE;
    case TRITONSERVER_ERROR_UNSUPPORTED:
      return Status::Code::UNSUPPORTED;
    case TRITONSERVER_ERROR_ALREADY_EXISTS:
      return Status::Code::ALREADY_EXISTS;

    default:
      break;
  }

  return Status::Code::UNKNOWN;
}

TRITONSERVER_Error_Code
StatusCodeToTritonCode(Status::Code status_code)
{
  switch (status_code) {
    case Status::Code::UNKNOWN:
      return TRITONSERVER_ERROR_UNKNOWN;
    case Status::Code::INTERNAL:
      return TRITONSERVER_ERROR_INTERNAL;
    case Status::Code::NOT_FOUND:
      return TRITONSERVER_ERROR_NOT_FOUND;
    case Status::Code::INVALID_ARG:
      return TRITONSERVER_ERROR_INVALID_ARG;
    case Status::Code::UNAVAILABLE:
      return TRITONSERVER_ERROR_UNAVAILABLE;
    case Status::Code::UNSUPPORTED:
      return TRITONSERVER_ERROR_UNSUPPORTED;
    case Status::Code::ALREADY_EXISTS:
      return TRITONSERVER_ERROR_ALREADY_EXISTS;

    default:
      break;
  }

  return TRITONSERVER_ERROR_UNKNOWN;
}

Status
CommonErrorToStatus(const triton::common::Error& error)
{
  return Status(error);
}

}  // namespace hercules::core
