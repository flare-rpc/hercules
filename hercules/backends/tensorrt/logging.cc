
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/


#include "logging.h"

#include "hercules/backend/backend_common.h"

namespace hercules::backend { namespace tensorrt {

TensorRTLogger tensorrt_logger;

void
TensorRTLogger::log(Severity severity, const char* msg) noexcept
{
  switch (severity) {
    case Severity::kINTERNAL_ERROR:
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR, msg);
      break;
    case Severity::kERROR:
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR, msg);
      break;
    case Severity::kWARNING:
      LOG_MESSAGE(TRITONSERVER_LOG_WARN, msg);
      break;
    case Severity::kINFO:
      LOG_MESSAGE(TRITONSERVER_LOG_INFO, msg);
      break;
    case Severity::kVERBOSE:
      LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, msg);
      break;
  }
}

}}}  // namespace hercules::backend::tensorrt
