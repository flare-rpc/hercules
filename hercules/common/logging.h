//
// Created by liyinbin on 2022/7/16.
//

#pragma once

#include "flare/log/logging.h"

#define LOG_TRITONSERVER_ERROR(X, MSG)                                  \
  do {                                                                  \
    TRITONSERVER_Error* err__ = (X);                                    \
    if (err__ != nullptr) {                                             \
      FLARE_LOG(ERROR) << (MSG) << ": " << TRITONSERVER_ErrorCodeString(err__) \
                << " - " << TRITONSERVER_ErrorMessage(err__);           \
      TRITONSERVER_ErrorDelete(err__);                                  \
    }                                                                   \
  } while (false)

#define LOG_STATUS_ERROR(X, MSG)                         \
  do {                                                   \
    const Status& status__ = (X);                        \
    if (!status__.IsOk()) {                              \
      FLARE_LOG(ERROR) << (MSG) << ": " << status__.AsString(); \
    }                                                    \
  } while (false)
