
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/

#pragma once

#include <NvInfer.h>

namespace hercules::backend { namespace tensorrt {

// Logger for TensorRT API
class TensorRTLogger : public nvinfer1::ILogger {
  void log(Severity severity, const char* msg) noexcept override;
};

extern TensorRTLogger tensorrt_logger;

}}}  // namespace hercules::backend::tensorrt
