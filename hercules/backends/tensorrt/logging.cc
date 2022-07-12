// Copyright 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "logging.h"

#include "triton/backend/backend_common.h"

namespace triton { namespace backend { namespace tensorrt {

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

}}}  // namespace triton::backend::tensorrt
