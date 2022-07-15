
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/

#pragma once

#include <onnxruntime_c_api.h>
#include <memory>
#include <mutex>
#include "triton/backend/backend_common.h"
#include "triton/core/tritonbackend.h"

namespace hercules::backend { namespace onnxruntime {

/// A singleton to load Onnx model because loading models requires
/// Onnx Runtime environment which is unique per process
class OnnxLoader {
 public:
  ~OnnxLoader();

  /// Initialize loader with default environment settings
  static TRITONSERVER_Error* Init(common::TritonJson::Value& backend_config);

  /// Stop loader, and once all Onnx sessions are unloaded via UnloadSession()
  /// the resource it allocated will be released
  static TRITONSERVER_Error* Stop();

  /// Load a Onnx model from a path and return the corresponding
  /// OrtSession.
  ///
  /// \param bool is_path If true 'model' is a path to the model file,
  /// if false 'model' is the serialized model.
  /// \param model The Onnx model or path to the model.
  /// \param session_options The options to use when creating the session
  /// \param session Returns the Onnx model session
  /// \return Error status.
  static TRITONSERVER_Error* LoadSession(
      const bool is_path, const std::string& model,
      const OrtSessionOptions* session_options, OrtSession** session);

  /// Unload a Onnx model session
  ///
  /// \param session The Onnx model session to be unloaded
  static TRITONSERVER_Error* UnloadSession(OrtSession* session);

  /// Returns whether global thread pool is enabled.
  /// If the loader is not initialized it returns false.
  static bool IsGlobalThreadPoolEnabled();

 private:
  OnnxLoader(OrtEnv* env, bool enable_global_threadpool = false)
      : env_(env), global_threadpool_enabled_(enable_global_threadpool),
        live_session_cnt_(0), closing_(false)
  {
  }

  /// Decrease 'live_session_cnt_' if 'decrement_session_cnt' is true, and then
  /// release Onnx Runtime environment if it is closing and no live sessions
  ///
  /// \param decrement_session_cnt Whether to decrease the 'live_session_cnt_'
  static void TryRelease(bool decrement_session_cnt);

  static std::unique_ptr<OnnxLoader> loader;

  OrtEnv* env_;
  bool global_threadpool_enabled_;
  std::mutex mu_;
  size_t live_session_cnt_;
  bool closing_;
};

}}}  // namespace hercules::backend::onnxruntime
