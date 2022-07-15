
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/

#pragma once

#include "http_server.h"

namespace triton { namespace server {

// Handle Vertex HTTP requests to inference server APIs
class VertexAiAPIServer : public HTTPAPIServer {
 public:
  static TRITONSERVER_Error* Create(
      const std::shared_ptr<TRITONSERVER_Server>& server,
      triton::server::TraceManager* trace_manager,
      const std::shared_ptr<SharedMemoryManager>& smb_manager,
      const int32_t port, const std::string address, const int thread_cnt,
      std::string default_model_name,
      std::unique_ptr<HTTPServer>* vertex_ai_server);

 private:
  explicit VertexAiAPIServer(
      const std::shared_ptr<TRITONSERVER_Server>& server,
      triton::server::TraceManager* trace_manager,
      const std::shared_ptr<SharedMemoryManager>& shm_manager,
      const int32_t port, const std::string address, const int thread_cnt,
      const std::string& prediction_route, const std::string& health_route,
      const std::string& default_model_name);

  void Handle(evhtp_request_t* req) override;

  void HandleMetrics(evhtp_request_t* req);

  TRITONSERVER_Error* GetInferenceHeaderLength(
      evhtp_request_t* req, int32_t content_length,
      size_t* header_length) override;

  // Currently the compresssion schema hasn't been defined,
  // assume identity compression type is used for both request and response
  DataCompressor::Type GetRequestCompressionType(evhtp_request_t* req) override
  {
    return DataCompressor::Type::IDENTITY;
  }
  DataCompressor::Type GetResponseCompressionType(evhtp_request_t* req) override
  {
    return DataCompressor::Type::IDENTITY;
  }
  re2::RE2 prediction_regex_;
  re2::RE2 health_regex_;
  const std::string health_mode_;

  // For default model, assume that only one version of "model" is presented
  const std::string model_name_;
  const std::string model_version_str_;

  static const std::string binary_mime_type_;
  static const std::string redirect_header_;
};

}}  // namespace triton::server
