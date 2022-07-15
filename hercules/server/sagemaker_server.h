
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/

#pragma once

#include <sys/stat.h>

#include <mutex>

#include "common.h"
#include "dirent.h"
#include "http_server.h"
#include "triton/core/tritonserver.h"

namespace triton { namespace server {

// Handle Sagemaker HTTP requests to inference server APIs
class SagemakerAPIServer : public HTTPAPIServer {
 public:
  static TRITONSERVER_Error* Create(
      const std::shared_ptr<TRITONSERVER_Server>& server,
      triton::server::TraceManager* trace_manager,
      const std::shared_ptr<SharedMemoryManager>& smb_manager,
      const int32_t port, const std::string address, const int thread_cnt,
      std::unique_ptr<HTTPServer>* sagemaker_server);

  class SagemakeInferRequestClass : public InferRequestClass {
   public:
    explicit SagemakeInferRequestClass(
        TRITONSERVER_Server* server, evhtp_request_t* req,
        DataCompressor::Type response_compression_type)
        : InferRequestClass(server, req, response_compression_type)
    {
    }
    using InferRequestClass::InferResponseComplete;
    static void InferResponseComplete(
        TRITONSERVER_InferenceResponse* response, const uint32_t flags,
        void* userp);

    void SetResponseHeader(
        const bool has_binary_data, const size_t header_length) override;
  };

 private:
  explicit SagemakerAPIServer(
      const std::shared_ptr<TRITONSERVER_Server>& server,
      triton::server::TraceManager* trace_manager,
      const std::shared_ptr<SharedMemoryManager>& shm_manager,
      const int32_t port, const std::string address, const int thread_cnt)
      : HTTPAPIServer(
            server, trace_manager, shm_manager, port, address, thread_cnt),
        ping_regex_(R"(/ping)"), invocations_regex_(R"(/invocations)"),
        models_regex_(R"(/models(?:/)?([^/]+)?(/invoke)?)"),
        model_path_regex_(
            R"((\/opt\/ml\/models\/[0-9A-Za-z._]+)\/(model)\/?([0-9A-Za-z._]+)?)"),
        ping_mode_("ready"),
        model_name_(GetEnvironmentVariableOrDefault(
            "SAGEMAKER_TRITON_DEFAULT_MODEL_NAME",
            "unspecified_SAGEMAKER_TRITON_DEFAULT_MODEL_NAME")),
        model_version_str_("")
  {
  }

  void ParseSageMakerRequest(
      evhtp_request_t* req,
      std::unordered_map<std::string, std::string>* parse_map,
      const std::string& action);

  void SageMakerMMEHandleInfer(
      evhtp_request_t* req, const std::string& model_name,
      const std::string& model_version_str);

  void SageMakerMMELoadModel(
      evhtp_request_t* req,
      const std::unordered_map<std::string, std::string> parse_map);

  void SageMakerMMEHandleOOMError(
      evhtp_request_t* req, TRITONSERVER_Error* load_err);

  static bool SageMakerMMECheckOOMError(TRITONSERVER_Error* load_err);

  void SageMakerMMEUnloadModel(evhtp_request_t* req, const char* model_name);

  void SageMakerMMEListModel(evhtp_request_t* req);

  void SageMakerMMEGetModel(evhtp_request_t* req, const char* model_name);

  void Handle(evhtp_request_t* req) override;

  /* Method to return 507 on invoke i.e. during SageMakerMMEHandleInfer
   */
  static void BADReplyCallback507(evthr_t* thr, void* arg, void* shared);

  std::unique_ptr<InferRequestClass> CreateInferRequest(
      evhtp_request_t* req) override
  {
    return std::unique_ptr<InferRequestClass>(new SagemakeInferRequestClass(
        server_.get(), req, GetResponseCompressionType(req)));
  }
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
  re2::RE2 ping_regex_;
  re2::RE2 invocations_regex_;
  re2::RE2 models_regex_;
  re2::RE2 model_path_regex_;

  const std::string ping_mode_;

  /* For single model mode, assume that only one version of "model" is presented
   */
  const std::string model_name_;
  const std::string model_version_str_;

  static const std::string binary_mime_type_;

  /* Maintain list of loaded models */
  std::unordered_map<std::string, std::string> sagemaker_models_list_;

  /* Mutex to handle concurrent updates */
  std::mutex mutex_;
};

}}  // namespace triton::server
