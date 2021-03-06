
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/

#pragma once

#include <deque>
#include <functional>
#include <string>
#include <vector>
#include "hercules/core/buffer_attributes.h"
#include "constants.h"
#include "infer_parameter.h"
#include "infer_trace.h"
#include "response_allocator.h"
#include "status.h"
#include "hercules/common/model_config.h"
#include "tritonserver_apis.h"

namespace hercules::core {

class Model;
class InferenceResponse;

//
// An inference response factory.
//
class InferenceResponseFactory {
 public:
  InferenceResponseFactory() = default;

  InferenceResponseFactory(
      const std::shared_ptr<Model>& model, const std::string& id,
      const ResponseAllocator* allocator, void* alloc_userp,
      TRITONSERVER_InferenceResponseCompleteFn_t response_fn,
      void* response_userp,
      const std::function<void(
          std::unique_ptr<InferenceResponse>&&, const uint32_t)>& delegator)
      : model_(model), id_(id), allocator_(allocator),
        alloc_userp_(alloc_userp), response_fn_(response_fn),
        response_userp_(response_userp), response_delegator_(delegator)
  {
  }

  const ResponseAllocator* Allocator() { return allocator_; }
  void* AllocatorUserp() { return alloc_userp_; }

  Status SetResponseDelegator(
      const std::function<void(
          std::unique_ptr<InferenceResponse>&&, const uint32_t)>& delegator)
  {
    response_delegator_ = delegator;
    return Status::Success;
  }

  // Create a new response.
  Status CreateResponse(std::unique_ptr<InferenceResponse>* response) const;

  // Send a "null" response with 'flags'.
  Status SendFlags(const uint32_t flags) const;

#ifdef HERCULES_ENABLE_TRACING
  const std::shared_ptr<InferenceTraceProxy>& Trace() const { return trace_; }
  void SetTrace(const std::shared_ptr<InferenceTraceProxy>& trace)
  {
    trace_ = trace;
  }
  void ReleaseTrace() { trace_ = nullptr; }
#endif  // HERCULES_ENABLE_TRACING

 private:
  // The model associated with this factory. For normal
  // requests/responses this will always be defined and acts to keep
  // the model loaded as long as this factory is live. It may be
  // nullptr for cases where the model itself created the request
  // (like running requests for warmup) and so must protect any uses
  // to handle the nullptr case.
  std::shared_ptr<Model> model_;

  // The ID of the corresponding request that should be included in
  // every response.
  std::string id_;

  // The response allocator and user pointer. The 'allocator_' is a
  // raw pointer because it is owned by the client, and the client is
  // responsible for ensuring that the lifetime of the allocator
  // extends longer that any request or response that depend on the
  // allocator.
  const ResponseAllocator* allocator_;
  void* alloc_userp_;

  // The response callback function and user pointer.
  TRITONSERVER_InferenceResponseCompleteFn_t response_fn_;
  void* response_userp_;

  // Delegator to be invoked on sending responses.
  std::function<void(std::unique_ptr<InferenceResponse>&&, const uint32_t)>
      response_delegator_;

#ifdef HERCULES_ENABLE_TRACING
  // Inference trace associated with this response.
  std::shared_ptr<InferenceTraceProxy> trace_;
#endif  // HERCULES_ENABLE_TRACING
};

//
// An inference response.
//
class InferenceResponse {
 public:
  // Output tensor
  class Output {
   public:
    Output(
        const std::string& name, const hercules::proto::DataType datatype,
        const std::vector<int64_t>& shape, const ResponseAllocator* allocator,
        void* alloc_userp)
        : name_(name), datatype_(datatype), shape_(shape),
          allocator_(allocator), alloc_userp_(alloc_userp),
          allocated_buffer_(nullptr)
    {
    }
    Output(
        const std::string& name, const hercules::proto::DataType datatype,
        std::vector<int64_t>&& shape, const ResponseAllocator* allocator,
        void* alloc_userp)
        : name_(name), datatype_(datatype), shape_(std::move(shape)),
          allocator_(allocator), alloc_userp_(alloc_userp),
          allocated_buffer_(nullptr)
    {
    }

    ~Output();

    // The name of the output tensor.
    const std::string& Name() const { return name_; }

    // Data type of the output tensor.
    hercules::proto::DataType DType() const { return datatype_; }

    // The shape of the output tensor.
    const std::vector<int64_t>& Shape() const { return shape_; }

    buffer_attributes* GetBufferAttributes() { return &buffer_attributes_; }

    // Reshape the output tensor. This function must only be called
    // for outputs that have respace specified in the model
    // configuration.
    void Reshape(
        const bool has_batch_dim, const hercules::proto::ModelOutput* output_config);

    // Get information about the buffer allocated for this output
    // tensor's data. If no buffer is allocated 'buffer' will return
    // nullptr and the other returned values will be undefined.
    Status DataBuffer(
        const void** buffer, size_t* buffer_byte_size,
        TRITONSERVER_MemoryType* memory_type, int64_t* memory_type_id,
        void** userp) const;

    // Allocate the buffer that should be used for this output
    // tensor's data. 'buffer' must return a buffer of size
    // 'buffer_byte_size'.  'memory_type' acts as both input and
    // output. On input gives the buffer memory type preferred by the
    // caller and on return holds the actual memory type of
    // 'buffer'. 'memory_type_id' acts as both input and output. On
    // input gives the buffer memory type id preferred by the caller
    // and returns the actual memory type id of 'buffer'. Only a
    // single buffer may be allocated for the output at any time, so
    // multiple calls to AllocateDataBuffer without intervening
    // ReleaseDataBuffer call will result in an error.
    Status AllocateDataBuffer(
        void** buffer, const size_t buffer_byte_size,
        TRITONSERVER_MemoryType* memory_type, int64_t* memory_type_id);

    // Release the buffer that was previously allocated by
    // AllocateDataBuffer(). Do nothing if AllocateDataBuffer() has
    // not been called.
    Status ReleaseDataBuffer();

   private:
    FLARE_DISALLOW_COPY_AND_ASSIGN(Output);
    friend std::ostream& operator<<(
        std::ostream& out, const InferenceResponse::Output& output);

    std::string name_;
    hercules::proto::DataType datatype_;
    std::vector<int64_t> shape_;

    // The response allocator and user pointer.
    const ResponseAllocator* allocator_;
    void* alloc_userp_;

    // Information about the buffer allocated by
    // AllocateDataBuffer(). This information is needed by
    // DataBuffer() and ReleaseDataBuffer().
    void* allocated_buffer_;
    buffer_attributes buffer_attributes_;
    void* allocated_userp_;
  };

  // InferenceResponse
  InferenceResponse(
      const std::shared_ptr<Model>& model, const std::string& id,
      const ResponseAllocator* allocator, void* alloc_userp,
      TRITONSERVER_InferenceResponseCompleteFn_t response_fn,
      void* response_userp,
      const std::function<void(
          std::unique_ptr<InferenceResponse>&&, const uint32_t)>& delegator);

  // "null" InferenceResponse is a special instance of InferenceResponse which
  // contains minimal information for calling InferenceResponse::Send,
  // InferenceResponse::NullResponse. nullptr will be passed as response in
  // 'response_fn'.
  InferenceResponse(
      TRITONSERVER_InferenceResponseCompleteFn_t response_fn,
      void* response_userp);

  const std::string& Id() const { return id_; }
  const std::string& ModelName() const;
  int64_t ActualModelVersion() const;
  const Status& ResponseStatus() const { return status_; }

  // The response parameters.
  const std::deque<inference_parameter>& Parameters() const
  {
    return parameters_;
  }

  // Add an parameter to the response.
  Status AddParameter(const char* name, const char* value);
  Status AddParameter(const char* name, const int64_t value);
  Status AddParameter(const char* name, const bool value);

  // The response outputs.
  const std::deque<Output>& Outputs() const { return outputs_; }

  // Add an output to the response. If 'output' is non-null
  // return a pointer to the newly added output.
  Status AddOutput(
      const std::string& name, const hercules::proto::DataType datatype,
      const std::vector<int64_t>& shape, Output** output = nullptr);
  Status AddOutput(
      const std::string& name, const hercules::proto::DataType datatype,
      std::vector<int64_t>&& shape, Output** output = nullptr);

  // Get the classification label associated with an output. Return
  // 'label' == nullptr if no label.
  Status ClassificationLabel(
      const Output& output, const uint32_t class_index,
      const char** label) const;

  // Send the response with success status. Calling this function
  // releases ownership of the response object and gives it to the
  // callback function.
  static Status Send(
      std::unique_ptr<InferenceResponse>&& response, const uint32_t flags);

  // Send the response with explicit status. Calling this function
  // releases ownership of the response object and gives it to the
  // callback function.
  static Status SendWithStatus(
      std::unique_ptr<InferenceResponse>&& response, const uint32_t flags,
      const Status& status);

#ifdef HERCULES_ENABLE_TRACING
  const std::shared_ptr<InferenceTraceProxy>& Trace() const { return trace_; }
  void SetTrace(const std::shared_ptr<InferenceTraceProxy>& trace)
  {
    trace_ = trace;
  }
  void ReleaseTrace() { trace_ = nullptr; }
#endif  // HERCULES_ENABLE_TRACING

 private:
  FLARE_DISALLOW_COPY_AND_ASSIGN(InferenceResponse);
  friend std::ostream& operator<<(
      std::ostream& out, const InferenceResponse& response);

#ifdef HERCULES_ENABLE_TRACING
  Status TraceOutputTensors(
      TRITONSERVER_InferenceTraceActivity activity, const std::string& msg);
#endif  // HERCULES_ENABLE_TRACING

  // The model associated with this factory. For normal
  // requests/responses this will always be defined and acts to keep
  // the model loaded as long as this factory is live. It may be
  // nullptr for cases where the model itself created the request
  // (like running requests for warmup) and so must protect any uses
  // to handle the nullptr case.
  std::shared_ptr<Model> model_;

  // The ID of the corresponding request that should be included in
  // every response.
  std::string id_;

  // Error status for the response.
  Status status_;

  // The parameters of the response. Use a deque so that there is no
  // reallocation.
  std::deque<inference_parameter> parameters_;

  // The result tensors. Use a deque so that there is no reallocation.
  std::deque<Output> outputs_;

  // The response allocator and user pointer.
  const ResponseAllocator* allocator_;
  void* alloc_userp_;

  // The response callback function and user pointer.
  TRITONSERVER_InferenceResponseCompleteFn_t response_fn_;
  void* response_userp_;

  // Delegator to be invoked on sending responses.
  std::function<void(std::unique_ptr<InferenceResponse>&&, const uint32_t)>
      response_delegator_;

  bool null_response_;

#ifdef HERCULES_ENABLE_TRACING
  // Inference trace associated with this response.
  std::shared_ptr<InferenceTraceProxy> trace_;
#endif  // HERCULES_ENABLE_TRACING
};

std::ostream& operator<<(std::ostream& out, const InferenceResponse& response);
std::ostream& operator<<(
    std::ostream& out, const InferenceResponse::Output& output);

}  // namespace hercules::core
