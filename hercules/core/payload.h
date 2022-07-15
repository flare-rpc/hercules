
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/

#pragma once

#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <vector>

#include "backend_model_instance.h"
#include "infer_request.h"
#include "status.h"

namespace hercules::core {

class Payload {
 public:
  enum Operation { INFER_RUN = 0, INIT = 1, WARM_UP = 2, EXIT = 3 };
  enum State {
    UNINITIALIZED = 0,
    READY = 1,
    REQUESTED = 2,
    SCHEDULED = 3,
    EXECUTING = 4,
    RELEASED = 5
  };

  Payload();
  void Reset(const Operation op_type, TritonModelInstance* instance = nullptr);
  Status MergePayload(std::shared_ptr<Payload>& payload);
  Operation GetOpType() { return op_type_; }
  std::mutex* GetExecMutex() { return exec_mu_.get(); }
  size_t RequestCount() { return requests_.size(); }
  size_t BatchSize();
  void ReserveRequests(size_t size);
  void AddRequest(std::unique_ptr<inference_request> request);
  std::vector<std::unique_ptr<inference_request>>& Requests()
  {
    return requests_;
  }
  uint64_t BatcherStartNs() { return batcher_start_ns_; }
  void SetCallback(std::function<void()> OnCallback);
  void Callback();
  void AddInternalReleaseCallback(std::function<void()>&& callback);
  void OnRelease();
  void SetInstance(TritonModelInstance* model_instance);
  TritonModelInstance* GetInstance() { return instance_; }
  void MarkSaturated();
  bool IsSaturated() { return saturated_; }


  State GetState() { return state_; }
  void SetState(State state);
  void Execute(bool* should_exit);
  Status Wait();
  void Release();

 private:
  Operation op_type_;
  std::vector<std::unique_ptr<inference_request>> requests_;
  std::function<void()> OnCallback_;
  std::vector<std::function<void()>> release_callbacks_;
  TritonModelInstance* instance_;
  State state_;
  std::unique_ptr<std::promise<Status>> status_;
  std::unique_ptr<std::mutex> exec_mu_;
  uint64_t batcher_start_ns_;

  bool saturated_;
};

}  // namespace hercules::core
