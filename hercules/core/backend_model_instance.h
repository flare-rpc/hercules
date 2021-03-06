
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/

#pragma once

#include <functional>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include "constants.h"
#include "memory.h"
#include "metric_model_reporter.h"
#include "hercules/proto/model_config.pb.h"
#include "server_message.h"
#include "status.h"
#include "hercules/common/sync_queue.h"

namespace hercules::core {

class TritonModel;
class inference_request;

//
// Represents a model instance.
//
class TritonModelInstance {
 public:
  static Status CreateInstances(
      TritonModel* model,
      const hercules::common::HostPolicyCmdlineConfigMap& host_policy_map,
      const hercules::proto::ModelConfig& model_config, const bool device_blocking);
  ~TritonModelInstance();

  const std::string& Name() const { return name_; }
  size_t Index() const { return index_; }
  TRITONSERVER_InstanceGroupKind Kind() const { return kind_; }
  int32_t DeviceId() const { return device_id_; }
  const hercules::common::HostPolicyCmdlineConfig& HostPolicy() const
  {
    return host_policy_;
  }
  const TritonServerMessage& HostPolicyMessage() const
  {
    return host_policy_message_;
  }
  bool IsPassive() const { return passive_; }
  const std::vector<std::string>& Profiles() const { return profile_names_; }

  struct SecondaryDevice {
    SecondaryDevice(const std::string kind, const int64_t id)
        : kind_(kind), id_(id)
    {
    }
    const std::string kind_;
    const int64_t id_;
  };
  const std::vector<SecondaryDevice>& SecondaryDevices() const
  {
    return secondary_devices_;
  }

  Status Initialize();
  Status WarmUp();
  void Schedule(
      std::vector<std::unique_ptr<inference_request>>&& requests,
      const std::function<void()>& OnCompletion);

  TritonModel* Model() const { return model_; }
  void* State() { return state_; }
  void SetState(void* state) { state_ = state; }

  metric_model_reporter* MetricReporter() const { return reporter_.get(); }

 private:
  FLARE_DISALLOW_COPY_AND_ASSIGN(TritonModelInstance);
  class TritonBackendThread;
  TritonModelInstance(
      TritonModel* model, const std::string& name, const size_t index,
      const TRITONSERVER_InstanceGroupKind kind, const int32_t device_id,
      const std::vector<std::string>& profile_names, const bool passive,
      const hercules::common::HostPolicyCmdlineConfig& host_policy,
      const TritonServerMessage& host_policy_message,
      const std::vector<SecondaryDevice>& secondary_devices);
  static Status CreateInstance(
      TritonModel* model, const std::string& name, const size_t index,
      const TRITONSERVER_InstanceGroupKind kind, const int32_t device_id,
      const std::vector<std::string>& profile_names, const bool passive,
      const std::string& host_policy_name,
      const hercules::common::HostPolicyCmdlineConfig& host_policy,
      const hercules::proto::ModelRateLimiter& rate_limiter_config,
      const bool device_blocking,
      std::map<uint32_t, std::shared_ptr<TritonBackendThread>>*
          device_to_thread_map,
      const std::vector<SecondaryDevice>& secondary_devices);
  Status SetBackendThread(
      const TRITONSERVER_InstanceGroupKind kind, const int32_t device_id,
      const bool device_blocking,
      std::map<uint32_t, std::shared_ptr<TritonBackendThread>>*
          device_to_thread_map);
  Status GenerateWarmupData();

  void Execute(std::vector<TRITONBACKEND_Request*>& triton_requests);

  class TritonBackendThread {
   public:
    static Status CreateBackendThread(
        const std::string name, TritonModelInstance* model, const int nice,
        const int32_t device_id,
        std::unique_ptr<TritonBackendThread>* triton_backend_thread);
    void AddModelInstance(TritonModelInstance* model_instance);
    Status InitAndWarmUpModelInstance(TritonModelInstance* model_instance);
    void StopBackendThread();
    ~TritonBackendThread();

   private:
    TritonBackendThread(const std::string& name, TritonModel* model);
    void BackendThread(const int nice, const int32_t device_id);

    std::string name_;

    TritonModel* model_;
    std::deque<TritonModelInstance*> model_instances_;

    std::thread backend_thread_;
    std::atomic<bool> backend_thread_exit_;
  };
  std::shared_ptr<TritonBackendThread> triton_backend_thread_;

  struct WarmupData {
    WarmupData(const std::string& sample_name, const size_t count)
        : sample_name_(sample_name), count_(std::max(count, size_t{1}))
    {
    }

    std::string sample_name_;
    size_t count_;
    // Using a batch of requests to satisfy batch size, this provides better
    // alignment on the batch expected by the model, especially for sequence
    // model.
    std::vector<std::unique_ptr<inference_request>> requests_;

    // Placeholder for input data
    std::unique_ptr<allocated_memory> zero_data_;
    std::unique_ptr<allocated_memory> random_data_;
    std::vector<std::unique_ptr<std::string>> provided_data_;
  };
  std::vector<WarmupData> warmup_samples_;

  // The TritonModel object that owns this instance. The instance
  // holds this as a raw pointer because the lifetime of the model is
  // guaranteed to be longer than the lifetime of an instance owned by the
  // model.
  TritonModel* model_;

  std::string name_;
  size_t index_;

  // For CPU device_id_ is always 0. For GPU device_id_ indicates the
  // GPU device to be used by the instance.
  TRITONSERVER_InstanceGroupKind kind_;
  int32_t device_id_;
  const hercules::common::HostPolicyCmdlineConfig host_policy_;
  TritonServerMessage host_policy_message_;
  std::vector<std::string> profile_names_;
  bool passive_;

  std::vector<SecondaryDevice> secondary_devices_;

  // Reporter for metrics, or nullptr if no metrics should be reported
  std::shared_ptr<metric_model_reporter> reporter_;

  // Opaque state associated with this model instance.
  void* state_;
};

}  // namespace hercules::core
