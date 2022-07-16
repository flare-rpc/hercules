
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/

//
#pragma once

#ifdef HERCULES_ENABLE_METRICS

#include <atomic>
#include <mutex>
#include <thread>
#include "prometheus/registry.h"
#include "prometheus/serializer.h"
#include "prometheus/text_serializer.h"
#include "response_cache.h"

#ifdef TRITON_ENABLE_METRICS_GPU
#include <dcgm_agent.h>
#endif  // TRITON_ENABLE_METRICS_GPU

namespace hercules::core {

#ifdef TRITON_ENABLE_METRICS_GPU
struct DcgmMetadata {
  // DCGM handles for initialization and destruction
  dcgmHandle_t dcgm_handle_ = 0;
  dcgmGpuGrp_t groupId_ = 0;
  // DCGM Flags
  bool standalone_ = false;
  // DCGM Fields
  size_t field_count_ = 0;
  std::vector<unsigned short> fields_;
  // GPU Device Mapping
  std::map<uint32_t, uint32_t> cuda_ids_to_dcgm_ids_;
  std::vector<uint32_t> available_cuda_gpu_ids_;
  // Stop attempting metrics if they fail multiple consecutive
  // times for a device.
  const int fail_threshold_ = 3;
  // DCGM Failure Tracking
  std::vector<int> power_limit_fail_cnt_;
  std::vector<int> power_usage_fail_cnt_;
  std::vector<int> energy_fail_cnt_;
  std::vector<int> util_fail_cnt_;
  std::vector<int> mem_fail_cnt_;
  // DCGM Energy Tracking
  std::vector<unsigned long long> last_energy_;
  // Track if DCGM handle initialized successfully
  bool dcgm_initialized_ = false;
};
#endif  // TRITON_ENABLE_METRICS_GPU

class Metrics {
 public:
  // Return the hash value of the labels
  static size_t HashLabels(const std::map<std::string, std::string>& labels);

  // Are metrics enabled?
  static bool Enabled();

  // Enable reporting of metrics
  static void EnableMetrics();

  // Enable reporting of GPU metrics
  static void EnableGPUMetrics();

  // Enable reporting of Cache metrics
  static void EnableCacheMetrics(
      std::shared_ptr<RequestResponseCache> response_cache);

  // Start a thread for polling enabled metrics if any
  static void StartPollingThreadSingleton(
      std::shared_ptr<RequestResponseCache> response_cache);

  // Set the time interval in secs at which metrics are collected
  static void SetMetricsInterval(uint64_t metrics_interval_ms);

  // Get the prometheus registry
  static std::shared_ptr<prometheus::Registry> GetRegistry();

  // Get serialized metrics
  static const std::string SerializedMetrics();

  // Get the UUID for a CUDA device. Return true and initialize 'uuid'
  // if a UUID is found, return false if a UUID cannot be returned.
  static bool UUIDForCudaDevice(int cuda_device, std::string* uuid);

  // Metric family counting successful inference requests
  static prometheus::Family<prometheus::Counter>& FamilyInferenceSuccess()
  {
    return GetSingleton()->inf_success_family_;
  }

  // Metric family counting failed inference requests
  static prometheus::Family<prometheus::Counter>& FamilyInferenceFailure()
  {
    return GetSingleton()->inf_failure_family_;
  }

  // Metric family counting inferences performed, where a batch-size
  // 'n' inference request is counted as 'n' inferences
  static prometheus::Family<prometheus::Counter>& FamilyInferenceCount()
  {
    return GetSingleton()->inf_count_family_;
  }

  // Metric family counting inferences performed, where a batch-size
  // 'n' inference request is counted as 'n' inferences
  static prometheus::Family<prometheus::Counter>&
  FamilyInferenceExecutionCount()
  {
    return GetSingleton()->inf_count_exec_family_;
  }

  // Metric family of cumulative inference request duration, in
  // microseconds
  static prometheus::Family<prometheus::Counter>&
  FamilyInferenceRequestDuration()
  {
    return GetSingleton()->inf_request_duration_us_family_;
  }

  // Metric family of cumulative inference queuing duration, in
  // microseconds
  static prometheus::Family<prometheus::Counter>& FamilyInferenceQueueDuration()
  {
    return GetSingleton()->inf_queue_duration_us_family_;
  }

  // Metric family of cumulative inference compute durations, in
  // microseconds
  static prometheus::Family<prometheus::Counter>&
  FamilyInferenceComputeInputDuration()
  {
    return GetSingleton()->inf_compute_input_duration_us_family_;
  }
  static prometheus::Family<prometheus::Counter>&
  FamilyInferenceComputeInferDuration()
  {
    return GetSingleton()->inf_compute_infer_duration_us_family_;
  }
  static prometheus::Family<prometheus::Counter>&
  FamilyInferenceComputeOutputDuration()
  {
    return GetSingleton()->inf_compute_output_duration_us_family_;
  }
  // Metric families of per-model response cache metrics
  static prometheus::Family<prometheus::Counter>& FamilyCacheHitCount()
  {
    return GetSingleton()->cache_num_hits_model_family_;
  }
  static prometheus::Family<prometheus::Counter>& FamilyCacheHitLookupDuration()
  {
    return GetSingleton()->cache_hit_lookup_duration_us_model_family_;
  }
  static prometheus::Family<prometheus::Counter>& FamilyCacheMissCount()
  {
    return GetSingleton()->cache_num_misses_model_family_;
  }
  static prometheus::Family<prometheus::Counter>&
  FamilyCacheMissLookupDuration()
  {
    return GetSingleton()->cache_miss_lookup_duration_us_model_family_;
  }
  static prometheus::Family<prometheus::Counter>&
  FamilyCacheMissInsertionDuration()
  {
    return GetSingleton()->cache_miss_insertion_duration_us_model_family_;
  }


 private:
  Metrics();
  virtual ~Metrics();
  static Metrics* GetSingleton();
  bool InitializeDcgmMetrics();
  bool InitializeCacheMetrics(
      std::shared_ptr<RequestResponseCache> response_cache);
  bool StartPollingThread(std::shared_ptr<RequestResponseCache> response_cache);
  bool PollCacheMetrics(std::shared_ptr<RequestResponseCache> response_cache);
  bool PollDcgmMetrics();

  std::string dcgmValueToErrorMessage(double val);
  std::string dcgmValueToErrorMessage(int64_t val);

  std::shared_ptr<prometheus::Registry> registry_;
  std::unique_ptr<prometheus::Serializer> serializer_;

  prometheus::Family<prometheus::Counter>& inf_success_family_;
  prometheus::Family<prometheus::Counter>& inf_failure_family_;
  prometheus::Family<prometheus::Counter>& inf_count_family_;
  prometheus::Family<prometheus::Counter>& inf_count_exec_family_;
  prometheus::Family<prometheus::Counter>& inf_request_duration_us_family_;
  prometheus::Family<prometheus::Counter>& inf_queue_duration_us_family_;
  prometheus::Family<prometheus::Counter>&
      inf_compute_input_duration_us_family_;
  prometheus::Family<prometheus::Counter>&
      inf_compute_infer_duration_us_family_;
  prometheus::Family<prometheus::Counter>&
      inf_compute_output_duration_us_family_;
  // Global Response Cache metrics
  prometheus::Family<prometheus::Gauge>& cache_num_entries_family_;
  prometheus::Family<prometheus::Gauge>& cache_num_lookups_family_;
  prometheus::Family<prometheus::Gauge>& cache_num_hits_family_;
  prometheus::Family<prometheus::Gauge>& cache_num_misses_family_;
  prometheus::Family<prometheus::Gauge>& cache_num_evictions_family_;
  prometheus::Family<prometheus::Gauge>& cache_lookup_duration_us_family_;
  prometheus::Family<prometheus::Gauge>& cache_insertion_duration_us_family_;
  prometheus::Family<prometheus::Gauge>& cache_util_family_;
  // Gauges for Global Response Cache metrics
  prometheus::Gauge* cache_num_entries_global_;
  prometheus::Gauge* cache_num_lookups_global_;
  prometheus::Gauge* cache_num_hits_global_;
  prometheus::Gauge* cache_num_misses_global_;
  prometheus::Gauge* cache_num_evictions_global_;
  prometheus::Gauge* cache_lookup_duration_us_global_;
  prometheus::Gauge* cache_insertion_duration_us_global_;
  prometheus::Gauge* cache_util_global_;
  // Per-model Response Cache metrics
  prometheus::Family<prometheus::Counter>& cache_num_hits_model_family_;
  prometheus::Family<prometheus::Counter>&
      cache_hit_lookup_duration_us_model_family_;
  prometheus::Family<prometheus::Counter>& cache_num_misses_model_family_;
  prometheus::Family<prometheus::Counter>&
      cache_miss_lookup_duration_us_model_family_;
  prometheus::Family<prometheus::Counter>&
      cache_miss_insertion_duration_us_model_family_;

#ifdef TRITON_ENABLE_METRICS_GPU
  prometheus::Family<prometheus::Gauge>& gpu_utilization_family_;
  prometheus::Family<prometheus::Gauge>& gpu_memory_total_family_;
  prometheus::Family<prometheus::Gauge>& gpu_memory_used_family_;
  prometheus::Family<prometheus::Gauge>& gpu_power_usage_family_;
  prometheus::Family<prometheus::Gauge>& gpu_power_limit_family_;
  prometheus::Family<prometheus::Counter>& gpu_energy_consumption_family_;

  std::vector<prometheus::Gauge*> gpu_utilization_;
  std::vector<prometheus::Gauge*> gpu_memory_total_;
  std::vector<prometheus::Gauge*> gpu_memory_used_;
  std::vector<prometheus::Gauge*> gpu_power_usage_;
  std::vector<prometheus::Gauge*> gpu_power_limit_;
  std::vector<prometheus::Counter*> gpu_energy_consumption_;

  DcgmMetadata dcgm_metadata_;
#endif  // TRITON_ENABLE_METRICS_GPU

  // Thread for polling cache/gpu metrics periodically
  std::unique_ptr<std::thread> poll_thread_;
  std::atomic<bool> poll_thread_exit_;
  bool metrics_enabled_;
  bool gpu_metrics_enabled_;
  bool cache_metrics_enabled_;
  bool poll_thread_started_;
  std::mutex gpu_metrics_enabling_;
  std::mutex cache_metrics_enabling_;
  std::mutex poll_thread_starting_;
  uint64_t metrics_interval_ms_;
};

}  // namespace hercules::core

#endif  // HERCULES_ENABLE_METRICS
