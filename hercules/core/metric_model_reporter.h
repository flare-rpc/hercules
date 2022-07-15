
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/

#pragma once

#include "status.h"
#include "hercules/common/model_config.h"

#ifdef TRITON_ENABLE_METRICS
#include "prometheus/registry.h"
#endif  // TRITON_ENABLE_METRICS

namespace hercules::core {

//
// Interface for a metric reporter for a given version of a model.
//
class metric_model_reporter {
 public:
#ifdef TRITON_ENABLE_METRICS
  static Status Create(
      const std::string& model_name, const int64_t model_version,
      const int device, const hercules::common::MetricTagsMap& model_tags,
      std::shared_ptr<metric_model_reporter>* metric_model_reporter);

  ~metric_model_reporter();

  // Get a metric for the given model, version and GPU index.
  prometheus::Counter& MetricInferenceSuccess() const
  {
    return *metric_inf_success_;
  }
  prometheus::Counter& MetricInferenceFailure() const
  {
    return *metric_inf_failure_;
  }
  prometheus::Counter& MetricInferenceCount() const
  {
    return *metric_inf_count_;
  }
  prometheus::Counter& MetricInferenceExecutionCount() const
  {
    return *metric_inf_exec_count_;
  }
  prometheus::Counter& MetricInferenceRequestDuration() const
  {
    return *metric_inf_request_duration_us_;
  }
  prometheus::Counter& MetricInferenceQueueDuration() const
  {
    return *metric_inf_queue_duration_us_;
  }
  prometheus::Counter& MetricInferenceComputeInputDuration() const
  {
    return *metric_inf_compute_input_duration_us_;
  }
  prometheus::Counter& MetricInferenceComputeInferDuration() const
  {
    return *metric_inf_compute_infer_duration_us_;
  }
  prometheus::Counter& MetricInferenceComputeOutputDuration() const
  {
    return *metric_inf_compute_output_duration_us_;
  }
  prometheus::Counter& MetricCacheHitCount() const
  {
    return *metric_cache_hit_count_;
  }
  prometheus::Counter& MetricCacheHitLookupDuration() const
  {
    return *metric_cache_hit_lookup_duration_us_;
  }
  prometheus::Counter& MetricCacheMissCount() const
  {
    return *metric_cache_miss_count_;
  }
  prometheus::Counter& MetricCacheMissLookupDuration() const
  {
    return *metric_cache_miss_lookup_duration_us_;
  }
  prometheus::Counter& MetricCacheMissInsertionDuration() const
  {
    return *metric_cache_miss_insertion_duration_us_;
  }

 private:
  metric_model_reporter(
      const std::string& model_name, const int64_t model_version,
      const int device, const hercules::common::MetricTagsMap& model_tags);

  static void GetMetricLabels(
      std::map<std::string, std::string>* labels, const std::string& model_name,
      const int64_t model_version, const int device,
      const hercules::common::MetricTagsMap& model_tags);
  prometheus::Counter* CreateCounterMetric(
      prometheus::Family<prometheus::Counter>& family,
      const std::map<std::string, std::string>& labels);

  prometheus::Counter* metric_inf_success_;
  prometheus::Counter* metric_inf_failure_;
  prometheus::Counter* metric_inf_count_;
  prometheus::Counter* metric_inf_exec_count_;
  prometheus::Counter* metric_inf_request_duration_us_;
  prometheus::Counter* metric_inf_queue_duration_us_;
  prometheus::Counter* metric_inf_compute_input_duration_us_;
  prometheus::Counter* metric_inf_compute_infer_duration_us_;
  prometheus::Counter* metric_inf_compute_output_duration_us_;
  prometheus::Counter* metric_cache_hit_count_;
  prometheus::Counter* metric_cache_hit_lookup_duration_us_;
  prometheus::Counter* metric_cache_miss_count_;
  prometheus::Counter* metric_cache_miss_lookup_duration_us_;
  prometheus::Counter* metric_cache_miss_insertion_duration_us_;
#endif  // TRITON_ENABLE_METRICS
};

}  // namespace hercules::core
