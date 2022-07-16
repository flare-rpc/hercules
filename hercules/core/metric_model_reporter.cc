
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/


#include "metric_model_reporter.h"

#ifdef HERCULES_ENABLE_METRICS

#include "constants.h"
#include "metrics.h"

namespace hercules::core {

Status
metric_model_reporter::Create(
    const std::string& model_name, const int64_t model_version,
    const int device, const hercules::common::MetricTagsMap& model_tags,
    std::shared_ptr<metric_model_reporter>* metric_model_reporter)
{
  static std::mutex mtx;
  static std::unordered_map<size_t, std::weak_ptr<metric_model_reporter>>
      reporter_map;

  std::map<std::string, std::string> labels;
  GetMetricLabels(&labels, model_name, model_version, device, model_tags);
  auto hash_labels = Metrics::HashLabels(labels);

  std::lock_guard<std::mutex> lock(mtx);

  const auto& itr = reporter_map.find(hash_labels);
  if (itr != reporter_map.end()) {
    // Found in map. If the weak_ptr is still valid that means that
    // there are other models using the reporter and we just reuse that
    // same reporter. If the weak_ptr is not valid then we need to remove
    // the weak_ptr from the map and create the reporter again.
    *metric_model_reporter = itr->second.lock();
    if (*metric_model_reporter != nullptr) {
      return Status::Success;
    }

    reporter_map.erase(itr);
  }

  metric_model_reporter->reset(
      new metric_model_reporter(model_name, model_version, device, model_tags));
  reporter_map.insert({hash_labels, *metric_model_reporter});
  return Status::Success;
}

metric_model_reporter::metric_model_reporter(
    const std::string& model_name, const int64_t model_version,
    const int device, const hercules::common::MetricTagsMap& model_tags)
{
  std::map<std::string, std::string> labels;
  GetMetricLabels(&labels, model_name, model_version, device, model_tags);

  metric_inf_success_ =
      CreateCounterMetric(Metrics::FamilyInferenceSuccess(), labels);
  metric_inf_failure_ =
      CreateCounterMetric(Metrics::FamilyInferenceFailure(), labels);
  metric_inf_count_ =
      CreateCounterMetric(Metrics::FamilyInferenceCount(), labels);
  metric_inf_exec_count_ =
      CreateCounterMetric(Metrics::FamilyInferenceExecutionCount(), labels);
  metric_inf_request_duration_us_ =
      CreateCounterMetric(Metrics::FamilyInferenceRequestDuration(), labels);
  metric_inf_queue_duration_us_ =
      CreateCounterMetric(Metrics::FamilyInferenceQueueDuration(), labels);
  metric_inf_compute_input_duration_us_ = CreateCounterMetric(
      Metrics::FamilyInferenceComputeInputDuration(), labels);
  metric_inf_compute_infer_duration_us_ = CreateCounterMetric(
      Metrics::FamilyInferenceComputeInferDuration(), labels);
  metric_inf_compute_output_duration_us_ = CreateCounterMetric(
      Metrics::FamilyInferenceComputeOutputDuration(), labels);
  metric_cache_hit_count_ =
      CreateCounterMetric(Metrics::FamilyCacheHitCount(), labels);
  metric_cache_hit_lookup_duration_us_ =
      CreateCounterMetric(Metrics::FamilyCacheHitLookupDuration(), labels);
  metric_cache_miss_count_ =
      CreateCounterMetric(Metrics::FamilyCacheMissCount(), labels);
  metric_cache_miss_lookup_duration_us_ =
      CreateCounterMetric(Metrics::FamilyCacheMissLookupDuration(), labels);
  metric_cache_miss_insertion_duration_us_ =
      CreateCounterMetric(Metrics::FamilyCacheMissInsertionDuration(), labels);
}

metric_model_reporter::~metric_model_reporter()
{
  Metrics::FamilyInferenceSuccess().Remove(metric_inf_success_);
  Metrics::FamilyInferenceFailure().Remove(metric_inf_failure_);
  Metrics::FamilyInferenceCount().Remove(metric_inf_count_);
  Metrics::FamilyInferenceExecutionCount().Remove(metric_inf_exec_count_);
  Metrics::FamilyInferenceRequestDuration().Remove(
      metric_inf_request_duration_us_);
  Metrics::FamilyInferenceQueueDuration().Remove(metric_inf_queue_duration_us_);
  Metrics::FamilyInferenceComputeInputDuration().Remove(
      metric_inf_compute_input_duration_us_);
  Metrics::FamilyInferenceComputeInferDuration().Remove(
      metric_inf_compute_infer_duration_us_);
  Metrics::FamilyInferenceComputeOutputDuration().Remove(
      metric_inf_compute_output_duration_us_);
  Metrics::FamilyCacheHitCount().Remove(metric_cache_hit_count_);
  Metrics::FamilyCacheHitLookupDuration().Remove(
      metric_cache_hit_lookup_duration_us_);
  Metrics::FamilyCacheMissCount().Remove(metric_cache_miss_count_);
  Metrics::FamilyCacheMissInsertionDuration().Remove(
      metric_cache_miss_insertion_duration_us_);
}

void
metric_model_reporter::GetMetricLabels(
    std::map<std::string, std::string>* labels, const std::string& model_name,
    const int64_t model_version, const int device,
    const hercules::common::MetricTagsMap& model_tags)
{
  labels->insert(std::map<std::string, std::string>::value_type(
      std::string(kMetricsLabelModelName), model_name));
  labels->insert(std::map<std::string, std::string>::value_type(
      std::string(kMetricsLabelModelVersion), std::to_string(model_version)));
  for (const auto& tag : model_tags) {
    labels->insert(std::map<std::string, std::string>::value_type(
        "_" + tag.first, tag.second));
  }

  // 'device' can be < 0 to indicate that the GPU is not known. In
  // that case use a metric that doesn't have the gpu_uuid label.
  if (device >= 0) {
    std::string uuid;
    if (Metrics::UUIDForCudaDevice(device, &uuid)) {
      labels->insert(std::map<std::string, std::string>::value_type(
          std::string(kMetricsLabelGpuUuid), uuid));
    }
  }
}

prometheus::Counter*
metric_model_reporter::CreateCounterMetric(
    prometheus::Family<prometheus::Counter>& family,
    const std::map<std::string, std::string>& labels)
{
  return &family.Add(labels);
}

}  // namespace hercules::core

#endif  // HERCULES_ENABLE_METRICS
