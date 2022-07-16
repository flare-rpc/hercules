
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/

#pragma once

#ifdef HERCULES_ENABLE_METRICS

#include <mutex>
#include <unordered_map>

#include "infer_parameter.h"
#include "prometheus/registry.h"
#include "tritonserver_apis.h"

namespace hercules::core {

//
// Implementation for TRITONSERVER_MetricFamily.
//
class MetricFamily {
 public:
  MetricFamily(
      TRITONSERVER_MetricKind kind, const char* name, const char* description);
  ~MetricFamily();

  void* Family() const { return family_; }
  TRITONSERVER_MetricKind Kind() const { return kind_; }

  void* Add(std::map<std::string, std::string> label_map);
  void Remove(void* metric);

 private:
  void* family_;
  std::mutex mtx_;
  // prometheus returns the existing metric pointer if the metric with the same
  // set of labels are requested, as a result, different Metric objects may
  // refer to the same prometheus metric. So we must track the reference count
  // of the metric and request prometheus to remove it only when all references
  // are released.
  std::unordered_map<void*, size_t> metric_ref_cnt_;
  TRITONSERVER_MetricKind kind_;
};

//
// Implementation for TRITONSERVER_Metric.
//
class Metric {
 public:
  Metric(
      TRITONSERVER_MetricFamily* family,
      std::vector<const inference_parameter*> labels);
  ~Metric();

  MetricFamily* Family() const { return family_; }
  TRITONSERVER_MetricKind Kind() const { return kind_; }

  TRITONSERVER_Error* Value(double* value);
  TRITONSERVER_Error* Increment(double value);
  TRITONSERVER_Error* Set(double value);

 private:
  void* metric_;
  MetricFamily* family_;
  TRITONSERVER_MetricKind kind_;
};

}  // namespace hercules::core

#endif  // HERCULES_ENABLE_METRICS
