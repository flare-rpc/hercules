
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/


#ifdef HERCULES_ENABLE_METRICS

#include "metric_family.h"
#include "metrics.h"
#include "hercules/common/logging.h"

namespace hercules::core {

//
// Implementation for TRITONSERVER_MetricFamily.
//
MetricFamily::MetricFamily(
    TRITONSERVER_MetricKind kind, const char* name, const char* description)
{
  auto registry = Metrics::GetRegistry();

  switch (kind) {
    case TRITONSERVER_METRIC_KIND_COUNTER:
      family_ = reinterpret_cast<void*>(&prometheus::BuildCounter()
                                             .Name(name)
                                             .Help(description)
                                             .Register(*registry));
      break;
    case TRITONSERVER_METRIC_KIND_GAUGE:
      family_ = reinterpret_cast<void*>(&prometheus::BuildGauge()
                                             .Name(name)
                                             .Help(description)
                                             .Register(*registry));
      break;
    default:
      throw std::invalid_argument(
          "Unsupported kind passed to MetricFamily constructor.");
  }

  kind_ = kind;
}

void*
MetricFamily::Add(std::map<std::string, std::string> label_map)
{
  void* metric = nullptr;
  switch (kind_) {
    case TRITONSERVER_METRIC_KIND_COUNTER: {
      auto counter_family_ptr =
          reinterpret_cast<prometheus::Family<prometheus::Counter>*>(family_);
      auto counter_ptr = &counter_family_ptr->Add(label_map);
      metric = reinterpret_cast<void*>(counter_ptr);
      break;
    }
    case TRITONSERVER_METRIC_KIND_GAUGE: {
      auto gauge_family_ptr =
          reinterpret_cast<prometheus::Family<prometheus::Gauge>*>(family_);
      auto gauge_ptr = &gauge_family_ptr->Add(label_map);
      metric = reinterpret_cast<void*>(gauge_ptr);
      break;
    }
    default:
      throw std::invalid_argument(
          "Unsupported family kind passed to Metric constructor.");
  }

  std::lock_guard<std::mutex> lk(mtx_);
  ++metric_ref_cnt_[metric];
  return metric;
}

void
MetricFamily::Remove(void* metric)
{
  {
    std::lock_guard<std::mutex> lk(mtx_);
    const auto it = metric_ref_cnt_.find(metric);
    if (it != metric_ref_cnt_.end()) {
      --it->second;
      if (it->second == 0) {
        metric_ref_cnt_.erase(it);
      } else {
        // Done as it is not the last reference
        return;
      }
    }
  }
  switch (kind_) {
    case TRITONSERVER_METRIC_KIND_COUNTER: {
      auto counter_family_ptr =
          reinterpret_cast<prometheus::Family<prometheus::Counter>*>(family_);
      auto counter_ptr = reinterpret_cast<prometheus::Counter*>(metric);
      counter_family_ptr->Remove(counter_ptr);
      break;
    }
    case TRITONSERVER_METRIC_KIND_GAUGE: {
      auto gauge_family_ptr =
          reinterpret_cast<prometheus::Family<prometheus::Gauge>*>(family_);
      auto gauge_ptr = reinterpret_cast<prometheus::Gauge*>(metric);
      gauge_family_ptr->Remove(gauge_ptr);
      break;
    }
    default:
      // Invalid kind should be caught in constructor
      FLARE_LOG(ERROR) << "Unsupported kind in Metric destructor.";
      break;
  }
}

MetricFamily::~MetricFamily()
{
  // NOTE: registry->Remove() not added until until prometheus-cpp v1.0 which
  // we do not currently install
}

//
// Implementation for TRITONSERVER_Metric.
//
Metric::Metric(
    TRITONSERVER_MetricFamily* family,
    std::vector<const inference_parameter*> labels)
{
  family_ = reinterpret_cast<MetricFamily*>(family);
  kind_ = family_->Kind();

  // Create map of labels from InferenceParameters
  std::map<std::string, std::string> label_map;
  for (const auto& param : labels) {
    if (param->Type() != TRITONSERVER_PARAMETER_STRING) {
      throw std::invalid_argument(
          "Parameter [" + param->Name() +
          "] must have a type of TRITONSERVER_PARAMETER_STRING to be "
          "added as a label.");
    }

    label_map[param->Name()] =
        std::string(reinterpret_cast<const char*>(param->ValuePointer()));
  }

  metric_ = family_->Add(label_map);
}

Metric::~Metric()
{
  family_->Remove(metric_);
}

TRITONSERVER_Error*
Metric::Value(double* value)
{
  switch (kind_) {
    case TRITONSERVER_METRIC_KIND_COUNTER: {
      auto counter_ptr = reinterpret_cast<prometheus::Counter*>(metric_);
      FLARE_LOG(DEBUG) << "SETTING COUNTER METRIC FROM: " << *value << " to "
                     << counter_ptr->Value();
      *value = counter_ptr->Value();
      break;
    }
    case TRITONSERVER_METRIC_KIND_GAUGE: {
      auto gauge_ptr = reinterpret_cast<prometheus::Gauge*>(metric_);
      FLARE_LOG(DEBUG) << "SETTING GAUGE METRIC FROM: " << *value << " to "
                     << gauge_ptr->Value();
      *value = gauge_ptr->Value();
      break;
    }
    default:
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_UNSUPPORTED,
          "Unsupported TRITONSERVER_MetricKind");
  }

  return nullptr;  // Success
}

TRITONSERVER_Error*
Metric::Increment(double value)
{
  switch (kind_) {
    case TRITONSERVER_METRIC_KIND_COUNTER: {
      if (value < 0.0) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            "TRITONSERVER_METRIC_KIND_COUNTER can only be incremented "
            "monotonically by non-negative values.");
      }

      auto counter_ptr = reinterpret_cast<prometheus::Counter*>(metric_);
      counter_ptr->Increment(value);
      break;
    }
    case TRITONSERVER_METRIC_KIND_GAUGE: {
      auto gauge_ptr = reinterpret_cast<prometheus::Gauge*>(metric_);
      // Gauge::Increment works for both positive and negative values as of
      // prometheus-cpp v1.0 but for now on v0.7 we defer call to
      // Increment/Decrement based on the sign of value
      // https://github.com/jupp0r/prometheus-cpp/blob/master/core/src/gauge.cc
      if (value < 0.0) {
        gauge_ptr->Decrement(-1.0 * value);
      } else {
        gauge_ptr->Increment(value);
      }
      break;
    }
    default:
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_UNSUPPORTED,
          "Unsupported TRITONSERVER_MetricKind");
  }

  return nullptr;  // Success
}

TRITONSERVER_Error*
Metric::Set(double value)
{
  switch (kind_) {
    case TRITONSERVER_METRIC_KIND_COUNTER: {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_UNSUPPORTED,
          "TRITONSERVER_METRIC_KIND_COUNTER does not support Set");
    }
    case TRITONSERVER_METRIC_KIND_GAUGE: {
      auto gauge_ptr = reinterpret_cast<prometheus::Gauge*>(metric_);
      gauge_ptr->Set(value);
      break;
    }
    default:
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_UNSUPPORTED,
          "Unsupported TRITONSERVER_MetricKind");
  }

  return nullptr;  // Success
}

}  // namespace hercules::core

#endif  // HERCULES_ENABLE_METRICS
