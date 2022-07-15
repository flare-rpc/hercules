
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/

#pragma once

#ifdef TRITON_ENABLE_ENSEMBLE

#include <memory>
#include "metric_model_reporter.h"
#include "hercules/proto/model_config.pb.h"
#include "model_config_utils.h"
#include "scheduler.h"
#include "status.h"

#ifdef TRITON_ENABLE_GPU
#include <cuda_runtime_api.h>
#endif  // TRITON_ENABLE_GPU

namespace hercules::core {

#ifndef TRITON_ENABLE_GPU
using cudaStream_t = void*;
#endif  // TRITON_ENABLE_GPU

class InferenceServer;

struct EnsembleInfo {
  struct StepInfo {
    StepInfo(const std::string& model_name, const int64_t model_version)
        : model_name_(model_name), model_version_(model_version)
    {
    }

    std::string model_name_;
    int64_t model_version_;
    std::unordered_map<std::string, std::string> input_to_tensor_;
    std::unordered_map<std::string, std::string> output_to_tensor_;
  };

  std::string ensemble_name_;

  bool is_decoupled_;

  // the ensemble output (re)shape expected by the ensemble
  std::unordered_map<std::string, triton::common::DimsList>
      ensemble_output_shape_;

  // Inputs that is marked optional for the ensemble
  std::set<std::string> optional_inputs_;

  std::vector<StepInfo> steps_;

  // Only include a step if the ensemble tensor is used as input in that step
  std::unordered_map<std::string, std::set<size_t>> tensor_to_step_;

  // backward path, ensemble tensor to the step that provides its data
  std::unordered_map<std::string, size_t> tensor_to_prev_step_;
};

// Scheduler that implements ensemble scheduling.
class EnsembleScheduler : public Scheduler {
 public:
  // Create a scheduler to process ensemble requests and
  // to dispatch requests to models in ensemble internally.
  static Status Create(
      InferenceStatsAggregator* const stats_aggregator,
      InferenceServer* const server, const hercules::proto::ModelConfig& config,
      std::unique_ptr<Scheduler>* scheduler);

  ~EnsembleScheduler();

  // \see Scheduler::Enqueue()
  Status Enqueue(std::unique_ptr<InferenceRequest>& request) override;

  // \see Scheduler::InflightInferenceCount()
  size_t InflightInferenceCount() override { return inflight_count_; }

  // \see Scheduler::Stop()
  void Stop() override {}

 private:
  EnsembleScheduler(
      InferenceStatsAggregator* const stats_aggregator,
      InferenceServer* const server, const hercules::proto::ModelConfig& config);

  std::shared_ptr<MetricModelReporter> metric_reporter_;
  InferenceStatsAggregator* const stats_aggregator_;
  InferenceServer* const is_;

  // Ensemble information that is built from model config
  std::unique_ptr<EnsembleInfo> info_;

  // The stream used for data transfer.
  cudaStream_t stream_;

  std::atomic<size_t> inflight_count_;
};

}  // namespace hercules::core

#endif  // TRITON_ENABLE_ENSEMBLE
