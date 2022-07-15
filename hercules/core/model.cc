
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/


#include "model.h"

#include <chrono>
#include <future>
#include "constants.h"
#include "filesystem.h"
#include "infer_request.h"
#include "model_config_utils.h"
#include "hercules/common/logging.h"

namespace hercules::core {

Status
Model::GetInput(
    const std::string& name, const hercules::proto::ModelInput** input) const
{
  const auto itr = input_map_.find(name);
  if (itr == input_map_.end()) {
    return Status(
        Status::Code::INVALID_ARG,
        "unexpected inference input '" + name + "' for model '" + Name() + "'");
  }

  *input = &itr->second;
  return Status::Success;
}

Status
Model::GetOutput(
    const std::string& name, const hercules::proto::ModelOutput** output) const
{
  const auto itr = output_map_.find(name);
  if (itr == output_map_.end()) {
    return Status(
        Status::Code::INVALID_ARG, "unexpected inference output '" + name +
                                       "' for model '" + Name() + "'");
  }

  *output = &itr->second;
  return Status::Success;
}

Status
Model::SetModelConfig(const hercules::proto::ModelConfig& config)
{
  config_ = config;
  return Status::Success;
}

Status
Model::SetScheduler(std::unique_ptr<Scheduler> scheduler)
{
  if (scheduler_ != nullptr) {
    return Status(
        Status::Code::INTERNAL, "Attempt to change scheduler not allowed");
  }

  scheduler_ = std::move(scheduler);
  return Status::Success;
}

Status
Model::Init()
{
  RETURN_IF_ERROR(ValidateModelConfig(config_, min_compute_capability_));
  RETURN_IF_ERROR(ValidateModelIOConfig(config_));

  // Initialize the input map
  for (const auto& io : config_.input()) {
    input_map_.insert(std::make_pair(io.name(), io));
    if (!io.optional()) {
      ++required_input_count_;
    }
  }

  // Initialize the output map and label provider for each output
  label_provider_ = std::make_shared<LabelProvider>();
  for (const auto& io : config_.output()) {
    output_map_.insert(std::make_pair(io.name(), io));

    if (!io.label_filename().empty()) {
      const auto label_path = JoinPath({model_dir_, io.label_filename()});
      RETURN_IF_ERROR(label_provider_->AddLabels(io.name(), label_path));
    }
  }

  if (config_.has_dynamic_batching()) {
    default_priority_level_ =
        config_.dynamic_batching().default_priority_level();
    max_priority_level_ = config_.dynamic_batching().priority_levels();
  } else if (config_.has_ensemble_scheduling()) {
    // For ensemble, allow any priority level to pass through
    default_priority_level_ = 0;
    max_priority_level_ = UINT32_MAX;
  } else {
    default_priority_level_ = 0;
    max_priority_level_ = 0;
  }

  return Status::Success;
}

}  // namespace hercules::core
