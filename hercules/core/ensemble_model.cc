
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/


#include "ensemble_model.h"

#include <stdint.h>
#include "constants.h"
#include "ensemble_scheduler.h"
#include "model_config_utils.h"
#include "hercules/common/logging.h"

namespace hercules::core {

Status
EnsembleModel::Create(
    InferenceServer* server, const std::string& path, const int64_t version,
    const hercules::proto::ModelConfig& model_config,
    const double min_compute_capability, std::unique_ptr<Model>* model)
{
  // Create the ensemble model.
  std::unique_ptr<EnsembleModel> local_model(
      new EnsembleModel(min_compute_capability, path, version, model_config));

  RETURN_IF_ERROR(local_model->Init());

  std::unique_ptr<Scheduler> scheduler;
  RETURN_IF_ERROR(EnsembleScheduler::Create(
      local_model->MutableStatsAggregator(), server, model_config, &scheduler));
  RETURN_IF_ERROR(local_model->SetScheduler(std::move(scheduler)));

  LOG_VERBOSE(1) << "ensemble model for " << local_model->Name() << std::endl;

  *model = std::move(local_model);
  return Status::Success;
}

std::ostream&
operator<<(std::ostream& out, const EnsembleModel& pb)
{
  out << "name=" << pb.Name() << std::endl;
  return out;
}

}  // namespace hercules::core
