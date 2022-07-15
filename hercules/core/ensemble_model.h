
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/

#pragma once

#include "model.h"
#include "hercules/proto/model_config.pb.h"
#include "scheduler.h"
#include "status.h"

namespace hercules::core {

class InferenceServer;

class EnsembleModel : public Model {
 public:
  EnsembleModel(EnsembleModel&&) = default;

  static Status Create(
      InferenceServer* server, const std::string& path, const int64_t version,
      const hercules::proto::ModelConfig& model_config,
      const double min_compute_capability, std::unique_ptr<Model>* model);

 private:
  DISALLOW_COPY_AND_ASSIGN(EnsembleModel);

  explicit EnsembleModel(
      const double min_compute_capability, const std::string& model_dir,
      const int64_t version, const hercules::proto::ModelConfig& config)
      : Model(min_compute_capability, model_dir, version, config)
  {
  }
  friend std::ostream& operator<<(std::ostream&, const EnsembleModel&);
};

std::ostream& operator<<(std::ostream& out, const EnsembleModel& pb);

}  // namespace hercules::core
