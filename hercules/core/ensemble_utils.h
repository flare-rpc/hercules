
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/

#pragma once

#ifdef TRITON_ENABLE_ENSEMBLE

#include <deque>
#include <unordered_map>
#include "hercules/proto/model_config.pb.h"
#include "model_repository_manager.h"
#include "status.h"
#include "hercules/common/model_config.h"

namespace hercules::core {

/// Validate that the ensemble are specified correctly. Assuming that the
/// inputs and outputs specified in depending model configurations are accurate.
/// \param model_repository_manager The model manager to acquire model config.
/// \param ensemble The ensemble to be validated.
/// \return The error status.
Status ValidateEnsembleConfig(
    ModelRepositoryManager* model_repository_manager,
    ModelRepositoryManager::DependencyNode* ensemble);

}  // namespace hercules::core

#endif  // TRITON_ENABLE_ENSEMBLE
