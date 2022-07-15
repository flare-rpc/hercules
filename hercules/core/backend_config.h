
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/

#pragma once

#include "status.h"
#include "hercules/common/model_config.h"

namespace hercules::core {

    /// Get a key's string value from a backend configuration.
    Status BackendConfiguration(
            const triton::common::BackendCmdlineConfig &config, const std::string &key,
            std::string *val);

    /// Convert a backend configuration string  value into a double.
    Status BackendConfigurationParseStringToDouble(
            const std::string &str, double *val);

    /// Convert a backend configuration string  value into a bool.
    Status BackendConfigurationParseStringToBool(const std::string &str, bool *val);

    /// Get the global backends directory from the backend configuration.
    Status BackendConfigurationGlobalBackendsDirectory(
            const triton::common::BackendCmdlineConfigMap &config_map,
            std::string *dir);

    /// Get the minimum compute capability from the backend configuration.
    Status BackendConfigurationMinComputeCapability(
            const triton::common::BackendCmdlineConfigMap &config_map, double *mcc);

    /// Get the model configuration auto-complete setting from the backend
    /// configuration.
    Status BackendConfigurationAutoCompleteConfig(
            const triton::common::BackendCmdlineConfigMap &config_map, bool *acc);

    /// Convert a backend name to the specialized version of that name
    /// based on the backend configuration. For example, "tensorflow" will
    /// convert to either "tensorflow1" or "tensorflow2" depending on how
    /// tritonserver is run.
    Status BackendConfigurationSpecializeBackendName(
            const triton::common::BackendCmdlineConfigMap &config_map,
            const std::string &backend_name, std::string *specialized_name);

    /// Return the shared library name for a backend.
    Status BackendConfigurationBackendLibraryName(
            const std::string &backend_name, std::string *libname);

}  // namespace hercules::core
