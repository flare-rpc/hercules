
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/


#include "backend_config.h"

#include "status.h"
#include "hercules/common/logging.h"
#include "hercules/common/model_config.h"

namespace hercules::core {

    namespace {

        Status
        GetTFSpecializedBackendName(
                const hercules::common::BackendCmdlineConfigMap &config_map,
                std::string *specialized_name) {
            std::string tf_version_str = "1";
            const auto &itr = config_map.find("tensorflow");
            if (itr != config_map.end()) {
                if (BackendConfiguration(itr->second, "version", &tf_version_str).IsOk()) {
                    if ((tf_version_str != "1") && (tf_version_str != "2")) {
                        return Status(
                                Status::Code::INVALID_ARG,
                                "unexpected TensorFlow library version '" + tf_version_str +
                                "', expects 1 or 2.");
                    }
                }
            }

            *specialized_name += tf_version_str;

            return Status::Success;
        }

        Status
        GetOVSpecializedBackendName(
                const hercules::common::BackendCmdlineConfigMap &config_map,
                std::string *specialized_name) {
            std::string ov_version_str = "2021_4";
            const auto &itr = config_map.find("openvino");
            if (itr != config_map.end()) {
                BackendConfiguration(itr->second, "version", &ov_version_str);
            }

            *specialized_name += ("_" + ov_version_str);

            return Status::Success;
        }
    }  // namespace

    Status
    BackendConfiguration(
            const hercules::common::BackendCmdlineConfig &config, const std::string &key,
            std::string *val) {
        for (const auto &pr : config) {
            if (pr.first == key) {
                *val = pr.second;
                return Status::Success;
            }
        }

        return Status(
                Status::Code::INTERNAL,
                std::string("unable to find common backend configuration for '") + key +
                "'");
    }

    Status
    BackendConfigurationParseStringToDouble(const std::string &str, double *val) {
        try {
            *val = std::stod(str);
        }
        catch (...) {
            return Status(
                    Status::Code::INTERNAL,
                    "unable to parse common backend configuration as double");
        }

        return Status::Success;
    }

    Status
    BackendConfigurationParseStringToBool(const std::string &str, bool *val) {
        try {
            std::string lowercase_str{str};
            std::transform(
                    lowercase_str.begin(), lowercase_str.end(), lowercase_str.begin(),
                    [](unsigned char c) { return std::tolower(c); });
            *val = (lowercase_str == "true");
        }
        catch (...) {
            return Status(
                    Status::Code::INTERNAL,
                    "unable to parse common backend configuration as bool");
        }

        return Status::Success;
    }

    Status
    BackendConfigurationGlobalBackendsDirectory(
            const hercules::common::BackendCmdlineConfigMap &config_map, std::string *dir) {
        const auto &itr = config_map.find(std::string());
        if (itr == config_map.end()) {
            return Status(
                    Status::Code::INTERNAL,
                    "unable to find global backends directory configuration");
        }

        RETURN_IF_ERROR(BackendConfiguration(itr->second, "backend-directory", dir));

        return Status::Success;
    }

    Status
    BackendConfigurationMinComputeCapability(
            const hercules::common::BackendCmdlineConfigMap &config_map, double *mcc) {
#ifdef HERCULES_ENABLE_GPU
        *mcc = TRITON_MIN_COMPUTE_CAPABILITY;
#else
        *mcc = 0;
#endif  // HERCULES_ENABLE_GPU

        const auto &itr = config_map.find(std::string());
        if (itr == config_map.end()) {
            return Status(
                    Status::Code::INTERNAL, "unable to find common backend configuration");
        }

        std::string min_compute_capability_str;
        RETURN_IF_ERROR(BackendConfiguration(
                itr->second, "min-compute-capability", &min_compute_capability_str));
        RETURN_IF_ERROR(
                BackendConfigurationParseStringToDouble(min_compute_capability_str, mcc));

        return Status::Success;
    }

    Status
    BackendConfigurationAutoCompleteConfig(
            const hercules::common::BackendCmdlineConfigMap &config_map, bool *acc) {
        const auto &itr = config_map.find(std::string());
        if (itr == config_map.end()) {
            return Status(
                    Status::Code::INTERNAL, "unable to find auto-complete configuration");
        }

        std::string auto_complete_config_str;
        RETURN_IF_ERROR(BackendConfiguration(
                itr->second, "auto-complete-config", &auto_complete_config_str));
        RETURN_IF_ERROR(
                BackendConfigurationParseStringToBool(auto_complete_config_str, acc));

        return Status::Success;
    }

    Status
    BackendConfigurationSpecializeBackendName(
            const hercules::common::BackendCmdlineConfigMap &config_map,
            const std::string &backend_name, std::string *specialized_name) {
        *specialized_name = backend_name;
        if (backend_name == "tensorflow") {
            RETURN_IF_ERROR(GetTFSpecializedBackendName(config_map, specialized_name));
        } else if (backend_name == "openvino") {
            RETURN_IF_ERROR(GetOVSpecializedBackendName(config_map, specialized_name));
        }

        return Status::Success;
    }

    Status
    BackendConfigurationBackendLibraryName(
            const std::string &backend_name, std::string *libname) {
#ifdef _WIN32
        *libname = "triton_" + backend_name + ".dll";
#else
        *libname = "libtriton_" + backend_name + ".so";
#endif

        return Status::Success;
    }

}  // namespace hercules::core
