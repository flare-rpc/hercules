
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/

#pragma once

#include <map>
#include <thread>
#include <vector>
#include "status.h"
#include "hercules/common/model_config.h"
#include "tritonserver_apis.h"

namespace hercules::core {

// Helper function to set memory policy and thread affinity on current thread
Status SetNumaConfigOnThread(
    const hercules::common::HostPolicyCmdlineConfig& host_policy);

// Restrict the memory allocation to specific NUMA node.
Status SetNumaMemoryPolicy(
    const hercules::common::HostPolicyCmdlineConfig& host_policy);

// Retrieve the node mask used to set memory policy for the current thread
Status GetNumaMemoryPolicyNodeMask(unsigned long* node_mask);

// Reset the memory allocation setting.
Status ResetNumaMemoryPolicy();

// Set a thread affinity to be on specific cpus.
Status SetNumaThreadAffinity(
    std::thread::native_handle_type thread,
    const hercules::common::HostPolicyCmdlineConfig& host_policy);


}  // namespace hercules::core
