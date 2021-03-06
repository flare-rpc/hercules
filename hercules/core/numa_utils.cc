
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/

#include "numa_utils.h"

#ifdef __linux__
#include <numa.h>
#include <numaif.h>
#endif
#include "hercules/common/logging.h"

namespace hercules::core {

namespace {
std::string
VectorToString(const std::vector<int>& vec)
{
  std::string str("[");
  for (const auto& element : vec) {
    str += std::to_string(element);
    str += ",";
  }

  str += "]";
  return str;
}

Status
ParseIntOption(const std::string& msg, const std::string& arg, int* value)
{
  try {
    *value = std::stoi(arg);
  }
  catch (const std::invalid_argument& ia) {
    return Status(
        Status::Code::INVALID_ARG,
        msg + ": Can't parse '" + arg + "' to integer");
  }
  return Status::Success;
}

}  // namespace

// NUMA setting will be ignored on Windows platform
#if defined(_WIN32) || defined(__MACH__)
Status
SetNumaConfigOnThread(
    const hercules::common::HostPolicyCmdlineConfig& host_policy)
{
  return Status::Success;
}

Status
SetNumaMemoryPolicy(const hercules::common::HostPolicyCmdlineConfig& host_policy)
{
  return Status::Success;
}

Status
GetNumaMemoryPolicyNodeMask(unsigned long* node_mask)
{
  *node_mask = 0;
  return Status::Success;
}

Status
ResetNumaMemoryPolicy()
{
  return Status::Success;
}

Status
SetNumaThreadAffinity(
    std::thread::native_handle_type thread,
    const hercules::common::HostPolicyCmdlineConfig& host_policy)
{
  return Status::Success;
}
#else
// Use variable to make sure no NUMA related function is actually called
// if Triton is not running with NUMA awareness. i.e. Extra docker permission
// is needed to call the NUMA functions and this ensures backward compatibility.
thread_local bool numa_set = false;

Status
SetNumaConfigOnThread(
    const hercules::common::HostPolicyCmdlineConfig& host_policy)
{
  // Set thread affinity
  RETURN_IF_ERROR(SetNumaThreadAffinity(pthread_self(), host_policy));

  // Set memory policy
  RETURN_IF_ERROR(SetNumaMemoryPolicy(host_policy));

  return Status::Success;
}

Status
SetNumaMemoryPolicy(const hercules::common::HostPolicyCmdlineConfig& host_policy)
{
  const auto it = host_policy.find("numa-node");
  if (it != host_policy.end()) {
    int node_id;
    RETURN_IF_ERROR(
        ParseIntOption("Parsing 'numa-node' value", it->second, &node_id));
    FLARE_LOG(DEBUG) << "Thread is binding to NUMA node " << it->second
                   << ". Max NUMA node count: " << (numa_max_node() + 1);
    numa_set = true;
    unsigned long node_mask = 1UL << node_id;
    if (set_mempolicy(MPOL_BIND, &node_mask, (numa_max_node() + 1) + 1) != 0) {
      return Status(
          Status::Code::INTERNAL,
          std::string("Unable to set NUMA memory policy: ") + strerror(errno));
    }
  }
  return Status::Success;
}

Status
GetNumaMemoryPolicyNodeMask(unsigned long* node_mask)
{
  *node_mask = 0;
  int mode;
  if (numa_set &&
      get_mempolicy(&mode, node_mask, numa_max_node() + 1, NULL, 0) != 0) {
    return Status(
        Status::Code::INTERNAL,
        std::string("Unable to get NUMA node for current thread: ") +
            strerror(errno));
  }
  return Status::Success;
}

Status
ResetNumaMemoryPolicy()
{
  if (numa_set && (set_mempolicy(MPOL_DEFAULT, nullptr, 0) != 0)) {
    return Status(
        Status::Code::INTERNAL,
        std::string("Unable to reset NUMA memory policy: ") + strerror(errno));
  }
  numa_set = false;
  return Status::Success;
}

Status
SetNumaThreadAffinity(
    std::thread::native_handle_type thread,
    const hercules::common::HostPolicyCmdlineConfig& host_policy)
{
  const auto it = host_policy.find("cpu-cores");
  if (it != host_policy.end()) {
    // Parse CPUs
    std::vector<int> cpus;
    {
      const auto& cpu_str = it->second;
      auto delim_cpus = cpu_str.find(",");
      int current_pos = 0;
      while (true) {
        auto delim_range = cpu_str.find("-", current_pos);
        if (delim_range == std::string::npos) {
          return Status(
              Status::Code::INVALID_ARG,
              std::string("host policy setting 'cpu-cores' format is "
                          "'<lower_cpu_core_id>-<upper_cpu_core_id>'. Got ") +
                  cpu_str.substr(
                      current_pos, ((delim_cpus == std::string::npos)
                                        ? (cpu_str.length() + 1)
                                        : delim_cpus) -
                                       current_pos));
        }
        int lower, upper;
        RETURN_IF_ERROR(ParseIntOption(
            "Parsing 'cpu-cores' value",
            cpu_str.substr(current_pos, delim_range - current_pos), &lower));
        RETURN_IF_ERROR(ParseIntOption(
            "Parsing 'cpu-cores' value",
            (delim_cpus == std::string::npos)
                ? cpu_str.substr(delim_range + 1)
                : cpu_str.substr(
                      delim_range + 1, delim_cpus - (delim_range + 1)),
            &upper));
        for (; lower <= upper; ++lower) {
          cpus.push_back(lower);
        }
        // break if the processed range is the last specified range
        if (delim_cpus != std::string::npos) {
          current_pos = delim_cpus + 1;
          delim_cpus = cpu_str.find(",", current_pos);
        } else {
          break;
        }
      }
    }

    FLARE_LOG(DEBUG) << "Thread is binding to one of the CPUs: "
                   << VectorToString(cpus);
    numa_set = true;
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (int cpu : cpus) {
      CPU_SET(cpu, &cpuset);
    }
    if (pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset) != 0) {
      return Status(
          Status::Code::INTERNAL,
          std::string("Unable to set NUMA thread affinity: ") +
              strerror(errno));
    }
  }
  return Status::Success;
}
#endif

}  // namespace hercules::core
