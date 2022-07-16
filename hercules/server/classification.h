
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/

#pragma once

#include <string>
#include <vector>
#include "hercules/core/tritonserver.h"

namespace triton { namespace server {

TRITONSERVER_Error* TopkClassifications(
    TRITONSERVER_InferenceResponse* response, const uint32_t output_idx,
    const char* base, const size_t byte_size,
    const TRITONSERVER_DataType datatype, const uint32_t req_class_count,
    std::vector<std::string>* class_strs);

}}  // namespace triton::server
