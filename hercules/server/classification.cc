
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/


#include "classification.h"

#include <algorithm>
#include <numeric>
#include "common.h"

namespace triton { namespace server {

namespace {

template <typename T>
TRITONSERVER_Error*
AddClassResults(
    TRITONSERVER_InferenceResponse* response, const uint32_t output_idx,
    const char* base, const size_t element_cnt, const uint32_t req_class_cnt,
    std::vector<std::string>* class_strs)
{
  const T* probs = reinterpret_cast<const T*>(base);

  std::vector<size_t> idx(element_cnt);
  iota(idx.begin(), idx.end(), 0);
  sort(idx.begin(), idx.end(), [&probs](size_t i1, size_t i2) {
    return probs[i1] > probs[i2];
  });

  const size_t class_cnt = std::min(element_cnt, (size_t)req_class_cnt);
  for (size_t k = 0; k < class_cnt; ++k) {
    class_strs->push_back(
        std::to_string(probs[idx[k]]) + ":" + std::to_string(idx[k]));

    const char* label;
    RETURN_IF_ERR(TRITONSERVER_InferenceResponseOutputClassificationLabel(
        response, output_idx, idx[k], &label));
    if (label != nullptr) {
      class_strs->back() += ":";
      class_strs->back().append(label);
    }
  }

  return nullptr;  // success
}

}  // namespace


TRITONSERVER_Error*
TopkClassifications(
    TRITONSERVER_InferenceResponse* response, const uint32_t output_idx,
    const char* base, const size_t byte_size,
    const TRITONSERVER_DataType datatype, const uint32_t req_class_count,
    std::vector<std::string>* class_strs)
{
  const size_t element_cnt =
      byte_size / TRITONSERVER_DataTypeByteSize(datatype);

  switch (datatype) {
    case TRITONSERVER_TYPE_UINT8:
      return AddClassResults<uint8_t>(
          response, output_idx, base, element_cnt, req_class_count, class_strs);
    case TRITONSERVER_TYPE_UINT16:
      return AddClassResults<uint16_t>(
          response, output_idx, base, element_cnt, req_class_count, class_strs);
    case TRITONSERVER_TYPE_UINT32:
      return AddClassResults<uint32_t>(
          response, output_idx, base, element_cnt, req_class_count, class_strs);
    case TRITONSERVER_TYPE_UINT64:
      return AddClassResults<uint64_t>(
          response, output_idx, base, element_cnt, req_class_count, class_strs);

    case TRITONSERVER_TYPE_INT8:
      return AddClassResults<int8_t>(
          response, output_idx, base, element_cnt, req_class_count, class_strs);
    case TRITONSERVER_TYPE_INT16:
      return AddClassResults<int16_t>(
          response, output_idx, base, element_cnt, req_class_count, class_strs);
    case TRITONSERVER_TYPE_INT32:
      return AddClassResults<int32_t>(
          response, output_idx, base, element_cnt, req_class_count, class_strs);
    case TRITONSERVER_TYPE_INT64:
      return AddClassResults<int64_t>(
          response, output_idx, base, element_cnt, req_class_count, class_strs);

    case TRITONSERVER_TYPE_FP32:
      return AddClassResults<float>(
          response, output_idx, base, element_cnt, req_class_count, class_strs);
    case TRITONSERVER_TYPE_FP64:
      return AddClassResults<double>(
          response, output_idx, base, element_cnt, req_class_count, class_strs);

    default:
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          std::string(
              std::string("class result not available for output due to "
                          "unsupported type '") +
              std::string(TRITONSERVER_DataTypeString(datatype)) + "'")
              .c_str());
  }

  return nullptr;  // success
}

}}  // namespace triton::server
