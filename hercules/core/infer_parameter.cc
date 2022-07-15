
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/


#include "infer_parameter.h"

namespace hercules::core {


const void*
inference_parameter::ValuePointer() const
{
  switch (type_) {
    case TRITONSERVER_PARAMETER_STRING:
      return reinterpret_cast<const void*>(value_string_.c_str());
    case TRITONSERVER_PARAMETER_INT:
      return reinterpret_cast<const void*>(&value_int64_);
    case TRITONSERVER_PARAMETER_BOOL:
      return reinterpret_cast<const void*>(&value_bool_);
    case TRITONSERVER_PARAMETER_BYTES:
      return reinterpret_cast<const void*>(value_bytes_);
    default:
      break;
  }

  return nullptr;
}

std::ostream&
operator<<(std::ostream& out, const inference_parameter& parameter)
{
  out << "[0x" << std::addressof(parameter) << "] "
      << "name: " << parameter.Name()
      << ", type: " << TRITONSERVER_ParameterTypeString(parameter.Type())
      << ", value: ";
  return out;
}

}  // namespace hercules::core
