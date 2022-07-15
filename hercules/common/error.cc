
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/


#include "hercules/common/error.h"

namespace hercules::common {

const Error Error::Success(Error::Code::SUCCESS);

std::string
Error::AsString() const
{
  std::string str(CodeString(code_));
  str += ": " + msg_;
  return str;
}

const char*
Error::CodeString(const Code code)
{
  switch (code) {
    case Error::Code::SUCCESS:
      return "OK";
    case Error::Code::UNKNOWN:
      return "Unknown";
    case Error::Code::INTERNAL:
      return "Internal";
    case Error::Code::NOT_FOUND:
      return "Not found";
    case Error::Code::INVALID_ARG:
      return "Invalid argument";
    case Error::Code::UNAVAILABLE:
      return "Unavailable";
    case Error::Code::UNSUPPORTED:
      return "Unsupported";
    case Error::Code::ALREADY_EXISTS:
      return "Already exists";
    default:
      break;
  }

  return "<invalid code>";
}

}  // namespace hercules::common
