
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/

#pragma once

#include <string>

namespace hercules::common {

//
// Error
//
// Error returned by utilities from common repo.
//
class Error {
 public:
  enum class Code {
    SUCCESS,
    UNKNOWN,
    INTERNAL,
    NOT_FOUND,
    INVALID_ARG,
    UNAVAILABLE,
    UNSUPPORTED,
    ALREADY_EXISTS
  };

  explicit Error(Code code = Code::SUCCESS) : code_(code) {}
  explicit Error(Code code, const std::string& msg) : code_(code), msg_(msg) {}

  // Convenience "success" value. Can be used as Error::Success to
  // indicate no error.
  static const Error Success;

  // Return the code for this status.
  Code ErrorCode() const { return code_; }

  // Return the message for this status.
  const std::string& Message() const { return msg_; }

  // Return true if this status indicates "ok"/"success", false if
  // status indicates some kind of failure.
  bool IsOk() const { return code_ == Code::SUCCESS; }

  // Return the status as a string.
  std::string AsString() const;

  // Return the constant string name for a code.
  static const char* CodeString(const Code code);

 protected:
  Code code_;
  std::string msg_;
};

}  // namespace hercules::common
