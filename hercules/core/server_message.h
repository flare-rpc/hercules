
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/

#pragma once

#include <string>
#include "status.h"

#define TRITONJSON_STATUSTYPE hercules::core::Status
#define TRITONJSON_STATUSRETURN(M) \
  return hercules::core::Status(hercules::core::Status::Code::INTERNAL, (M))
#define TRITONJSON_STATUSSUCCESS hercules::core::Status::Success
#include "hercules/common/triton_json.h"

namespace hercules::core {

//
// Implementation for TRITONSERVER_Message.
//
class TritonServerMessage {
 public:
  TritonServerMessage(const hercules::common::TritonJson::Value& msg)
  {
    json_buffer_.Clear();
    msg.Write(&json_buffer_);
    base_ = json_buffer_.Base();
    byte_size_ = json_buffer_.Size();
    from_json_ = true;
  }

  TritonServerMessage(std::string&& msg)
  {
    str_buffer_ = std::move(msg);
    base_ = str_buffer_.data();
    byte_size_ = str_buffer_.size();
    from_json_ = false;
  }

  TritonServerMessage(const TritonServerMessage& rhs)
  {
    from_json_ = rhs.from_json_;
    if (from_json_) {
      json_buffer_ = rhs.json_buffer_;
      base_ = json_buffer_.Base();
      byte_size_ = json_buffer_.Size();
    } else {
      str_buffer_ = rhs.str_buffer_;
      base_ = str_buffer_.data();
      byte_size_ = str_buffer_.size();
    }
  }

  void Serialize(const char** base, size_t* byte_size) const
  {
    *base = base_;
    *byte_size = byte_size_;
  }

 private:
  bool from_json_;
  hercules::common::TritonJson::WriteBuffer json_buffer_;
  std::string str_buffer_;

  const char* base_;
  size_t byte_size_;
};

}  // namespace hercules::core
