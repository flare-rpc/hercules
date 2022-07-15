
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/

#pragma once

#include "payload.h"

namespace hercules::core {

//
// InstanceQueue
//
// A queue implementation holding Payloads ready to be scheduled on
// model instance.
class InstanceQueue {
 public:
  explicit InstanceQueue(size_t max_batch_size, uint64_t max_queue_delay_ns);

  size_t Size();
  bool Empty();
  void Enqueue(const std::shared_ptr<Payload>& payload);
  void Dequeue(
      std::shared_ptr<Payload>* payload,
      std::vector<std::shared_ptr<Payload>>* merged_payloads);

 private:
  size_t max_batch_size_;
  uint64_t max_queue_delay_ns_;

  std::deque<std::shared_ptr<Payload>> payload_queue_;
  std::shared_ptr<Payload> staged_payload_;
  std::mutex mu_;
};

}  // namespace hercules::core
