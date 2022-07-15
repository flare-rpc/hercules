
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/


#include "instance_queue.h"

#include "hercules/common/logging.h"

namespace hercules::core {

InstanceQueue::InstanceQueue(size_t max_batch_size, uint64_t max_queue_delay_ns)
    : max_batch_size_(max_batch_size), max_queue_delay_ns_(max_queue_delay_ns)
{
}

size_t
InstanceQueue::Size()
{
  return payload_queue_.size();
}

bool
InstanceQueue::Empty()
{
  return payload_queue_.empty();
}

void
InstanceQueue::Enqueue(const std::shared_ptr<Payload>& payload)
{
  payload_queue_.push_back(payload);
}

void
InstanceQueue::Dequeue(
    std::shared_ptr<Payload>* payload,
    std::vector<std::shared_ptr<Payload>>* merged_payloads)
{
  *payload = payload_queue_.front();
  payload_queue_.pop_front();
  {
    std::lock_guard<std::mutex> exec_lock(*((*payload)->GetExecMutex()));
    (*payload)->SetState(Payload::State::EXECUTING);
    if ((!payload_queue_.empty()) && (max_queue_delay_ns_ > 0) &&
        (max_batch_size_ > 1) && (!(*payload)->IsSaturated())) {
      bool continue_merge;
      do {
        continue_merge = false;
        uint64_t now_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch())
                .count();
        size_t batch_size = (*payload)->BatchSize();
        if ((!payload_queue_.empty()) &&
            (!payload_queue_.front()->IsSaturated()) &&
            (now_ns - payload_queue_.front()->BatcherStartNs()) >
                max_queue_delay_ns_) {
          std::lock_guard<std::mutex> exec_lock(
              *(payload_queue_.front()->GetExecMutex()));
          payload_queue_.front()->SetState(Payload::State::EXECUTING);
          size_t front_batch_size = payload_queue_.front()->BatchSize();
          if ((batch_size + front_batch_size) <= max_batch_size_) {
            auto status = (*payload)->MergePayload(payload_queue_.front());
            if (status.IsOk()) {
              merged_payloads->push_back(payload_queue_.front());
              payload_queue_.pop_front();
              continue_merge = true;
            }
          }
        }
      } while (continue_merge);
    }
  }
}

}  // namespace hercules::core
