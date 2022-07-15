
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/

#pragma once

#include <deque>
#include <unordered_map>
#include "scheduler.h"

namespace hercules::core {

// A collection of inputs in the request, an nullptr for inference_request::Input
// indicates that the inputs doesn't require equality check
using RequiredEqualInputs = std::unordered_map<
    std::string,
    std::pair<const inference_request::Input*, bool /* compare contents */>>;

Status InitRequiredEqualInputs(
    const std::unique_ptr<inference_request>& request,
    const std::unordered_map<std::string, bool>& enforce_equal_shape_tensors,
    const bool has_optional_input, RequiredEqualInputs* required_equal_inputs);

bool CompareWithRequiredEqualInputs(
    const std::unique_ptr<inference_request>& request,
    const bool has_optional_input,
    const RequiredEqualInputs& required_equal_inputs);

//
// PriorityQueue
//
using ModelQueuePolicyMap = ::google::protobuf::Map<
    ::google::protobuf::uint32, hercules::proto::ModelQueuePolicy>;

class PriorityQueue {
 public:
  // Construct a queue with no priority level with default queue policy,
  // which will behave the same as regular queue.
  PriorityQueue();

  // Construct a queue with 'priority_levels', the priority starts from 1.
  // Different priority level may follow different queue policies given by
  // 'queue_policy_map', otherwise, the 'default_queue_policy' will be used.
  PriorityQueue(
      const hercules::proto::ModelQueuePolicy& default_queue_policy,
      uint32_t priority_levels, const ModelQueuePolicyMap queue_policy_map);

  // Enqueue a request with priority set to 'priority_level'. If
  // Status::Success is returned then the queue has taken ownership of
  // the request object and so 'request' will be nullptr. If
  // non-success is returned then the caller still retains ownership
  // of 'request'.
  Status Enqueue(
      uint32_t priority_level, std::unique_ptr<inference_request>& request);

  // Dequeue the request at the front of the queue.
  Status Dequeue(std::unique_ptr<inference_request>* request);

  // Retrieve the requests that are rejected based on the queue policies.
  void ReleaseRejectedRequests(
      std::shared_ptr<
          std::vector<std::deque<std::unique_ptr<inference_request>>>>*
          requests);

  // Return the number of requests in the queue, rejected requests are
  // not included.
  size_t Size() { return size_; }

  // Is the queue is empty? Rejected requests are not included.
  bool Empty() { return Size() == 0; }

  // Reset the cursor such that it is representing an empty pending batch.
  void ResetCursor() { pending_cursor_ = Cursor(queues_.begin()); }

  // Record the current cursor. The cursor can be restored to recorded state
  // by invoking SetCursorToMark(). Note that Enqueue(), Dequeue(), and
  // ResetCursor() will invalidate the marker, it is the function caller's
  // responsibility to ensure the marker is valid before calling
  // SetCursorToMark().
  void MarkCursor() { current_mark_ = pending_cursor_; }

  // Apply the queue policy and alter the underlying queue accordingly. After
  // the function returns, the cursor may be at its end to indicate that
  // there no request after the pending batch.
  // Returns the total batch size of the newly rejected requests.
  size_t ApplyPolicyAtCursor();

  // Return the request at the cursor.
  const std::unique_ptr<inference_request>& RequestAtCursor()
  {
    return pending_cursor_.curr_it_->second.At(pending_cursor_.queue_idx_);
  }

  // Advance the cursor for pending batch. This function will not trigger the
  // queue policy. No effect if the cursor already reach the end of the queue.
  void AdvanceCursor();

  // Whether the cursor reaches its end,
  bool CursorEnd() { return pending_cursor_.pending_batch_count_ == size_; }

  // Restore the cursor state to the marker.
  void SetCursorToMark() { pending_cursor_ = current_mark_; }

  // Whether the cursor is still valid. The cursor is valid only if the pending
  // batch is unchanged.
  bool IsCursorValid();

  // Return the oldest queued time of requests in pending batch.
  uint64_t OldestEnqueueTime()
  {
    return pending_cursor_.pending_batch_oldest_enqueue_time_ns_;
  }

  // Return the closest timeout of requests in pending batch.
  uint64_t ClosestTimeout()
  {
    return pending_cursor_.pending_batch_closest_timeout_ns_;
  }

  // Return the number of requests in pending batch.
  size_t PendingBatchCount() { return pending_cursor_.pending_batch_count_; }

 private:
  class PolicyQueue {
   public:
    // Construct a policy queue with default policy, which will behave the same
    // as regular queue.
    PolicyQueue()
        : timeout_action_(hercules::proto::ModelQueuePolicy::REJECT),
          default_timeout_us_(0), allow_timeout_override_(false),
          max_queue_size_(0)
    {
    }

    // Construct a policy queue with given 'policy'.
    PolicyQueue(const hercules::proto::ModelQueuePolicy& policy)
        : timeout_action_(policy.timeout_action()),
          default_timeout_us_(policy.default_timeout_microseconds()),
          allow_timeout_override_(policy.allow_timeout_override()),
          max_queue_size_(policy.max_queue_size())
    {
    }

    // Enqueue a request and set up its timeout accordingly. If
    // Status::Success is returned then the queue has taken ownership
    // of the request object and so 'request' will be nullptr. If
    // non-success is returned then the caller still retains ownership
    // of 'request'.
    Status Enqueue(std::unique_ptr<inference_request>& request);

    // Dequeue the request at the front of the queue.
    Status Dequeue(std::unique_ptr<inference_request>* request);

    // Apply the queue policy to the request at 'idx'.
    // 'rejected_count' will be incremented by the number of the newly rejected
    // requets after applying the policy.
    // 'rejected_batch_size' will be incremented by the total batch size of the
    // newly rejected requests after applying the policy.
    // Return true if the 'idx' still points to a request after applying the
    // policy, false otherwise.
    bool ApplyPolicy(
        size_t idx, size_t* rejected_count, size_t* rejected_batch_size);

    // Return the rejected requests held by the queue.
    void ReleaseRejectedQueue(
        std::deque<std::unique_ptr<inference_request>>* requests);

    // Return the request at 'idx'.
    const std::unique_ptr<inference_request>& At(size_t idx) const;

    // Return the timeout timestamp of the request at 'idx', in ns. A value of 0
    // indicates that the request doesn't specify a timeout.
    uint64_t TimeoutAt(size_t idx);

    // Return whether the queue is empty, rejected requests are not included.
    bool Empty() { return Size() == 0; }

    // Return the number of requests in the queue, rejected requests are not
    // included.
    size_t Size() { return queue_.size() + delayed_queue_.size(); }

    // Return the number of unexpired requests in the queue
    size_t UnexpiredSize() { return queue_.size(); }

   private:
    // Variables that define the policy for the queue
    const hercules::proto::ModelQueuePolicy::TimeoutAction timeout_action_;
    const uint64_t default_timeout_us_;
    const bool allow_timeout_override_;
    const uint32_t max_queue_size_;

    std::deque<uint64_t> timeout_timestamp_ns_;
    std::deque<std::unique_ptr<inference_request>> queue_;
    std::deque<std::unique_ptr<inference_request>> delayed_queue_;
    std::deque<std::unique_ptr<inference_request>> rejected_queue_;
  };
  using PriorityQueues = std::map<uint32_t, PolicyQueue>;

  // Cursor for tracking pending batch, the cursor points to the item after
  // the pending batch.
  struct Cursor {
    Cursor() = default;
    Cursor(PriorityQueues::iterator start_it);

    Cursor(const Cursor& rhs) = default;
    Cursor& operator=(const Cursor& rhs) = default;

    PriorityQueues::iterator curr_it_;
    size_t queue_idx_;
    bool at_delayed_queue_;
    uint64_t pending_batch_closest_timeout_ns_;
    uint64_t pending_batch_oldest_enqueue_time_ns_;
    size_t pending_batch_count_;
    bool valid_;
  };

  PriorityQueues queues_;
  size_t size_;

  // Keep track of the priority level that the first request in the queue
  // is at to avoid traversing 'queues_'
  uint32_t front_priority_level_;
  uint32_t last_priority_level_;

  Cursor pending_cursor_;
  Cursor current_mark_;
};

}  // namespace hercules::core
