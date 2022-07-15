
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/

#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>
#include "backend_model.h"
#include "backend_model_instance.h"
#include "hercules/proto/model_config.pb.h"
#include "rate_limiter.h"
#include "scheduler.h"
#include "scheduler_utils.h"
#include "sequence_state.h"
#include "status.h"
#include "hercules/common/model_config.h"

namespace hercules::core {

class SequenceBatch;

// Scheduler that implements batching across sequences of correlated
// inferences.
class SequenceBatchScheduler : public Scheduler {
 public:
  using ControlInputs = std::vector<std::shared_ptr<inference_request::Input>>;

  SequenceBatchScheduler() = default;
  ~SequenceBatchScheduler();

  // Create a scheduler to support a given number of runners and a run
  // function to call when a request is scheduled.
  static Status Create(
      TritonModel* model,
      const std::unordered_map<std::string, bool>& enforce_equal_shape_tensors,
      std::unique_ptr<Scheduler>* scheduler);

  // \see Scheduler::Enqueue()
  Status Enqueue(std::unique_ptr<inference_request>& request) override;

  // \see Scheduler::InflightInferenceCount()
  size_t InflightInferenceCount() override
  {
    std::unique_lock<std::mutex> lock(mu_);
    return sequence_to_batcherseqslot_map_.size();
  }

  // \see Scheduler::Stop()
  void Stop() override { stop_ = true; }

  // A batcher-sequence_slot combination. The batcher is represented
  // by the index into 'batchers_'.
  struct BatcherSequenceSlot {
    BatcherSequenceSlot() = default;
    BatcherSequenceSlot(const BatcherSequenceSlot&) = default;
    BatcherSequenceSlot(size_t b, uint32_t s) : batcher_idx_(b), seq_slot_(s) {}
    size_t batcher_idx_;
    uint32_t seq_slot_;
  };

  // Fill a sequence slot with a sequence from the backlog or show
  // that the sequence slot is no longer being used.
  inference_request::SequenceId ReleaseSequenceSlot(
      const BatcherSequenceSlot& seq_slot,
      std::deque<std::unique_ptr<inference_request>>* requests);

  // For debugging/testing, batcher reports how many waiting requests
  // and returns true if the batcher should continue waiting.
  bool DelayScheduler(
      const uint32_t batcher_idx, const size_t cnt, const size_t total);

  const std::unordered_map<
      std::string, const hercules::proto::ModelSequenceBatching_State&>&
  StateOutputConfigMap()
  {
    return state_output_config_map_;
  }

  size_t MaxBatchSize() { return max_batch_size_; }
  const std::unordered_map<std::string, SequenceStates::InitialStateData>&
  InitialState()
  {
    return initial_state_;
  }

 private:
  void ReaperThread(const int nice);

  Status CreateBooleanControlTensors(
      const hercules::proto::ModelConfig& config,
      std::shared_ptr<ControlInputs>* start_input_overrides,
      std::shared_ptr<ControlInputs>* end_input_overrides,
      std::shared_ptr<ControlInputs>* startend_input_overrides,
      std::shared_ptr<ControlInputs>* continue_input_overrides,
      std::shared_ptr<ControlInputs>* notready_input_overrides);

  Status GenerateInitialStateData(
      const hercules::proto::ModelSequenceBatching_InitialState& initial_state,
      const hercules::proto::ModelSequenceBatching_State& state, TritonModel* model);

  struct BatcherSequenceSlotCompare {
    bool operator()(
        const BatcherSequenceSlot& a, const BatcherSequenceSlot& b) const
    {
      return a.seq_slot_ > b.seq_slot_;
    }
  };

  // The max_sequence_idle_microseconds value for this scheduler.
  uint64_t max_sequence_idle_microseconds_;

  bool stop_;

  // Mutex
  std::mutex mu_;

  // The reaper thread
  std::unique_ptr<std::thread> reaper_thread_;
  std::condition_variable reaper_cv_;
  bool reaper_thread_exit_;

  // The SequenceBatchs being managed by this scheduler.
  std::vector<std::unique_ptr<SequenceBatch>> batchers_;

  // Map from a request's correlation ID to the BatcherSequenceSlot
  // assigned to that correlation ID.
  using BatcherSequenceSlotMap =
      std::unordered_map<inference_request::SequenceId, BatcherSequenceSlot>;
  BatcherSequenceSlotMap sequence_to_batcherseqslot_map_;

  // Map from a request's correlation ID to the backlog queue
  // collecting requests for that correlation ID.
  using BacklogMap = std::unordered_map<
      inference_request::SequenceId,
      std::shared_ptr<std::deque<std::unique_ptr<inference_request>>>>;
  BacklogMap sequence_to_backlog_map_;

  // The ordered backlog of sequences waiting for a free sequenceslot.
  std::deque<std::shared_ptr<std::deque<std::unique_ptr<inference_request>>>>
      backlog_queues_;

  // The batcher/sequence-slot locations ready to accept a new
  // sequence. Ordered from lowest sequence-slot-number to highest so
  // that all batchers grow at the same rate and attempt to remain as
  // small as possible.
  std::priority_queue<
      BatcherSequenceSlot, std::vector<BatcherSequenceSlot>,
      BatcherSequenceSlotCompare>
      ready_batcher_seq_slots_;

  // For each correlation ID the most recently seen timestamp, in
  // microseconds, for a request using that correlation ID.
  std::unordered_map<inference_request::SequenceId, uint64_t>
      correlation_id_timestamps_;

  // Used for debugging/testing.
  size_t backlog_delay_cnt_;
  std::vector<size_t> queue_request_cnts_;

  // IO mapping between the output state name and the state configuration.
  std::unordered_map<std::string, const hercules::proto::ModelSequenceBatching_State&>
      state_output_config_map_;
  size_t max_batch_size_;

  // Initial state used for implicit state.
  std::unordered_map<std::string, SequenceStates::InitialStateData>
      initial_state_;
};

// Base class for a scheduler that implements a particular scheduling
// strategy for a model instance.
class SequenceBatch {
 public:
  SequenceBatch(
      SequenceBatchScheduler* base, const uint32_t batcher_idx,
      const size_t seq_slot_cnt,
      const std::unordered_map<std::string, bool>& enforce_equal_shape_tensors,
      const bool has_optional_input,
      const std::shared_ptr<SequenceBatchScheduler::ControlInputs>&
          start_input_overrides,
      const std::shared_ptr<SequenceBatchScheduler::ControlInputs>&
          end_input_overrides,
      const std::shared_ptr<SequenceBatchScheduler::ControlInputs>&
          startend_input_overrides,
      const std::shared_ptr<SequenceBatchScheduler::ControlInputs>&
          continue_input_overrides,
      const std::shared_ptr<SequenceBatchScheduler::ControlInputs>&
          notready_input_overrides);
  virtual ~SequenceBatch() = default;

  // Enqueue a request into the appropriate queue for the requested
  // sequence slot. This function takes ownership of 'request' so on
  // request 'request' will be nullptr.
  virtual void Enqueue(
      const uint32_t seq_slot,
      const inference_request::SequenceId& correlation_id,
      std::unique_ptr<inference_request>& request) = 0;

 protected:
  bool CreateCorrelationIDControl(const hercules::proto::ModelConfig& config);
  void SetControlTensors(
      std::unique_ptr<inference_request>& irequest, const int32_t seq_slot,
      const inference_request::SequenceId& corr_id,
      const bool not_ready = false);

  // Update the implicit state and set the required input states.
  void UpdateImplicitState(
      std::unique_ptr<inference_request>& irequest, const int32_t seq_slot);

  // The controlling scheduler.
  SequenceBatchScheduler* const base_;

  // The index of this batcher within the controlling scheduler.
  const uint32_t batcher_idx_;

  // The number of candidate sequence slots.
  const size_t seq_slot_cnt_;

  // The input tensors that require shape checking before being
  // allowed in a batch. As a map from the tensor name to a bool. If
  // tensor is in map then its shape must match shape of same tensor
  // in requests already in the batch. If value is "true" then
  // additional tensor is treated as a shape tensor and the values
  // contained in the shape tensor must match same tensor already in
  // the batch.
  const std::unordered_map<std::string, bool> enforce_equal_shape_tensors_;

  // Store information on whether the model contains optional inputs.
  bool has_optional_input_;

  // The control values, delivered as input tensors, that should be
  // used when starting a sequence, continuing a sequence, ending a
  // sequence, and showing that a sequence has not input available.
  std::shared_ptr<SequenceBatchScheduler::ControlInputs> start_input_overrides_;
  std::shared_ptr<SequenceBatchScheduler::ControlInputs> end_input_overrides_;
  std::shared_ptr<SequenceBatchScheduler::ControlInputs>
      startend_input_overrides_;
  std::shared_ptr<SequenceBatchScheduler::ControlInputs>
      continue_input_overrides_;
  std::shared_ptr<SequenceBatchScheduler::ControlInputs>
      notready_input_overrides_;

  // The correlation ID override. Empty if model does not specify the
  // CONTROL_SEQUENCE_CORRID control.
  std::shared_ptr<inference_request::Input> seq_slot_corrid_override_;

  // For each sequence slot store the optional state i/o tensors.
  std::vector<std::shared_ptr<SequenceStates>> sequence_states_;
};

// Scheduler that implements the Direct sequence scheduling strategy
// for a model instance.
class DirectSequenceBatch : public SequenceBatch {
 public:
  DirectSequenceBatch(
      SequenceBatchScheduler* base, const uint32_t batcher_idx,
      const size_t seq_slot_cnt, TritonModelInstance* model_instance,
      const std::unordered_map<std::string, bool>& enforce_equal_shape_tensors,
      const bool has_optional_input,
      const std::shared_ptr<SequenceBatchScheduler::ControlInputs>&
          start_input_overrides,
      const std::shared_ptr<SequenceBatchScheduler::ControlInputs>&
          end_input_overrides,
      const std::shared_ptr<SequenceBatchScheduler::ControlInputs>&
          startend_input_overrides,
      const std::shared_ptr<SequenceBatchScheduler::ControlInputs>&
          continue_input_overrides,
      const std::shared_ptr<SequenceBatchScheduler::ControlInputs>&
          notready_input_overrides,
      bool* is_initialized);
  ~DirectSequenceBatch();

  void Enqueue(
      const uint32_t seq_slot,
      const inference_request::SequenceId& correlation_id,
      std::unique_ptr<inference_request>& request) override;

 private:
  void BatcherThread(const int nice);
  void NewPayload();

  std::shared_ptr<Payload> curr_payload_;
  TritonModelInstance* model_instance_;

  // The thread scheduling requests that are queued in this batch.
  std::unique_ptr<std::thread> scheduler_thread_;
  bool scheduler_thread_exit_;
  bool scheduler_idle_;

  // Mutex protecting correlation queues, etc.
  std::mutex mu_;
  std::condition_variable cv_;

  // Execution state of the last enqueued payload
  bool exec_complete_;

  // Mutex protecting execution state of payload
  std::mutex payload_mu_;
  std::condition_variable payload_cv_;

  // Queues holding inference requests. There are 'seq_slot_cnt'
  // queues, one for each sequence slot where requests assigned to
  // that slot are enqueued to wait for inferencing.
  std::vector<std::deque<std::unique_ptr<inference_request>>> queues_;

  // Is each sequence slot active or not? A zero or empty value indicates
  // inactive, a non-zero/non-empty value indicates active and is the
  // correlation ID of the sequence active in the slot. An empty
  // queue for a sequence slot does not mean it's inactive... it
  // could just not have any requests pending at the moment.
  std::vector<inference_request::SequenceId> seq_slot_correlation_ids_;

  // The maximum active sequence slot. A value of -1 indicates that
  // no slots are active in the model.
  int32_t max_active_seq_slot_;

  size_t max_batch_size_;
  float minimum_slot_utilization_;
  uint64_t pending_batch_delay_ns_;
};

// Scheduler that implements the oldest-first sequence scheduling
// strategy for a model instance.
class OldestSequenceBatch : public SequenceBatch {
 public:
  OldestSequenceBatch(
      SequenceBatchScheduler* base, const uint32_t batcher_idx,
      const size_t seq_slot_cnt, TritonModelInstance* model_instance,
      const std::unordered_map<std::string, bool>& enforce_equal_shape_tensors,
      const bool has_optional_input,
      const std::shared_ptr<SequenceBatchScheduler::ControlInputs>&
          start_input_overrides,
      const std::shared_ptr<SequenceBatchScheduler::ControlInputs>&
          end_input_overrides,
      const std::shared_ptr<SequenceBatchScheduler::ControlInputs>&
          startend_input_overrides,
      const std::shared_ptr<SequenceBatchScheduler::ControlInputs>&
          continue_input_overrides,
      const std::shared_ptr<SequenceBatchScheduler::ControlInputs>&
          notready_input_overrides,
      bool* is_initialized);
  ~OldestSequenceBatch();

  void Enqueue(
      const uint32_t seq_slot,
      const inference_request::SequenceId& correlation_id,
      std::unique_ptr<inference_request>& request) override;

 private:
  void CompleteAndNext(const uint32_t seq_slot);

  // The dynamic batcher for this scheduler
  std::unique_ptr<Scheduler> dynamic_batcher_;

  TritonModelInstance* model_instance_;

  // Mutex protecting queues, etc.
  std::mutex mu_;

  // For each sequence slot, true if there is a request for that
  // sequence in-flight in the dynamic batcher. Used to ensure that at
  // most one request from each sequence can be scheduled at a time.
  std::vector<bool> in_flight_;

  // Queues holding inference requests. There are 'seq_slot_cnt'
  // queues, one for each sequence slot where requests assigned to
  // that slot are enqueued to wait for inferencing.
  std::vector<std::deque<std::unique_ptr<inference_request>>> queues_;
};

}  // namespace hercules::core
