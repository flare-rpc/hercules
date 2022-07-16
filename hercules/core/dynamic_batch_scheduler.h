
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
#include <map>
#include <mutex>
#include <queue>
#include <set>
#include <thread>
#include "backend_model.h"
#include "backend_model_instance.h"
#include "hercules/proto/model_config.pb.h"
#include "rate_limiter.h"
#include "scheduler.h"
#include "scheduler_utils.h"
#include "status.h"
#include "hercules/common/model_config.h"

namespace hercules::core {

    // Scheduler that implements dynamic batching.
    class dynamic_batch_scheduler : public Scheduler {
    public:
        // Create a scheduler to support a given number of runners and a run
        // function to call when a request is scheduled.
        static Status Create(
                TritonModel *model, TritonModelInstance *model_instance, const int nice,
                const bool dynamic_batching_enabled, const int32_t max_batch_size,
                const std::unordered_map<std::string, bool> &enforce_equal_shape_tensors,
                const bool preserve_ordering, const bool response_cache_enable,
                const std::set<int32_t> &preferred_batch_sizes,
                const uint64_t max_queue_delay_microseconds,
                std::unique_ptr<Scheduler> *scheduler);

        // Create a scheduler to support a given number of runners and a run
        // function to call when a request is scheduled. And the scheduler also
        // supports different queue policies for different priority levels.
        static Status Create(
                TritonModel *model, TritonModelInstance *model_instance, const int nice,
                const bool dynamic_batching_enabled, const int32_t max_batch_size,
                const std::unordered_map<std::string, bool> &enforce_equal_shape_tensors,
                const hercules::proto::ModelDynamicBatching &batcher_config,
                const bool response_cache_enable, std::unique_ptr<Scheduler> *scheduler);

        ~dynamic_batch_scheduler();

        // \see Scheduler::Enqueue()
        Status Enqueue(std::unique_ptr<inference_request> &request) override;

        // \see Scheduler::InflightInferenceCount()
        size_t InflightInferenceCount() override {
            std::unique_lock<std::mutex> lock(mu_);
            if (curr_payload_ != nullptr) {
                return queue_.Size() + curr_payload_->RequestCount();
            }
            return queue_.Size();
        }

        // \see Scheduler::Stop()
        void Stop() override { stop_ = true; }

        metric_model_reporter *MetricReporter() const { return reporter_.get(); }

    private:
        dynamic_batch_scheduler(
                TritonModel *model, TritonModelInstance *model_instance,
                const bool dynamic_batching_enabled, const int32_t max_batch_size,
                const std::unordered_map<std::string, bool> &enforce_equal_shape_tensors,
                const bool preserve_ordering, const bool response_cache_enable,
                const std::set<int32_t> &preferred_batch_sizes,
                const uint64_t max_queue_delay_microseconds,
                const hercules::proto::ModelQueuePolicy &default_queue_policy,
                const uint32_t priority_levels,
                const ModelQueuePolicyMap &queue_policy_map);

        void BatcherThread(const int nice);

        void NewPayload();

        uint64_t GetDynamicBatch();

        void DelegateResponse(std::unique_ptr<inference_request> &request);

        void CacheLookUp(
                std::unique_ptr<inference_request> &request,
                std::unique_ptr<InferenceResponse> &cached_response);

        void FinalizeResponses();

        TritonModel *model_;
        TritonModelInstance *model_instance_;

        // True if dynamic batching is enabled.
        const bool dynamic_batching_enabled_;

        // Map from priority level to queue holding inference requests for the model
        // represented by this scheduler. If priority queues are not supported by the
        // scheduler, then priority zero entry is used as the single queue.
        PriorityQueue queue_;
        bool stop_;

        std::thread scheduler_thread_;
        std::atomic<bool> scheduler_thread_exit_;

        // Mutex and condvar for signaling scheduler thread
        std::mutex mu_;
        std::condition_variable cv_;

        std::shared_ptr<RateLimiter> rate_limiter_;

        std::shared_ptr<Payload> curr_payload_;
        bool payload_saturated_;

        size_t max_batch_size_;
        size_t max_preferred_batch_size_;
        std::set<int32_t> preferred_batch_sizes_;
        uint64_t pending_batch_delay_ns_;
        size_t pending_batch_size_;
        RequiredEqualInputs required_equal_inputs_;

        size_t queued_batch_size_;
        size_t next_preferred_batch_size_;

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

        // If true the ordering of responses matches the order of requests
        // even when there are multiple scheduler threads.
        const bool preserve_ordering_;

        // If true, the scheduler will try to retrieve responses from cache.
        bool response_cache_enabled_;

        // Per completion-id queues to store the ready responses
        std::deque<
                std::vector<std::pair<std::unique_ptr<InferenceResponse>, uint32_t>>>
                completion_queue_;
        // Lock to protect the completion_queues_
        std::mutex completion_queue_mtx_;

        // Reporter for metrics, or nullptr if no metrics should be reported
        std::shared_ptr<metric_model_reporter> reporter_;
    };

}  // namespace hercules::core
