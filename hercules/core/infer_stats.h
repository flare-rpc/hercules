
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/

#pragma once

#include <time.h>
#include <map>
#include <memory>
#include <mutex>
#include "constants.h"
#include "status.h"
#include "tritonserver_apis.h"

namespace hercules::core {

    class metric_model_reporter;


    //
    // inference_stats_aggregator
    //
    // A statistics aggregator.
    //
    class inference_stats_aggregator {
#ifdef HERCULES_ENABLE_STATS
        public:
         struct InferStats {
           InferStats()
               : failure_count_(0), failure_duration_ns_(0), success_count_(0),
                 request_duration_ns_(0), queue_duration_ns_(0),
                 compute_input_duration_ns_(0), compute_infer_duration_ns_(0),
                 compute_output_duration_ns_(0), cache_hit_count_(0),
                 cache_hit_lookup_duration_ns_(0), cache_miss_count_(0),
                 cache_miss_lookup_duration_ns_(0),
                 cache_miss_insertion_duration_ns_(0)
           {
           }
           uint64_t failure_count_;
           uint64_t failure_duration_ns_;

           uint64_t success_count_;
           uint64_t request_duration_ns_;
           uint64_t queue_duration_ns_;
           uint64_t compute_input_duration_ns_;
           uint64_t compute_infer_duration_ns_;
           uint64_t compute_output_duration_ns_;

           // Cache hit stats
           uint64_t cache_hit_count_;
           uint64_t cache_hit_lookup_duration_ns_;
           // Cache miss stats
           uint64_t cache_miss_count_;
           uint64_t cache_miss_lookup_duration_ns_;
           uint64_t cache_miss_insertion_duration_ns_;
         };

         struct InferBatchStats {
           InferBatchStats()
               : count_(0), compute_input_duration_ns_(0),
                 compute_infer_duration_ns_(0), compute_output_duration_ns_(0)
           {
           }
           uint64_t count_;
           uint64_t compute_input_duration_ns_;
           uint64_t compute_infer_duration_ns_;
           uint64_t compute_output_duration_ns_;
         };

         // Create an aggregator for model statistics
         inference_stats_aggregator()
             : last_inference_ms_(0), inference_count_(0), execution_count_(0)
         {
         }

         uint64_t LastInferenceMs() const { return last_inference_ms_; }
         uint64_t InferenceCount() const { return inference_count_; }
         uint64_t ExecutionCount() const { return execution_count_; }
         const InferStats& ImmutableInferStats() const { return infer_stats_; }
         const std::map<size_t, InferBatchStats>& ImmutableInferBatchStats() const
         {
           return batch_stats_;
         }

         // Add durations to Infer stats for a failed inference request.
         void UpdateFailure(
             metric_model_reporter* metric_reporter, const uint64_t request_start_ns,
             const uint64_t request_end_ns);

         // Add durations to infer stats for a successful inference request.
         void UpdateSuccess(
             metric_model_reporter* metric_reporter, const size_t batch_size,
             const uint64_t request_start_ns, const uint64_t queue_start_ns,
             const uint64_t compute_start_ns, const uint64_t compute_input_end_ns,
             const uint64_t compute_output_start_ns, const uint64_t compute_end_ns,
             const uint64_t request_end_ns);

         // Add durations to infer stats for a successful inference request.
         void UpdateSuccessWithDuration(
             metric_model_reporter* metric_reporter, const size_t batch_size,
             const uint64_t request_start_ns, const uint64_t queue_start_ns,
             const uint64_t compute_start_ns, const uint64_t request_end_ns,
             const uint64_t compute_input_duration_ns,
             const uint64_t compute_infer_duration_ns,
             const uint64_t compute_output_duration_ns);

         // Add durations to infer stats for a successful cached response.
         void UpdateSuccessCacheHit(
             metric_model_reporter* metric_reporter, const size_t batch_size,
             const uint64_t request_start_ns, const uint64_t queue_start_ns,
             const uint64_t cache_lookup_start_ns, const uint64_t request_end_ns,
             const uint64_t cache_hit_lookup_duration_ns);

         // Add durations to infer stats for a cache miss and update request duration
         // to account for cache insertion after backend computes the response.
         void UpdateSuccessCacheMiss(
             metric_model_reporter* metric_reporter,
             const uint64_t cache_miss_lookup_duration_ns,
             const uint64_t cache_miss_insertion_duration_ns);

         // Add durations to batch infer stats for a batch execution.
         // 'success_request_count' is the number of sucess requests in the
         // batch that have infer_stats attached.
         void UpdateInferBatchStats(
             metric_model_reporter* metric_reporter, const size_t batch_size,
             const uint64_t compute_start_ns, const uint64_t compute_input_end_ns,
             const uint64_t compute_output_start_ns, const uint64_t compute_end_ns);

         // Add durations to batch infer stats for a batch execution.
         // 'success_request_count' is the number of sucess requests in the
         // batch that have infer_stats attached.
         void UpdateInferBatchStatsWithDuration(
             metric_model_reporter* metric_reporter, size_t batch_size,
             const uint64_t compute_input_duration_ns,
             const uint64_t compute_infer_duration_ns,
             const uint64_t compute_output_duration_ns);

        private:
         std::mutex mu_;
         uint64_t last_inference_ms_;
         uint64_t inference_count_;
         uint64_t execution_count_;
         InferStats infer_stats_;
         std::map<size_t, InferBatchStats> batch_stats_;
#endif  // HERCULES_ENABLE_STATS
    };


//
// Macros to set infer stats.
//
#ifdef HERCULES_ENABLE_STATS
#define INFER_STATS_SET_TIMESTAMP(TS_NS)                             \
  {                                                                  \
    TS_NS = std::chrono::duration_cast<std::chrono::nanoseconds>(    \
                std::chrono::steady_clock::now().time_since_epoch()) \
                .count();                                            \
  }
#define INFER_STATS_DECL_TIMESTAMP(TS_NS) \
  uint64_t TS_NS;                         \
  INFER_STATS_SET_TIMESTAMP(TS_NS);
#else
#define INFER_STATS_DECL_TIMESTAMP(TS_NS)
#define INFER_STATS_SET_TIMESTAMP(TS_NS)
#endif  // HERCULES_ENABLE_STATS

}  // namespace hercules::core
