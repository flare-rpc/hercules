
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/

#pragma once

#include "infer_stats.h"
#include "label_provider.h"
#include "hercules/proto/model_config.pb.h"
#include "scheduler.h"
#include "status.h"

namespace hercules::core {

    class inference_request;

    //
    // Interface for models that handle inference requests.
    //
    class Model {
    public:
        explicit Model(
                const double min_compute_capability, const std::string &model_dir,
                const int64_t version, const hercules::proto::ModelConfig &config)
                : config_(config), min_compute_capability_(min_compute_capability),
                  version_(version), required_input_count_(0), model_dir_(model_dir) {
        }

        virtual ~Model() {}

        // Get the name of model being served.
        const std::string &Name() const { return config_.name(); }

        // Get the version of model being served.
        int64_t Version() const { return version_; }

        // Get the configuration of model being served.
        const hercules::proto::ModelConfig &Config() const { return config_; }

        // Get the number of required inputs
        size_t RequiredInputCount() const { return required_input_count_; }

        // Get the stats collector for the model being served.
        inference_stats_aggregator *MutableStatsAggregator() {
            return &stats_aggregator_;
        }

        const inference_stats_aggregator &StatsAggregator() const {
            return stats_aggregator_;
        }

        // Get the model configuration for a named input.
        Status GetInput(
                const std::string &name, const hercules::proto::ModelInput **input) const;

        // Get the model configuration for a named output.
        Status GetOutput(
                const std::string &name, const hercules::proto::ModelOutput **output) const;

        // Get a label provider for the model.
        const std::shared_ptr<LabelProvider> &GetLabelProvider() const {
            return label_provider_;
        }

        // Initialize the instance for Triton core usage
        Status Init();

        // Enqueue a request for execution. If Status::Success is returned
        // then the model has taken ownership of the request object and so
        // 'request' will be nullptr. If non-success is returned then the
        // caller still retains ownership of 'request'.
        Status Enqueue(std::unique_ptr<inference_request> &request) {
            return scheduler_->Enqueue(request);
        }

        // Return the number of in-flight inferences.
        size_t InflightInferenceCount() {
            return scheduler_->InflightInferenceCount();
        }

        // Stop processing future requests unless they are considered as in-flight.
        void Stop() { scheduler_->Stop(); }

        uint32_t DefaultPriorityLevel() const { return default_priority_level_; }

        uint32_t MaxPriorityLevel() const { return max_priority_level_; }

    protected:
        // Set the configuration of the model being served.
        Status SetModelConfig(const hercules::proto::ModelConfig &config);

        // Explicitly set the scheduler to use for inference requests to the
        // model. The scheduler can only be set once for a model.
        Status SetScheduler(std::unique_ptr<Scheduler> scheduler);

        // The scheduler to use for this model.
        std::unique_ptr<Scheduler> scheduler_;

        // Configuration of the model.
        hercules::proto::ModelConfig config_;

    private:
        // The minimum supported CUDA compute capability.
        const double min_compute_capability_;

        // Version of the model.
        int64_t version_;

        // The stats collector for the model.
        inference_stats_aggregator stats_aggregator_;

        // Label provider for this model.
        std::shared_ptr<LabelProvider> label_provider_;

        size_t required_input_count_;

        // Map from input name to the model configuration for that input.
        std::unordered_map<std::string, hercules::proto::ModelInput> input_map_;

        // Map from output name to the model configuration for that output.
        std::unordered_map<std::string, hercules::proto::ModelOutput> output_map_;

        // Path to model
        std::string model_dir_;

        // The default priority level for the model.
        uint32_t default_priority_level_;

        // The largest priority value for the model.
        uint32_t max_priority_level_;
    };

}  // namespace hercules::core
