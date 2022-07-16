
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/


#include <functional>
#include <map>
#include <memory>
#include "memory.h"
#include "status.h"
#include "hercules/common/model_config.h"

#pragma once

namespace hercules::core {

//
// Sequence state tensors.
//
class SequenceState {
 public:
  SequenceState();
  SequenceState(
      const std::string& name, const hercules::proto::DataType datatype,
      const std::vector<int64_t>& shape);
  SequenceState(
      const std::string& name, const hercules::proto::DataType datatype,
      const int64_t* shape, const uint64_t dim_count);

  // The name of the state tensor.
  const std::string& Name() const { return name_; }

  // Data type of the state tensor.
  hercules::proto::DataType DType() const { return datatype_; }

  // Mutable data type of the state tensor.
  hercules::proto::DataType* MutableDType() { return &datatype_; }

  // The shape of the state tensor after normalization.
  const std::vector<int64_t>& Shape() const { return shape_; }
  std::vector<int64_t>* MutableShape() { return &shape_; }

  // The data for this shape.
  std::shared_ptr<memory_base>& Data() { return data_; }

  // Set the data for this shape. Error if state already has some
  // data.
  Status SetData(const std::shared_ptr<memory_base>& data);

  // Sets state tensors that have type string to zero
  Status SetStringDataToZero();

  // Remove all existing data for the state.
  Status RemoveAllData();

  // Set the state update callback.
  void SetStateUpdateCallback(std::function<Status()>&& state_update_cb)
  {
    state_update_cb_ = std::move(state_update_cb);
  }

  // Call the state update callback. This function will be called when
  // TRITONBACKEND_StateUpdate is called.
  Status Update() { return state_update_cb_(); }

 private:
  FLARE_DISALLOW_COPY_AND_ASSIGN(SequenceState);
  std::string name_;
  hercules::proto::DataType datatype_;
  std::vector<int64_t> shape_;
  std::vector<int64_t> batch_dim_;
  std::shared_ptr<memory_base> data_;
  std::function<Status()> state_update_cb_ = []() {
    // By default calling the TRITONBACKEND_StateUpdate will return an error.
    return Status(
        Status::Code::INVALID_ARG,
        "TRITONBACKEND_StateUpdate called when sequence batching is disabled "
        "or the 'states' section of the model configuration is empty.");
  };
};

class SequenceStates {
 public:
  struct InitialStateData {
    InitialStateData(const std::string& state_init_name)
        : state_init_name_(state_init_name)
    {
    }

    std::string state_init_name_;
    std::shared_ptr<mutable_memory> data_;
  };

  // Initialize the state tensors according to the state model configuration.
  // Will use a default value of 1 for the variable dimensions in the state
  // tensor configuration.
  Status Initialize(
      const std::unordered_map<
          std::string, const hercules::proto::ModelSequenceBatching_State&>&
          state_output_config_map,
      const size_t max_batch_size,
      const std::unordered_map<std::string, InitialStateData>& initial_state);

  // Get a buffer holding the output state.
  Status OutputState(
      const std::string& name, const hercules::proto::DataType datatype,
      const int64_t* shape, const uint64_t dim_count,
      SequenceState** output_state);
  Status OutputState(
      const std::string& name, const hercules::proto::DataType datatype,
      const std::vector<int64_t>& shape, SequenceState** output_state);

  // Create a copy of the 'from' sequence states for NULL requests.
  static std::shared_ptr<SequenceStates> CopyAsNull(
      const std::shared_ptr<SequenceStates>& from);

  const std::map<std::string, std::unique_ptr<SequenceState>>& InputStates()
  {
    return input_states_;
  }

  std::map<std::string, std::unique_ptr<SequenceState>>& OutputStates()
  {
    return output_states_;
  }

  void SetNullSequenceStates(std::shared_ptr<SequenceStates> sequence_states)
  {
    null_sequence_states_ = sequence_states;
    is_null_request_ = true;
  }

  const std::shared_ptr<SequenceStates>& NullSequenceStates()
  {
    return null_sequence_states_;
  }

  bool IsNullRequest() { return is_null_request_; }

 private:
  std::map<std::string, std::unique_ptr<SequenceState>> input_states_;
  std::map<std::string, std::unique_ptr<SequenceState>> output_states_;
  std::shared_ptr<SequenceStates> null_sequence_states_;
  bool is_null_request_ = false;
};

}  // namespace hercules::core
