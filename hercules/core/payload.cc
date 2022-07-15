
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/


#include "payload.h"

namespace hercules::core {

Payload::Payload()
    : op_type_(Operation::INFER_RUN),
      requests_(std::vector<std::unique_ptr<inference_request>>()),
      OnCallback_([]() {}), instance_(nullptr), state_(State::UNINITIALIZED),
      batcher_start_ns_(0), saturated_(false)
{
  exec_mu_.reset(new std::mutex());
}

Status
Payload::MergePayload(std::shared_ptr<Payload>& payload)
{
  if ((payload->GetOpType() != Operation::INFER_RUN) ||
      (op_type_ != Operation::INFER_RUN)) {
    return Status(
        Status::Code::INTERNAL,
        "Attempted to merge payloads of type that are not INFER_RUN");
  }
  if (payload->GetInstance() != instance_) {
    return Status(
        Status::Code::INTERNAL,
        "Attempted to merge payloads of mismatching instance");
  }
  if ((payload->GetState() != State::EXECUTING) ||
      (state_ != State::EXECUTING)) {
    return Status(
        Status::Code::INTERNAL,
        "Attempted to merge payloads that are not in executing state");
  }

  requests_.insert(
      requests_.end(), std::make_move_iterator(payload->Requests().begin()),
      std::make_move_iterator(payload->Requests().end()));

  payload->Callback();

  return Status::Success;
}

void
Payload::Reset(const Operation op_type, TritonModelInstance* instance)
{
  op_type_ = op_type;
  requests_.clear();
  OnCallback_ = []() {};
  release_callbacks_.clear();
  instance_ = instance;
  state_ = State::UNINITIALIZED;
  status_.reset(new std::promise<Status>());
  batcher_start_ns_ = 0;
  saturated_ = false;
}

void
Payload::Release()
{
  op_type_ = Operation::INFER_RUN;
  requests_.clear();
  OnCallback_ = []() {};
  release_callbacks_.clear();
  instance_ = nullptr;
  state_ = State::RELEASED;
  batcher_start_ns_ = 0;
  saturated_ = false;
}

size_t
Payload::BatchSize()
{
  size_t batch_size = 0;
  for (const auto& request : requests_) {
    batch_size += std::max(1U, request->BatchSize());
  }
  return batch_size;
}

void
Payload::ReserveRequests(size_t size)
{
  requests_.reserve(size);
}

void
Payload::AddRequest(std::unique_ptr<inference_request> request)
{
  if ((batcher_start_ns_ == 0) ||
      (batcher_start_ns_ > request->BatcherStartNs())) {
    batcher_start_ns_ = request->BatcherStartNs();
  }
  requests_.push_back(std::move(request));
}

void
Payload::SetCallback(std::function<void()> OnCallback)
{
  OnCallback_ = OnCallback;
}

void
Payload::SetInstance(TritonModelInstance* model_instance)
{
  instance_ = model_instance;
}

void
Payload::AddInternalReleaseCallback(std::function<void()>&& callback)
{
  release_callbacks_.emplace_back(std::move(callback));
}

void
Payload::MarkSaturated()
{
  saturated_ = true;
}

void
Payload::SetState(Payload::State state)
{
  state_ = state;
}

Status
Payload::Wait()
{
  return status_->get_future().get();
}

void
Payload::Callback()
{
  OnCallback_();
}

void
Payload::OnRelease()
{
  // Invoke the release callbacks added internally before releasing the
  // request to user provided callback.
  for (auto it = release_callbacks_.rbegin(); it != release_callbacks_.rend();
       it++) {
    (*it)();
  }
  release_callbacks_.clear();
}

void
Payload::Execute(bool* should_exit)
{
  *should_exit = false;

  Status status;
  switch (op_type_) {
    case Operation::INFER_RUN:
      instance_->Schedule(std::move(requests_), OnCallback_);
      break;
    case Operation::INIT:
      status = instance_->Initialize();
      break;
    case Operation::WARM_UP:
      status = instance_->WarmUp();
      break;
    case Operation::EXIT:
      *should_exit = true;
  }

  status_->set_value(status);
}

}  // namespace hercules::core
