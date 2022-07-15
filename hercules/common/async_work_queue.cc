
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/


#include "hercules/common/async_work_queue.h"

namespace hercules::common {

AsyncWorkQueue::~AsyncWorkQueue()
{
  GetSingleton()->thread_pool_.reset();
}

AsyncWorkQueue*
AsyncWorkQueue::GetSingleton()
{
  static AsyncWorkQueue singleton;
  return &singleton;
}

Error
AsyncWorkQueue::Initialize(size_t worker_count)
{
  if (worker_count < 1) {
    return Error(
        Error::Code::INVALID_ARG,
        "Async work queue must be initialized with positive 'worker_count'");
  }

  static std::mutex init_mtx;
  std::lock_guard<std::mutex> lk(init_mtx);

  if (GetSingleton()->thread_pool_) {
    return Error(
        Error::Code::ALREADY_EXISTS,
        "Async work queue has been initialized with " +
            std::to_string(GetSingleton()->thread_pool_->Size()) +
            " 'worker_count'");
  }

  GetSingleton()->thread_pool_.reset(new ThreadPool(worker_count));
  return Error::Success;
}

size_t
AsyncWorkQueue::WorkerCount()
{
  if (!GetSingleton()->thread_pool_) {
    return 0;
  }
  return GetSingleton()->thread_pool_->Size();
}

Error
AsyncWorkQueue::AddTask(std::function<void(void)>&& task)
{
  if (!GetSingleton()->thread_pool_) {
    return Error(
        Error::Code::UNAVAILABLE,
        "Async work queue must be initialized before adding task");
  }
  GetSingleton()->thread_pool_->Enqueue(std::move(task));

  return Error::Success;
}

void
AsyncWorkQueue::Reset()
{
  // Reconstruct the singleton to reset it
  GetSingleton()->~AsyncWorkQueue();
  new (GetSingleton()) AsyncWorkQueue();
}

}  // namespace hercules::common
