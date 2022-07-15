
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/

#pragma once

#include "error.h"
#include "thread_pool.h"

namespace triton { namespace common {
// Manager for asynchronous worker threads. Use to accelerate copies and
// other such operations by running them in parallel.
// Call Initialize to start the worker threads (once) and AddTask to tasks to
// the queue.

class AsyncWorkQueue {
 public:
  // Start 'worker_count' number of worker threads.
  static Error Initialize(size_t worker_count);

  // Get the number of worker threads.
  static size_t WorkerCount();

  // Add a 'task' to the queue. The function will take ownership of 'task'.
  // Therefore std::move should be used when calling AddTask.
  static Error AddTask(std::function<void(void)>&& task);

 protected:
  static void Reset();

 private:
  AsyncWorkQueue() = default;
  ~AsyncWorkQueue();
  static AsyncWorkQueue* GetSingleton();
  std::unique_ptr<ThreadPool> thread_pool_;
};

}}  // namespace triton::common
