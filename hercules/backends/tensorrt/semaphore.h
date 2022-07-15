
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/


#include <condition_variable>
#include <mutex>

namespace hercules::backend { namespace tensorrt {

class Semaphore {
 public:
  explicit Semaphore(const int count) : count_(count) {}

  void Release()
  {
    std::unique_lock<std::mutex> lck(mtx_);
    count_++;
    cv_.notify_one();
  }

  void Acquire()
  {
    std::unique_lock<std::mutex> lck(mtx_);
    cv_.wait(lck, [this]() { return (count_ > 0); });
    count_--;
  }

 private:
  int count_;

  std::mutex mtx_;
  std::condition_variable cv_;
};

}}}  // namespace hercules::backend::tensorrt
