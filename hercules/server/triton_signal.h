
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/

#pragma once

#include <condition_variable>
#include <mutex>
#include "triton/core/tritonserver.h"

namespace triton { namespace server {

// Exit mutex and cv used to signal the main thread that it should
// close the server and exit.
extern bool signal_exiting_;
extern std::mutex signal_exit_mu_;
extern std::condition_variable signal_exit_cv_;

// Register signal handler. Return true if success, false if failure.
TRITONSERVER_Error* RegisterSignalHandler();

}}  // namespace triton::server
