
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/

#pragma once

namespace hercules::core {

    // Currently there is just a global memory manager that is used for
    // all backends and which simply forwards requests on to the core
    // memory manager.
    struct hercules_memory_manager {
    };

}  // namespace hercules::core
