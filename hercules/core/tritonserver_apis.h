
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/

#pragma once

#define _COMPILING_TRITONSERVER 1
#define _COMPILING_TRITONBACKEND 1
#define _COMPILING_TRITONREPOAGENT 1

#include "hercules/core/tritonbackend.h"
#include "hercules/core/tritonrepoagent.h"
#include "hercules/core/tritonserver.h"

#undef _COMPILING_TRITONSERVER
#undef _COMPILING_TRITONBACKEND
#undef _COMPILING_TRITONREPOAGENT
