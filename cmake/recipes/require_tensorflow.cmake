
find_path(TF_INCLUDE_PATH NAMES tensorflow/cc/framework/ops.h)
find_library(TF_LIB NAMES tensorflow_cc tensorflow_framework)
if ((NOT TF_INCLUDE_PATH) OR (NOT TF_LIB))
    message(FATAL_ERROR "Fail to find tensorflow")
endif()
include_directories(${TF_INCLUDE_PATH})

