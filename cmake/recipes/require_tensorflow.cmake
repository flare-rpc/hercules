
find_path(TF_INCLUDE_PATH NAMES tensorflow/core/platform/env.h)
find_library(TF_CC_LIB NAMES tensorflow_cc)
find_library(TF_FR_LIB NAMES tensorflow_framework)

if ((NOT TF_INCLUDE_PATH) OR (NOT TF_CC_LIB) OR (NOT TF_FR_LIB))
    message(FATAL_ERROR "Fail to find tf")
endif()
include_directories(${TF_INCLUDE_PATH})

