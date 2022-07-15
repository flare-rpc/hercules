
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/


#include "backend_memory_manager.h"

#include "pinned_memory_manager.h"
#include "status.h"
#include "tritonserver_apis.h"

#ifdef HERCULES_ENABLE_GPU
#include <cuda_runtime_api.h>
#include "cuda_memory_manager.h"
#endif  // HERCULES_ENABLE_GPU

// For unknown reason, windows will not export the TRITONBACKEND_*
// functions declared with dllexport in tritonbackend.h. To get those
// functions exported it is (also?) necessary to mark the definitions
// in this file with dllexport as well.
#if defined(_MSC_VER)
#define TRITONAPI_DECLSPEC __declspec(dllexport)
#elif defined(__GNUC__)
#define TRITONAPI_DECLSPEC __attribute__((__visibility__("default")))
#else
#define TRITONAPI_DECLSPEC
#endif

namespace hercules::core {

    extern "C" {

    TRITONAPI_DECLSPEC TRITONSERVER_Error *
    TRITONBACKEND_MemoryManagerAllocate(
            TRITONBACKEND_MemoryManager *manager, void **buffer,
            const TRITONSERVER_MemoryType memory_type, const int64_t memory_type_id,
            const uint64_t byte_size) {
        switch (memory_type) {
            case TRITONSERVER_MEMORY_GPU:
#ifdef HERCULES_ENABLE_GPU
                {
                  auto status = cuda_memory_manager::Alloc(buffer, byte_size, memory_type_id);
                  if (!status.IsOk()) {
                    return TRITONSERVER_ErrorNew(
                        TRITONSERVER_ERROR_UNAVAILABLE, status.Message().c_str());
                  }
                  break;
                }
#else
                return TRITONSERVER_ErrorNew(
                        TRITONSERVER_ERROR_UNSUPPORTED,
                        "GPU memory allocation not supported");
#endif  // HERCULES_ENABLE_GPU

            case TRITONSERVER_MEMORY_CPU_PINNED:
#ifdef HERCULES_ENABLE_GPU
                {
                  TRITONSERVER_MemoryType mt = memory_type;
                  auto status = PinnedMemoryManager::Alloc(buffer, byte_size, &mt, false);
                  if (!status.IsOk()) {
                    return TRITONSERVER_ErrorNew(
                        TRITONSERVER_ERROR_UNAVAILABLE, status.Message().c_str());
                  }
                  break;
                }
#else
                return TRITONSERVER_ErrorNew(
                        TRITONSERVER_ERROR_UNSUPPORTED,
                        "Pinned memory allocation not supported");
#endif  // HERCULES_ENABLE_GPU

            case TRITONSERVER_MEMORY_CPU: {
                *buffer = malloc(byte_size);
                if (*buffer == nullptr) {
                    return TRITONSERVER_ErrorNew(
                            TRITONSERVER_ERROR_UNAVAILABLE, "CPU memory allocation failed");
                }
                break;
            }
        }

        return nullptr;  // success
    }

    TRITONAPI_DECLSPEC TRITONSERVER_Error *
    TRITONBACKEND_MemoryManagerFree(
            TRITONBACKEND_MemoryManager *manager, void *buffer,
            const TRITONSERVER_MemoryType memory_type, const int64_t memory_type_id) {
        switch (memory_type) {
            case TRITONSERVER_MEMORY_GPU: {
#ifdef HERCULES_ENABLE_GPU
                auto status = cuda_memory_manager::Free(buffer, memory_type_id);
                if (!status.IsOk()) {
                  return TRITONSERVER_ErrorNew(
                      StatusCodeToTritonCode(status.StatusCode()),
                      status.Message().c_str());
                }
#endif  // HERCULES_ENABLE_GPU
                break;
            }

            case TRITONSERVER_MEMORY_CPU_PINNED: {
#ifdef HERCULES_ENABLE_GPU
                auto status = PinnedMemoryManager::Free(buffer);
                if (!status.IsOk()) {
                  return TRITONSERVER_ErrorNew(
                      StatusCodeToTritonCode(status.StatusCode()),
                      status.Message().c_str());
                }
#endif  // HERCULES_ENABLE_GPU
                break;
            }

            case TRITONSERVER_MEMORY_CPU:
                free(buffer);
                break;
        }

        return nullptr;  // success
    }

    }  // extern C

}  // namespace hercules::core
