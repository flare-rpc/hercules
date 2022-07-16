
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/


#include "shared_library.h"
#include "filesystem.h"
#include "mutex"
#include "hercules/common/logging.h"
#include <dlfcn.h>

namespace hercules::core {

    static std::mutex mu_;

    Status
    shared_library::acquire(std::unique_ptr<shared_library> *slib) {
        mu_.lock();
        slib->reset(new shared_library());
        return Status::Success;
    }

    shared_library::~shared_library() {
        mu_.unlock();
    }

    Status
    shared_library::set_library_directory(const std::string &path) {
        return Status::Success;
    }

    Status
    shared_library::reset_library_directory() {

        return Status::Success;
    }

    Status
    shared_library::open_library_handle(const std::string &path, void **handle) {
        FLARE_LOG(DEBUG) << "open_library_handle: " << path;

        *handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (*handle == nullptr) {
            return Status(
                    Status::Code::NOT_FOUND,
                    "unable to load shared library: " + std::string(dlerror()));
        }

        return Status::Success;
    }

    Status
    shared_library::close_library_handle(void *handle) {
        if (handle != nullptr) {
            if (dlclose(handle) != 0) {
                return Status(
                        Status::Code::INTERNAL,
                        "unable to unload shared library: " + std::string(dlerror()));
            }
        }

        return Status::Success;
    }

    Status
    shared_library::get_entrypoint(
            void *handle, const std::string &name, const bool optional, void **befn) {
        *befn = nullptr;
        dlerror();
        void *fn = dlsym(handle, name.c_str());
        const char *dlsym_error = dlerror();
        if (dlsym_error != nullptr) {
            if (optional) {
                return Status::Success;
            }

            std::string errstr(dlsym_error);  // need copy as dlclose overwrites
            return Status(
                    Status::Code::NOT_FOUND, "unable to find required entrypoint '" + name +
                                             "' in shared library: " + errstr);
        }

        if (fn == nullptr) {
            if (optional) {
                return Status::Success;
            }

            return Status(
                    Status::Code::NOT_FOUND,
                    "unable to find required entrypoint '" + name + "' in shared library");
        }

        *befn = fn;
        return Status::Success;
    }

}  // namespace hercules::core
