
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/

#pragma once

#include <memory>
#include <string>
#include "constants.h"
#include "status.h"

namespace hercules::core {

    // shared_library
    //
    // Utility functions for shared libraries. Because some operations
    // require serialization, this object cannot be directly constructed
    // and must instead be accessed using acquire().
    class shared_library {
    public:
        // acquire a shared_library object exclusively. Any other attempts to
        // concurrently acquire a shared_library object will block.
        // object. Ownership is released by destroying the shared_library
        // object.
        static Status acquire(std::unique_ptr<shared_library> *slib);

        ~shared_library();

        // Configuration so that dependent libraries will be searched for in
        // 'path' during open_library_handle.
        Status set_library_directory(const std::string &path);

        // Reset any configuration done by set_library_directory.
        Status reset_library_directory();

        // Open shared library and return generic handle.
        Status open_library_handle(const std::string &path, void **handle);

        // Close shared library.
        Status close_library_handle(void *handle);

        // Get a generic pointer for an entrypoint into a shared library.
        Status get_entrypoint(
                void *handle, const std::string &name, const bool optional, void **befn);

    private:
        FLARE_DISALLOW_COPY_AND_ASSIGN(shared_library);

        explicit shared_library() = default;
    };

}  // namespace hercules::core
