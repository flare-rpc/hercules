
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

// SharedLibrary
//
// Utility functions for shared libraries. Because some operations
// require serialization, this object cannot be directly constructed
// and must instead be accessed using Acquire().
class SharedLibrary {
 public:
  // Acquire a SharedLibrary object exclusively. Any other attempts to
  // concurrently acquire a SharedLibrary object will block.
  // object. Ownership is released by destroying the SharedLibrary
  // object.
  static Status Acquire(std::unique_ptr<SharedLibrary>* slib);

  ~SharedLibrary();

  // Configuration so that dependent libraries will be searched for in
  // 'path' during OpenLibraryHandle.
  Status SetLibraryDirectory(const std::string& path);

  // Reset any configuration done by SetLibraryDirectory.
  Status ResetLibraryDirectory();

  // Open shared library and return generic handle.
  Status OpenLibraryHandle(const std::string& path, void** handle);

  // Close shared library.
  Status CloseLibraryHandle(void* handle);

  // Get a generic pointer for an entrypoint into a shared library.
  Status GetEntrypoint(
      void* handle, const std::string& name, const bool optional, void** befn);

 private:
  FLARE_DISALLOW_COPY_AND_ASSIGN(SharedLibrary);
  explicit SharedLibrary() = default;
};

}  // namespace hercules::core
