#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "hercules::proto" for configuration "Debug"
set_property(TARGET hercules::proto APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(hercules::proto PROPERTIES
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/libproto.dylib"
  IMPORTED_SONAME_DEBUG "@rpath/libproto.dylib"
  )

list(APPEND _IMPORT_CHECK_TARGETS hercules::proto )
list(APPEND _IMPORT_CHECK_FILES_FOR_hercules::proto "${_IMPORT_PREFIX}/lib/libproto.dylib" )

# Import target "hercules::core" for configuration "Debug"
set_property(TARGET hercules::core APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(hercules::core PROPERTIES
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/libcore.dylib"
  IMPORTED_SONAME_DEBUG "@rpath/libcore.dylib"
  )

list(APPEND _IMPORT_CHECK_TARGETS hercules::core )
list(APPEND _IMPORT_CHECK_FILES_FOR_hercules::core "${_IMPORT_PREFIX}/lib/libcore.dylib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
