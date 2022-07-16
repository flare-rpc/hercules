
FILE(GLOB_RECURSE COMMON_SRC "common/*.cc")
FILE(GLOB_RECURSE CORE_SRC "core/*.cc")
FILE(GLOB_RECURSE BACKEND_SRC "backend/*.cc")
FILE(GLOB_RECURSE BACKEND_TF_SRC "backends/tf/*.cc")

add_definitions(
        -DTRITON_ENABLE_ENSEMBLE
        -DTRITON_ENABLE_ENSEMBLE
)

SET(HERCULES_SRC
        ${CORE_SRC}
        ${COMMON_SRC}
        ${BACKEND_SRC}
        ${BACKEND_TF_SRC}
        )

#
#if (APPLE)
#    find_library(CoreFoundation CoreFoundation)
#    carbin_cc_library(
#            NAME
#            abel
#            SRCS
#            ${ABEL_SRC}
#            COPTS
#            ${CARBIN_DEFAULT_COPTS}
#            DEPS
#                ${CoreFoundation}
#                pthread
#            PUBLIC
#    )
#elseif (${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
#    carbin_cc_library(
#            NAME
#            abel
#            SRCS
#            ${ABEL_SRC}
#            COPTS
#            ${CARBIN_DEFAULT_COPTS}
#            DEPS
#            rt
#            pthread
#            PUBLIC
#    )
#endif ()
#

# carbin_cc_library(  NAME myLibrary
#                  NAMESPACE myNamespace
#                  SOURCES
#                       myLib.cpp
#                       myLib_functions.cpp
#                  HEADERS
#                        mylib.h
#                  PUBLIC_DEFINITIONS
#                     USE_DOUBLE_PRECISION=1
#                  PRIVATE_DEFINITIONS
#                     DEBUG_VERBOSE
#                  PUBLIC_INCLUDE_PATHS
#                     ${CMAKE_SOURCE_DIR}/mylib/include
#                  PRIVATE_INCLUDE_PATHS
#                     ${CMAKE_SOURCE_DIR}/include
#                  PRIVATE_LINKED_TARGETS
#                     Threads::Threads
#                  PUBLIC_LINKED_TARGETS
#                     Threads::Threads
#                  LINKED_TARGETS
#                     Threads::Threads
#                  PUBLIC
#                  SHARED

link_directories($ENV{CONDA_PREFIX}/lib)
carbin_cc_library(
        NAMESPACE hercules
        NAME core
        SOURCES ${HERCULES_SRC}
        PUBLIC_LINKED_TARGETS ${ABEL_DYLINK} ${DYNAMIC_LIB} hercules::proto tensorflow_cc tensorflow_framework
        PRIVATE_COMPILE_OPTIONS ${CARBIN_DEFAULT_COPTS}
        PUBLIC
        SHARED
)
