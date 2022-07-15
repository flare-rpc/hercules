
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/

#pragma once

#include <iostream>
#include <string>
#include "tritonserver_apis.h"

namespace hercules::core {

    //
    // An inference parameter.
    //
    class inference_parameter {
    public:
        inference_parameter(const char *name, const char *value)
                : name_(name), type_(TRITONSERVER_PARAMETER_STRING), value_string_(value) {
            byte_size_ = value_string_.size();
        }

        inference_parameter(const char *name, const int64_t value)
                : name_(name), type_(TRITONSERVER_PARAMETER_INT), value_int64_(value),
                  byte_size_(sizeof(int64_t)) {
        }

        inference_parameter(const char *name, const bool value)
                : name_(name), type_(TRITONSERVER_PARAMETER_BOOL), value_bool_(value),
                  byte_size_(sizeof(bool)) {
        }

        inference_parameter(const char *name, const void *ptr, const uint64_t size)
                : name_(name), type_(TRITONSERVER_PARAMETER_BYTES), value_bytes_(ptr),
                  byte_size_(size) {
        }

        // The name of the parametre.
        const std::string &Name() const { return name_; }

        // Data type of the parameter.
        TRITONSERVER_ParameterType Type() const { return type_; }

        // Return a pointer to the parameter, or a pointer to the data content
        // if type_ is TRITONSERVER_PARAMETER_BYTES. This returned pointer must be
        // cast correctly based on 'type_'.
        //   TRITONSERVER_PARAMETER_STRING -> const char*
        //   TRITONSERVER_PARAMETER_INT -> int64_t*
        //   TRITONSERVER_PARAMETER_BOOL -> bool*
        //   TRITONSERVER_PARAMETER_BYTES -> const void*
        const void *ValuePointer() const;

        // Return the data byte size of the parameter.
        uint64_t ValueByteSize() const { return byte_size_; }

        // Return the parameter value string, the return value is valid only if
        // Type() returns TRITONSERVER_PARAMETER_STRING
        const std::string &ValueString() const { return value_string_; }

    private:
        friend std::ostream &operator<<(
                std::ostream &out, const inference_parameter &parameter);

        std::string name_;
        TRITONSERVER_ParameterType type_;

        std::string value_string_;
        int64_t value_int64_;
        bool value_bool_;
        const void *value_bytes_;
        uint64_t byte_size_;
    };

    std::ostream &operator<<(
            std::ostream &out, const inference_parameter &parameter);

}  // namespace hercules::core
