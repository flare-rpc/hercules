
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/


#include "hercules/common/logging.h"

#ifdef _WIN32
// suppress the min and max definitions in Windef.h.
#define NOMINMAX
#include <Windows.h>
#else
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#endif
#include <algorithm>
#include <iomanip>
#include <iostream>

namespace hercules::common {

Logger gLogger_;

Logger::Logger() : enables_{true, true, true}, vlevel_(0), format_(Format::kDEFAULT) {}

void
Logger::Log(const std::string& msg)
{
  const std::lock_guard<std::mutex> lock(mutex_);
  std::cerr << msg << std::endl;
}

void
Logger::Flush()
{
  std::cerr << std::flush;
}


const std::vector<char> LogMessage::level_name_{'E', 'W', 'I'};

LogMessage::LogMessage(const char* file, int line, uint32_t level)
{
  std::string path(file);
  size_t pos = path.rfind('/');
  if (pos != std::string::npos) {
    path = path.substr(pos + 1, std::string::npos);
  }

  // 'L' below is placeholder for showing log level
  switch (gLogger_.LogFormat())
  {
  case Logger::Format::kDEFAULT: {
    // LMMDD hh:mm:ss.ssssss
#ifdef _WIN32
    SYSTEMTIME system_time;
    GetSystemTime(&system_time);
    stream_ << level_name_[std::min(level, (uint32_t)Level::kINFO)]
            << std::setfill('0') << std::setw(2) << system_time.wMonth
            << std::setw(2) << system_time.wDay << ' ' << std::setw(2)
            << system_time.wHour << ':' << std::setw(2) << system_time.wMinute
            << ':' << std::setw(2) << system_time.wSecond << '.' << std::setw(6)
            << system_time.wMilliseconds * 1000 << ' '
            << static_cast<uint32_t>(GetCurrentProcessId()) << ' ' << path << ':'
            << line << "] ";
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    struct tm tm_time;
    gmtime_r(((time_t*)&(tv.tv_sec)), &tm_time);
    stream_ << level_name_[std::min(level, (uint32_t)Level::kINFO)]
            << std::setfill('0') << std::setw(2) << (tm_time.tm_mon + 1)
            << std::setw(2) << tm_time.tm_mday << ' ' << std::setw(2)
            << tm_time.tm_hour << ':' << std::setw(2) << tm_time.tm_min << ':'
            << std::setw(2) << tm_time.tm_sec << '.' << std::setw(6) << tv.tv_usec
            << ' ' << static_cast<uint32_t>(getpid()) << ' ' << path << ':'
            << line << "] ";
#endif
    break;
  }
  case Logger::Format::kISO8601: {
    // YYYY-MM-DDThh:mm:ssZ L
#ifdef _WIN32
    SYSTEMTIME system_time;
    GetSystemTime(&system_time);
    stream_ << system_time.wYear << '-'
            << std::setfill('0') << std::setw(2) << system_time.wMonth << '-'
            << std::setw(2) << system_time.wDay << 'T' << std::setw(2)
            << system_time.wHour << ':' << std::setw(2) << system_time.wMinute
            << ':' << std::setw(2) << system_time.wSecond << "Z "
            << level_name_[std::min(level, (uint32_t)Level::kINFO)] << ' '
            << static_cast<uint32_t>(GetCurrentProcessId()) << ' ' << path << ':'
            << line << "] ";
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    struct tm tm_time;
    gmtime_r(((time_t*)&(tv.tv_sec)), &tm_time);
    stream_ << (tm_time.tm_year + 1900) << '-'
            << std::setfill('0') << std::setw(2) << (tm_time.tm_mon + 1) << '-'
            << std::setw(2) << tm_time.tm_mday << 'T' << std::setw(2)
            << tm_time.tm_hour << ':' << std::setw(2) << tm_time.tm_min << ':'
            << std::setw(2) << tm_time.tm_sec << "Z "
            << level_name_[std::min(level, (uint32_t)Level::kINFO)] << ' '
            << static_cast<uint32_t>(getpid()) << ' ' << path << ':'
            << line << "] ";
#endif
    break;
  }
  }
}

LogMessage::~LogMessage()
{
  gLogger_.Log(stream_.str());
}

}  // namespace hercules::common
