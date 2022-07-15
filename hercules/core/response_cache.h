
/****************************************************************
 * Copyright (c) 2022, liyinbin
 * All rights reserved.
 * Author by liyinbin (jeff.li) lijippy@163.com
 *****************************************************************/


#pragma once

#include <list>
#include <string>
#include <unordered_map>

#include "infer_request.h"
#include "infer_response.h"
#include "model.h"
#include "status.h"

#include <boost/functional/hash.hpp>
#include <boost/interprocess/managed_external_buffer.hpp>

namespace hercules::core {

    // Assuming CPU memory only for now
    struct Output {
        // Output tensor data buffer
        void *buffer_;
        // Size of "buffer" above
        uint64_t buffer_size_ = 0;
        // Name of the output
        std::string name_;
        // Datatype of the output
        hercules::proto::DataType dtype_;
        // Shape of the output
        std::vector<int64_t> shape_;
    };

    struct CacheEntry {
        explicit CacheEntry() {}

        // Point to key in LRU list for maintaining LRU order
        std::list<uint64_t>::iterator lru_iter_;
        // each output buffer = managed_buffer.allocate(size, ...)
        std::vector<Output> outputs_;
    };

    class RequestResponseCache {
    public:
        ~RequestResponseCache();

        // Create the request/response cache object
        static Status Create(
                uint64_t cache_size, std::unique_ptr<RequestResponseCache> *cache);

        // Hash inference request for cache access and store it in "request" object.
        // This will also be called internally in Lookup/Insert if the request hasn't
        // already stored it's hash. It is up to the user to update the hash in the
        // request if modifying any hashed fields of the request object after storing.
        // Return Status object indicating success or failure.
        Status HashAndSet(inference_request *const request);

        // Lookup 'request' hash in cache and return the inference response in
        // 'response' on cache hit or nullptr on cache miss
        // Return Status object indicating success or failure.
        Status Lookup(
                InferenceResponse *const response, inference_request *const request);

        // Insert response into cache, evict entries to make space if necessary
        // Return Status object indicating success or failure.
        Status Insert(
                const InferenceResponse &response, inference_request *const request);

        // Evict entry from cache based on policy
        // Return Status object indicating success or failure.
        Status Evict();

        // Returns number of items in cache
        size_t NumEntries() {
            std::lock_guard<std::recursive_mutex> lk(cache_mtx_);
            return cache_.size();
        }

        // Returns number of items evicted in cache lifespan
        size_t NumEvictions() {
            std::lock_guard<std::recursive_mutex> lk(cache_mtx_);
            return num_evictions_;
        }

        // Returns number of lookups in cache lifespan, should sum to hits + misses
        size_t NumLookups() {
            std::lock_guard<std::recursive_mutex> lk(cache_mtx_);
            return num_lookups_;
        }

        // Returns number of cache hits in cache lifespan
        size_t NumHits() {
            std::lock_guard<std::recursive_mutex> lk(cache_mtx_);
            return num_hits_;
        }

        // Returns number of cache hits in cache lifespan
        size_t NumMisses() {
            std::lock_guard<std::recursive_mutex> lk(cache_mtx_);
            return num_misses_;
        }

        // Returns the total lookup latency (nanoseconds) of all lookups in cache
        // lifespan
        uint64_t TotalLookupLatencyNs() {
            std::lock_guard<std::recursive_mutex> lk(cache_mtx_);
            return total_lookup_latency_ns_;
        }

        uint64_t TotalInsertionLatencyNs() {
            std::lock_guard<std::recursive_mutex> lk(cache_mtx_);
            return total_insertion_latency_ns_;
        }

        // Returns total number of bytes allocated for cache
        size_t TotalBytes() {
            std::lock_guard<std::recursive_mutex> lk(buffer_mtx_);
            return managed_buffer_.get_size();
        }

        // Returns number of free bytes in cache
        size_t FreeBytes() {
            std::lock_guard<std::recursive_mutex> lk(buffer_mtx_);
            return managed_buffer_.get_free_memory();
        }

        // Returns number of bytes in use by cache
        size_t AllocatedBytes() {
            std::lock_guard<std::recursive_mutex> lk(buffer_mtx_);
            return managed_buffer_.get_size() - managed_buffer_.get_free_memory();
        }

        // Returns fraction of bytes allocated over total cache size between [0, 1]
        double TotalUtilization() {
            std::lock_guard<std::recursive_mutex> lk(buffer_mtx_);
            return static_cast<double>(AllocatedBytes()) /
                   static_cast<double>(TotalBytes());
        }

    private:
        explicit RequestResponseCache(const uint64_t cache_size);

        // Update LRU ordering on lookup
        void UpdateLRU(std::unordered_map<uint64_t, CacheEntry>::iterator &);

        // Build CacheEntry from InferenceResponse
        Status BuildCacheEntry(
                const InferenceResponse &response, CacheEntry *const entry);

        // Build InferenceResponse from CacheEntry
        Status BuildInferenceResponse(
                const CacheEntry &entry, InferenceResponse *const response);

        // Helper function to hash data buffers used by "input"
        Status HashInputBuffers(const inference_request::Input *input, size_t *seed);

        // Helper function to hash each input in "request"
        Status HashInputs(const inference_request &request, size_t *seed);

        // Helper function to hash request and store it in "key"
        Status Hash(const inference_request &request, uint64_t *key);

        // Cache buffer
        void *buffer_;
        // Managed buffer
        boost::interprocess::managed_external_buffer managed_buffer_;
        // key -> CacheEntry containing values and list iterator for LRU management
        std::unordered_map<uint64_t, CacheEntry> cache_;
        // List of keys sorted from most to least recently used
        std::list<uint64_t> lru_;
        // Cache metrics
        size_t num_evictions_ = 0;
        size_t num_lookups_ = 0;
        size_t num_hits_ = 0;
        size_t num_misses_ = 0;
        uint64_t total_lookup_latency_ns_ = 0;
        uint64_t total_insertion_latency_ns_ = 0;
        // Mutex for buffer synchronization
        std::recursive_mutex buffer_mtx_;
        // Mutex for cache synchronization
        std::recursive_mutex cache_mtx_;
    };

}  // namespace hercules::core
