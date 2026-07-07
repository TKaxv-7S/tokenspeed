// Copyright (c) 2026 LightSeek Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <cstdint>
#include <list>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "utils.h"

namespace tokenspeed {

// Host page-id index for the flat L2 tier: key -> page, LRU eviction of
// indexed entries; in-flight (Begin..Commit/Abort) entries are unevictable.
// Touch of an unknown key is a no-op (reads race with eviction); misuse of
// the in-flight protocol throws.
class HostBlockStore {
public:
    // Host ids are 0-based over the Python pinned pool; unrelated to PageAllocator's 1-based radix host ids.
    explicit HostBlockStore(std::int32_t num_pages) {
        _assert(num_pages >= 0, "num_pages must be >= 0");
        free_.reserve(static_cast<std::size_t>(num_pages));
        for (std::int32_t i = num_pages - 1; i >= 0; --i) {
            free_.push_back(i);
        }
    }

    std::optional<std::int32_t> Lookup(const std::string& key) const {
        auto it = index_.find(key);
        if (it == index_.end()) {
            return std::nullopt;
        }
        return it->second.page;
    }

    std::optional<std::int32_t> BeginStore(const std::string& key) {
        if (index_.count(key) != 0 || in_flight_.count(key) != 0) {
            return std::nullopt;
        }
        if (free_.empty() && !evictLru()) {
            return std::nullopt;
        }
        std::int32_t page = free_.back();
        free_.pop_back();
        in_flight_.emplace(key, page);
        return page;
    }

    void CommitStore(const std::string& key, std::int32_t page) {
        eraseInFlight(key, page);
        lru_.push_back(key);
        index_.emplace(key, Entry{page, std::prev(lru_.end())});
    }

    void AbortStore(const std::string& key, std::int32_t page) {
        eraseInFlight(key, page);
        free_.push_back(page);
    }

    void Touch(const std::string& key) {
        auto it = index_.find(key);
        if (it == index_.end()) {
            return;
        }
        lru_.splice(lru_.end(), lru_, it->second.lru_it);
    }

    std::int32_t NumFreePages() const { return static_cast<std::int32_t>(free_.size()); }
    std::int32_t NumIndexed() const { return static_cast<std::int32_t>(index_.size()); }

private:
    struct Entry {
        std::int32_t page;
        std::list<std::string>::iterator lru_it;
    };

    bool evictLru() {
        if (lru_.empty()) {
            return false;
        }
        auto it = index_.find(lru_.front());
        _assert(it != index_.end(), "LRU/index out of sync");
        free_.push_back(it->second.page);
        index_.erase(it);
        lru_.pop_front();
        return true;
    }

    void eraseInFlight(const std::string& key, std::int32_t page) {
        auto it = in_flight_.find(key);
        _assert(it != in_flight_.end() && it->second == page, "unknown in-flight store");
        in_flight_.erase(it);
    }

    std::vector<std::int32_t> free_;
    std::unordered_map<std::string, Entry> index_;
    std::unordered_map<std::string, std::int32_t> in_flight_;
    std::list<std::string> lru_;
};

}  // namespace tokenspeed
