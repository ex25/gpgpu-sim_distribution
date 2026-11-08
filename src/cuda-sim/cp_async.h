#ifndef PTX_CP_ASYNC_H
#define PTX_CP_ASYNC_H

#include <cassert>
#include <map>

// Only unfinished groups need storage. The monotonic commit number also counts
// empty and already completed groups, so wait N always refers to the newest N.
class cp_async_groups {
 public:
  unsigned long long add() {
    ++pending[committed];
    return committed;
  }
  void commit() { ++committed; }
  void complete(unsigned long long group) {
    auto it = pending.find(group);
    assert(it != pending.end() && it->second);
    if (--it->second == 0) pending.erase(it);
  }
  void wait(unsigned n) { cutoff = committed > n ? committed - n : 0; }
  bool waiting() const {
    return !pending.empty() && pending.begin()->first < cutoff;
  }
  bool outstanding() const { return !pending.empty(); }

  unsigned long long committed = 0;
  unsigned long long cutoff = 0;
  std::map<unsigned long long, unsigned> pending;
};

#endif
