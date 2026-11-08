#ifndef PTX_MATRIX_H
#define PTX_MATRIX_H

#include <array>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <vector>

enum ptx_matrix_kind {
  PTX_MATRIX_NONE,
  PTX_LDMATRIX,
  PTX_STMATRIX,
  PTX_WGMMA,
  PTX_WGMMA_FENCE,
  PTX_WGMMA_COMMIT,
  PTX_WGMMA_WAIT,
  PTX_SETMAX_DEC,
  PTX_SETMAX_INC
};
class ptx_thread_info;
class symbol;
struct ptx_matrix_warp {
  std::array<ptx_thread_info *, 32> threads;
  std::array<const symbol *, 64> dst;
  std::array<float, 2048> initial{};
  uint64_t a, b;
  bool scale;
};
struct ptx_matrix_op {
  std::array<std::shared_ptr<ptx_matrix_warp>, 4> warps;
  std::array<std::shared_ptr<ptx_matrix_op>, 64> source;
  std::array<unsigned, 64> source_index{};
  std::array<float, 8192> result;
  unsigned arrived = 0, collected = 0, pending = 15;
  uint64_t read_ready = 0;
  bool computed = false;
  void compute();
  void writeback(unsigned warp);
};
// Shared by the four warps for one aligned control instruction.
struct ptx_matrix_sync {
  unsigned arrived = 0, argument = 0, kind = 0;
  bool released = false;
};
struct ptx_matrix_group {
  std::vector<std::shared_ptr<ptx_matrix_op>> ops;
  bool complete() const {
    for (const auto &op : ops)
      if (op->pending) return false;
    return true;
  }
};
struct ptx_matrix_warpgroup {
  std::array<unsigned, 4> collector_next{};
  std::map<unsigned, std::shared_ptr<ptx_matrix_op>> operations;
  std::map<unsigned, std::shared_ptr<ptx_matrix_sync>> controls;
  std::map<const symbol *, std::pair<std::weak_ptr<ptx_matrix_op>, unsigned>>
      latest;
  ptx_matrix_group uncommitted;
  std::deque<ptx_matrix_group> groups;
  bool waiting(unsigned keep) {
    while (!groups.empty() && groups.front().complete()) groups.pop_front();
    return groups.size() > keep;
  }
  unsigned quota = 0;
  std::shared_ptr<ptx_matrix_sync> quota_request;
};
struct ptx_matrix_cta {
  unsigned initial = 0, free = 0;
  std::map<unsigned, std::shared_ptr<ptx_matrix_warpgroup>> groups;
  void allocate();
};
#endif
