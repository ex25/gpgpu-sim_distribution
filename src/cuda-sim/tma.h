#ifndef HOPPER_PTX_TMA_H
#define HOPPER_PTX_TMA_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <vector>

// Logical, host-registered tiled descriptor. This is NOT CUtensorMap's ABI.
struct ptx_tma_descriptor {
  uint64_t base = 0;
  unsigned rank = 0, element_bytes = 0;
  std::array<uint64_t, 5> dimensions{}, byte_strides{};
  std::array<unsigned, 5> box{};
};

struct ptx_tma_transfer {
  unsigned element_bytes = 0;
  size_t bytes = 0, oob_bytes = 0;
  // One entry per element, including OOB positions in shared order.
  std::vector<uint64_t> addresses;
  std::vector<bool> in_bounds;
};

struct ptx_tma_statistics {
  uint64_t issue = 0, completion = 0, prefetch = 0;
  uint64_t global_reads = 0, shared_writes = 0;
  uint64_t global_bytes = 0, shared_bytes = 0, oob_bytes = 0;
  uint64_t wait_success = 0, wait_failure = 0;
  uint64_t detailed_issue = 0, global_requests = 0, global_responses = 0;
  uint64_t oob_completions = 0, in_bounds_completions = 0;
};

struct ptx_tma_dynamic_metadata {
  std::vector<uint64_t> addresses;
  unsigned element_bytes = 0;
  size_t in_bounds_bytes = 0, oob_bytes = 0;
  uint32_t barrier = 0, mask = 0;
};

// Harness-declared identity; the writer and barrier are bound only when the
// corresponding execution-driven CTA actually initializes its mbarrier.
struct ptx_tma_cta_binding {
  uint64_t kernel = 0;
  unsigned cta = 0, cluster = 0, rank = 0;
  uint32_t shared_address = 0, barrier_offset = 0;
  size_t capacity = 0;
  bool started = false;
  std::function<void(size_t, const void *, size_t)> write;
  std::function<void(size_t)> complete;
  std::function<int()> transaction_count;
};

struct ptx_tma_completion_record {
  ptx_tma_transfer transfer;
  uint32_t mask = 0;
  unsigned source_rank = 0;
  bool issued = false, finished = false;
  size_t returned_bytes = 0;
  unsigned global_requests = 0;
  std::set<uint64_t> returned_addresses;
  std::vector<uint8_t> data;
  std::vector<std::shared_ptr<ptx_tma_cta_binding>> targets;
  std::function<void(uint64_t, void *, size_t)> read;
  std::function<void(const char *, unsigned, size_t)> event;
};

inline ptx_tma_transfer ptx_tma_expand(
    const ptx_tma_descriptor &d, unsigned rank,
    const std::array<int32_t, 5> &coordinates) {
  if (rank < 1 || rank > 5 || rank != d.rank ||
      (d.element_bytes != 1 && d.element_bytes != 2 &&
       d.element_bytes != 4 && d.element_bytes != 8))
    throw std::runtime_error("unsupported TMA descriptor rank/element size");
  size_t elements = 1;
  for (unsigned r = 0; r < rank; ++r) {
    if (!d.dimensions[r] || !d.box[r] || !d.byte_strides[r] ||
        elements > (1u << 20) / d.box[r])
      throw std::runtime_error("invalid or oversized TMA descriptor");
    elements *= d.box[r];
  }
  ptx_tma_transfer result;
  result.element_bytes = d.element_bytes;
  result.bytes = elements * d.element_bytes;
  for (size_t e = 0; e < elements; ++e) {
    size_t index = e;
    uint64_t address = d.base;
    bool valid = true;
    for (unsigned r = 0; r < rank; ++r) {
      int64_t coordinate = int64_t(coordinates[r]) + index % d.box[r];
      index /= d.box[r];
      if (coordinate < 0 || uint64_t(coordinate) >= d.dimensions[r]) {
        valid = false;
        continue;
      }
      if (uint64_t(coordinate) >
          (std::numeric_limits<uint64_t>::max() - address) / d.byte_strides[r])
        throw std::runtime_error("TMA address overflow");
      address += uint64_t(coordinate) * d.byte_strides[r];
    }
    if (valid && address > std::numeric_limits<uint64_t>::max() - d.element_bytes)
      throw std::runtime_error("TMA element address overflow");
    result.addresses.push_back(address);
    result.in_bounds.push_back(valid);
    if (!valid) result.oob_bytes += d.element_bytes;
  }
  return result;
}

// Each target owns a shared-memory writer and a completion callback. The engine
// validates the complete mask before writing, and publishes completion last.
struct ptx_tma_target {
  unsigned rank;
  std::function<void(size_t, const void *, size_t)> write;
  std::function<void(size_t)> complete;
};

inline void ptx_tma_complete(
    const ptx_tma_transfer &transfer, uint32_t mask,
    const std::vector<ptx_tma_target> &targets,
    const std::function<void(uint64_t, void *, size_t)> &read) {
  uint32_t available = 0;
  for (const auto &target : targets) {
    if (target.rank >= 16 || !target.write || !target.complete ||
        (available & (1u << target.rank)))
      throw std::runtime_error("invalid TMA multicast target");
    available |= 1u << target.rank;
  }
  if (!mask || (mask & ~available))
    throw std::runtime_error("TMA multicast target is not registered");
  std::vector<uint8_t> data(transfer.bytes, 0);
  for (size_t e = 0; e < transfer.addresses.size(); ++e)
    if (transfer.in_bounds[e])
      read(transfer.addresses[e], &data[e * transfer.element_bytes], transfer.element_bytes);
  for (const auto &target : targets)
    if (mask & (1u << target.rank)) target.write(0, data.data(), data.size());
  for (const auto &target : targets)
    if (mask & (1u << target.rank)) target.complete(transfer.bytes);
}

#endif
