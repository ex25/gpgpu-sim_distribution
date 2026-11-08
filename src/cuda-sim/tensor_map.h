#ifndef PTX_TENSOR_MAP_H
#define PTX_TENSOR_MAP_H

#include <cstdint>
#include <cstring>

// Simulator-owned, byte-copyable ABI; this is not NVIDIA's opaque encoding.
struct ptx_tensor_map {
  unsigned char bytes[128]{};
  template <class T>
  T get(unsigned offset) const {
    T value;
    std::memcpy(&value, bytes + offset, sizeof(T));
    return value;
  }
  template <class T>
  void set(unsigned offset, T value) {
    std::memcpy(bytes + offset, &value, sizeof(T));
  }
};

inline unsigned ptx_tensor_shared_address(unsigned base, unsigned offset,
                                          unsigned swizzle) {
  unsigned address = base + offset;
  unsigned mask = swizzle == 2 ? 3 : 7;  // CUDA 64B / 128B enums
  return address ^ (((address >> 7) & mask) << 4);
}

enum ptx_tensor_kind {
  PTX_TENSOR_NONE,
  PTX_TENSOR_ADDRESS,
  PTX_TENSOR_DIM,
  PTX_TENSOR_STRIDE,
  PTX_TENSOR_PUBLISH,
  PTX_TENSOR_ACQUIRE,
  PTX_ASYNC_FENCE,
  PTX_TENSOR_LOAD,
  PTX_TENSOR_MULTICAST,
  PTX_TENSOR_STORE,
  PTX_BULK_COMMIT,
  PTX_BULK_WAIT
};
#endif
