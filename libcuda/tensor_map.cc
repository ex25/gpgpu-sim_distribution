#include "../src/cuda-sim/tensor_map.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cassert>

#if CUDART_VERSION >= 12000
// The application copies these bytes into parameter/global/shared memory.
static CUresult CUDAAPI sim_tensor_map_encode(
    CUtensorMap *output, CUtensorMapDataType type, cuuint32_t rank, void *base,
    const cuuint64_t *dims, const cuuint64_t *strides, const cuuint32_t *box,
    const cuuint32_t *element, CUtensorMapInterleave interleave,
    CUtensorMapSwizzle swizzle, CUtensorMapL2promotion promotion,
    CUtensorMapFloatOOBfill fill) {
  assert(rank == 2 && (type == CU_TENSOR_MAP_DATA_TYPE_UINT16 ||
                       type == CU_TENSOR_MAP_DATA_TYPE_BFLOAT16));
  assert(interleave == CU_TENSOR_MAP_INTERLEAVE_NONE);
  assert(swizzle == CU_TENSOR_MAP_SWIZZLE_64B ||
         swizzle == CU_TENSOR_MAP_SWIZZLE_128B);
  ptx_tensor_map map;
  map.set<uint64_t>(0, reinterpret_cast<uint64_t>(base));
  for (unsigned i = 0; i < rank; ++i) {
    assert(dims[i] <= UINT32_MAX && element[i] == 1);
    map.set<uint32_t>(40 + 4 * i, dims[i]);
    map.set<uint32_t>(60 + 4 * i, box[i]);
    map.set<uint32_t>(80 + 4 * i, element[i]);
  }
  map.set<uint64_t>(8, strides[0]);
  const uint32_t flags[] = {rank,
                            uint32_t(type),
                            uint32_t(interleave),
                            uint32_t(swizzle),
                            uint32_t(promotion),
                            uint32_t(fill)};
  memcpy(map.bytes + 100, flags, sizeof(flags));
  memcpy(output, map.bytes, 128);
  return CUDA_SUCCESS;
}

extern "C" cudaError_t CUDARTAPI cudaGetDriverEntryPointByVersion(
    const char *symbol, void **pointer, unsigned version,
    unsigned long long flags, cudaDriverEntryPointQueryResult *status) {
  if (strcmp(symbol, "cuTensorMapEncodeTiled") || version != 12000 ||
      flags != cudaEnableDefault) {
    if (status) *status = cudaDriverEntryPointSymbolNotFound;
    *pointer = nullptr;
    return cudaErrorInvalidValue;
  }
  *pointer = reinterpret_cast<void *>(&sim_tensor_map_encode);
  if (status) *status = cudaDriverEntryPointSuccess;
  return cudaSuccess;
}
#endif
