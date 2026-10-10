// Modern Driver API boundaries use the installed CUDA header rather than
// duplicating opaque tensor-map or extensible launch ABI layouts.
#include <cuda.h>
#include <cstdio>

// Referenced by cuInit so static linking retains the modern API exports.
extern "C" void gpgpu_driver_modern_anchor() {}

#if CUDA_VERSION >= 12000
CUresult CUDAAPI cuLaunchKernelEx(const CUlaunchConfig *config, CUfunction f,
                                 void **kernelParams, void **extra) {
  if (!config || (config->numAttrs && !config->attrs))
    return CUDA_ERROR_INVALID_VALUE;
  if (config->numAttrs) {
    std::fprintf(stderr,
                 "GPGPU-Sim Driver API: cuLaunchKernelEx: "
                 "CUDA_ERROR_NOT_SUPPORTED attrs=%u first_id=%d\n",
                 config->numAttrs, int(config->attrs[0].id));
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  return cuLaunchKernel(f, config->gridDimX, config->gridDimY, config->gridDimZ,
                        config->blockDimX, config->blockDimY, config->blockDimZ,
                        config->sharedMemBytes, config->hStream, kernelParams,
                        extra);
}

CUresult CUDAAPI cuOccupancyMaxActiveClusters(int *numClusters, CUfunction func,
                                             const CUlaunchConfig *config) {
  if (!numClusters || !config || !func) return CUDA_ERROR_INVALID_VALUE;
  std::fprintf(stderr, "GPGPU-Sim Driver API: cuOccupancyMaxActiveClusters: "
                       "CUDA_ERROR_NOT_SUPPORTED\n");
  return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuTensorMapEncodeTiled(
    CUtensorMap *tensorMap, CUtensorMapDataType tensorDataType,
    cuuint32_t tensorRank, void *globalAddress, const cuuint64_t *globalDim,
    const cuuint64_t *globalStrides, const cuuint32_t *boxDim,
    const cuuint32_t *elementStrides, CUtensorMapInterleave interleave,
    CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion,
    CUtensorMapFloatOOBfill oobFill) {
  if (!tensorMap || !globalAddress || !globalDim || !boxDim || !elementStrides ||
      tensorRank < 1 || tensorRank > 5 || (tensorRank > 1 && !globalStrides))
    return CUDA_ERROR_INVALID_VALUE;
  // A simulator fixture descriptor is not NVIDIA's opaque 128-byte ABI.
  // Preserve caller storage, and expose the rejected request in the log.
  std::fprintf(stderr,
               "DRIVER_DESCRIPTOR api=cuTensorMapEncodeTiled status=REJECTED "
               "error=CUDA_ERROR_NOT_SUPPORTED rank=%u type=%d swizzle=%d\n",
               tensorRank, int(tensorDataType), int(swizzle));
  return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult CUDAAPI cuTensorMapReplaceAddress(CUtensorMap *tensorMap,
                                          void *globalAddress) {
  if (!tensorMap || !globalAddress) return CUDA_ERROR_INVALID_VALUE;
  std::fprintf(stderr, "GPGPU-Sim Driver API: cuTensorMapReplaceAddress: "
                       "CUDA_ERROR_NOT_SUPPORTED\n");
  return CUDA_ERROR_NOT_SUPPORTED;
}
#endif
