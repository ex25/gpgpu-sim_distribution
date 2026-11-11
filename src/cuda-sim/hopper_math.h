/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2025, Faizan A. Khattak and Mantas Mikaitis
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *       and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

// BF16 block accumulation follows Khattak and Mikaitis,
// "Accurate Models of NVIDIA Tensor Cores", Sec. IV-A5 (H100).
// https://arxiv.org/html/2512.07004v2#S4.SS1.SSS5
// Reference model: north-numerical-computing/MATLAB-tensor-core,
// commit 79143bd29a4a71dc7c2ed47741b63572498b17f6, H100TC.m and
// Generic_BFMA_TC.m. Validated against recorded H100 WGMMA outputs.

#ifndef HOPPER_MATH_H
#define HOPPER_MATH_H
#include <cmath>
#include <cstdint>
#include <cstring>

inline float hopper_bf16_block(const uint16_t *a, const uint16_t *b, float c) {
  uint32_t cv;
  memcpy(&cv, &c, sizeof(cv));
  uint32_t sig[17];
  int exp[17];
  bool negative[17];
  exp[16] = ((cv >> 23) & 255) ? int((cv >> 23) & 255) - 127 : -126;
  sig[16] = ((cv & 0x7fffff) | (((cv >> 23) & 255) ? 0x800000 : 0)) << 2;
  negative[16] = cv >> 31;
  // H100 retains two fractional alignment bits beyond FP32. Products are
  // aligned before normalization using the sum of the input exponents.
  // The reference model clamps the common alignment exponent at -133.
  int max_exp = -133;
  if (sig[16] && exp[16] > max_exp) max_exp = exp[16];
  if (std::isnan(c)) return c;
  bool infinity = std::isinf(c);
  bool infinity_negative = std::signbit(c);
  for (unsigned i = 0; i < 16; ++i) {
    unsigned ea = (a[i] >> 7) & 255, eb = (b[i] >> 7) & 255;
    unsigned sa = (a[i] & 127) | (ea ? 128 : 0);
    unsigned sb = (b[i] & 127) | (eb ? 128 : 0);
    if (ea == 255 || eb == 255) {
      bool sign = (a[i] ^ b[i]) >> 15;
      if ((ea == 255 && (a[i] & 127)) || (eb == 255 && (b[i] & 127)) || !sa ||
          !sb || (infinity && sign != infinity_negative))
        return NAN;
      infinity = true;
      infinity_negative = sign;
      sig[i] = 0;
      exp[i] = 0;
      negative[i] = false;
      continue;
    }
    exp[i] = int(ea ? ea : 1) + int(eb ? eb : 1) - 254;
    // BF16 products have 14 fractional bits; alignment uses 23 + 2.
    sig[i] = (sa * sb) << 11;
    negative[i] = (a[i] ^ b[i]) >> 15;
    if (sig[i] && exp[i] > max_exp) max_exp = exp[i];
  }
  if (infinity) return infinity_negative ? -INFINITY : INFINITY;
  // Truncate each unsigned magnitude during alignment, then add signed
  // integers without rounding or normalization between the 16 products.
  int64_t total = 0;
  for (unsigned i = 0; i < 17; ++i) {
    unsigned shift = unsigned(max_exp - exp[i]);
    uint32_t aligned = shift < 32 ? sig[i] >> shift : 0;
    total += negative[i] ? -int64_t(aligned) : int64_t(aligned);
  }
  // Normalize once and truncate to FP32; BF16 conversion is a later PTX op.
  uint32_t sign = total < 0 ? 0x80000000 : 0;
  uint64_t magnitude = total < 0 ? -total : total;
  if (!magnitude) return 0;
  int top = 63 - __builtin_clzll(magnitude);
  int exponent = max_exp - 25 + top;
  uint32_t bits;
  if (exponent > 127) {
    bits = sign | 0x7f800000;
  } else if (exponent < -126) {
    int shift = max_exp - 25 + 149;
    bits =
        sign | uint32_t(shift >= 0 ? magnitude << shift : magnitude >> -shift);
  } else {
    uint32_t fraction =
        top > 23 ? magnitude >> (top - 23) : magnitude << (23 - top);
    bits = sign | (uint32_t(exponent + 127) << 23) | (fraction & 0x7fffff);
  }
  float result;
  memcpy(&result, &bits, sizeof(result));
  return result;
}
#endif
