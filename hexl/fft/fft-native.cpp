// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "hexl/fft/fft-native.hpp"

#include <cstring>

#include "util/util-internal.hpp"

namespace intel {
namespace hexl {

inline void ComplexFwdButterflyRadix2(std::complex<double>* X_r,
                                      std::complex<double>* Y_r,
                                      const std::complex<double>* X_op,
                                      const std::complex<double>* Y_op,
                                      const std::complex<double> W) {
  std::complex<double> U = *X_op;
  std::complex<double> V = *Y_op * W;
  *X_r = U + V;
  *Y_r = U - V;
}

inline void ComplexInvButterflyRadix2(std::complex<double>* X_r,
                                      std::complex<double>* Y_r,
                                      const std::complex<double>* X_op,
                                      const std::complex<double>* Y_op,
                                      const std::complex<double> W) {
  std::complex<double> U = *X_op;
  *X_r = U + *Y_op;
  *Y_r = (U - *Y_op) * W;
}

inline void ScaledComplexInvButterflyRadix2(std::complex<double>* X_r,
                                            std::complex<double>* Y_r,
                                            const std::complex<double>* X_op,
                                            const std::complex<double>* Y_op,
                                            const std::complex<double> W,
                                            const double* scalar) {
  std::complex<double> U = *X_op;
  *X_r = (U + *Y_op) * (*scalar);
  *Y_r = (U - *Y_op) * W;
}

void Forward_FFT_ToBitReverseRadix2(
    std::complex<double>* result, const std::complex<double>* operand,
    const std::complex<double>* root_of_unity_powers, const uint64_t n,
    const double* scalar) {
  HEXL_CHECK(root_of_unity_powers != nullptr,
             "root_of_unity_powers == nullptr");
  HEXL_CHECK(operand != nullptr, "operand == nullptr");
  HEXL_CHECK(result != nullptr, "result == nullptr");

  size_t gap = (n >> 1);

  // In case of out-of-place operation do first pass and convert to in-place
  {
    const std::complex<double> W = root_of_unity_powers[1];
    std::complex<double>* X_r = result;
    std::complex<double>* Y_r = X_r + gap;
    const std::complex<double>* X_op = operand;
    const std::complex<double>* Y_op = X_op + gap;

    // First pass for out-of-order case
    switch (gap) {
      case 8: {
        ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W);
        ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W);
        ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W);
        ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W);
        ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W);
        ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W);
        ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W);
        ComplexFwdButterflyRadix2(X_r, Y_r, X_op, Y_op, W);
        break;
      }
      case 4: {
        ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W);
        ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W);
        ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W);
        ComplexFwdButterflyRadix2(X_r, Y_r, X_op, Y_op, W);
        break;
      }
      case 2: {
        ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W);
        ComplexFwdButterflyRadix2(X_r, Y_r, X_op, Y_op, W);
        break;
      }
      case 1: {
        std::complex<double> scaled_W = W;
        if (scalar != nullptr) scaled_W = W * *scalar;
        ComplexFwdButterflyRadix2(X_r, Y_r, X_op, Y_op, W);
        break;
      }
      default: {
        HEXL_LOOP_UNROLL_8
        for (size_t j = 0; j < gap; j += 8) {
          ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W);
          ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W);
          ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W);
          ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W);
          ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W);
          ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W);
          ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W);
          ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W);
        }
      }
    }
    gap >>= 1;
  }

  // Continue with in-place operation
  for (size_t m = 2; m < n; m <<= 1) {
    size_t offset = 0;
    switch (gap) {
      case 8: {
        for (size_t i = 0; i < m; i++) {
          if (i != 0) {
            offset += (gap << 1);
          }
          const std::complex<double> W = root_of_unity_powers[m + i];
          std::complex<double>* X_r = result + offset;
          std::complex<double>* Y_r = X_r + gap;
          const std::complex<double>* X_op = X_r;
          const std::complex<double>* Y_op = Y_r;
          ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W);
          ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W);
          ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W);
          ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W);
          ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W);
          ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W);
          ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W);
          ComplexFwdButterflyRadix2(X_r, Y_r, X_op, Y_op, W);
        }
        break;
      }
      case 4: {
        for (size_t i = 0; i < m; i++) {
          if (i != 0) {
            offset += (gap << 1);
          }
          const std::complex<double> W = root_of_unity_powers[m + i];
          std::complex<double>* X_r = result + offset;
          std::complex<double>* Y_r = X_r + gap;
          const std::complex<double>* X_op = X_r;
          const std::complex<double>* Y_op = Y_r;
          ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W);
          ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W);
          ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W);
          ComplexFwdButterflyRadix2(X_r, Y_r, X_op, Y_op, W);
        }
        break;
      }
      case 2: {
        for (size_t i = 0; i < m; i++) {
          if (i != 0) {
            offset += (gap << 1);
          }
          const std::complex<double> W = root_of_unity_powers[m + i];
          std::complex<double>* X_r = result + offset;
          std::complex<double>* Y_r = X_r + gap;
          const std::complex<double>* X_op = X_r;
          const std::complex<double>* Y_op = Y_r;
          ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W);
          ComplexFwdButterflyRadix2(X_r, Y_r, X_op, Y_op, W);
        }
        break;
      }
      case 1: {
        if (scalar == nullptr) {
          for (size_t i = 0; i < m; i++) {
            if (i != 0) {
              offset += (gap << 1);
            }
            const std::complex<double> W = root_of_unity_powers[m + i];
            std::complex<double>* X_r = result + offset;
            std::complex<double>* Y_r = X_r + gap;
            const std::complex<double>* X_op = X_r;
            const std::complex<double>* Y_op = Y_r;
            ComplexFwdButterflyRadix2(X_r, Y_r, X_op, Y_op, W);
          }
        } else {
          for (size_t i = 0; i < m; i++) {
            if (i != 0) {
              offset += (gap << 1);
            }
            const std::complex<double> W =
                *scalar * root_of_unity_powers[m + i];
            std::complex<double>* X_r = result + offset;
            std::complex<double>* Y_r = X_r + gap;
            *X_r = (*scalar) * (*X_r);
            const std::complex<double>* X_op = X_r;
            const std::complex<double>* Y_op = Y_r;
            ComplexFwdButterflyRadix2(X_r, Y_r, X_op, Y_op, W);
          }
        }
        break;
      }
      default: {
        for (size_t i = 0; i < m; i++) {
          if (i != 0) {
            offset += (gap << 1);
          }
          const std::complex<double> W = root_of_unity_powers[m + i];
          std::complex<double>* X_r = result + offset;
          std::complex<double>* Y_r = X_r + gap;
          const std::complex<double>* X_op = X_r;
          const std::complex<double>* Y_op = Y_r;
          HEXL_LOOP_UNROLL_8
          for (size_t j = 0; j < gap; j += 8) {
            ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W);
            ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W);
            ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W);
            ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W);
            ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W);
            ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W);
            ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W);
            ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W);
          }
        }
      }
    }
    gap >>= 1;
  }
}

void Inverse_FFT_FromBitReverseRadix2(
    std::complex<double>* result, const std::complex<double>* operand,
    const std::complex<double>* inv_root_of_unity_powers, const uint64_t n,
    const double* scalar) {
  HEXL_CHECK(inv_root_of_unity_powers != nullptr,
             "inv_root_of_unity_powers == nullptr");
  HEXL_CHECK(operand != nullptr, "operand == nullptr");
  HEXL_CHECK(result != nullptr, "result == nullptr");

  uint64_t n_div_2 = (n >> 1);
  size_t gap = 1;
  size_t root_index = 1;

  size_t stop_loop = (scalar == nullptr) ? 0 : 1;
  size_t m = n_div_2;
  for (; m > stop_loop; m >>= 1) {
    size_t offset = 0;

    switch (gap) {
      case 1: {
        for (size_t i = 0; i < m; i++, root_index++) {
          if (i != 0) {
            offset += (gap << 1);
          }
          const std::complex<double> W = inv_root_of_unity_powers[root_index];

          std::complex<double>* X_r = result + offset;
          std::complex<double>* Y_r = X_r + gap;
          const std::complex<double>* X_op = operand + offset;
          const std::complex<double>* Y_op = X_op + gap;
          ComplexInvButterflyRadix2(X_r, Y_r, X_op, Y_op, W);
        }
        break;
      }
      case 2: {
        for (size_t i = 0; i < m; i++, root_index++) {
          if (i != 0) {
            offset += (gap << 1);
          }
          const std::complex<double> W = inv_root_of_unity_powers[root_index];
          std::complex<double>* X_r = result + offset;
          std::complex<double>* Y_r = X_r + gap;
          const std::complex<double>* X_op = X_r;
          const std::complex<double>* Y_op = Y_r;
          ComplexInvButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W);
          ComplexInvButterflyRadix2(X_r, Y_r, X_op, Y_op, W);
        }
        break;
      }
      case 4: {
        for (size_t i = 0; i < m; i++, root_index++) {
          if (i != 0) {
            offset += (gap << 1);
          }
          const std::complex<double> W = inv_root_of_unity_powers[root_index];
          std::complex<double>* X_r = result + offset;
          std::complex<double>* Y_r = X_r + gap;
          const std::complex<double>* X_op = X_r;
          const std::complex<double>* Y_op = Y_r;
          ComplexInvButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W);
          ComplexInvButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W);
          ComplexInvButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W);
          ComplexInvButterflyRadix2(X_r, Y_r, X_op, Y_op, W);
        }
        break;
      }
      case 8: {
        for (size_t i = 0; i < m; i++, root_index++) {
          if (i != 0) {
            offset += (gap << 1);
          }
          const std::complex<double> W = inv_root_of_unity_powers[root_index];
          std::complex<double>* X_r = result + offset;
          std::complex<double>* Y_r = X_r + gap;
          const std::complex<double>* X_op = X_r;
          const std::complex<double>* Y_op = Y_r;
          ComplexInvButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W);
          ComplexInvButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W);
          ComplexInvButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W);
          ComplexInvButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W);
          ComplexInvButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W);
          ComplexInvButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W);
          ComplexInvButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W);
          ComplexInvButterflyRadix2(X_r, Y_r, X_op, Y_op, W);
        }
        break;
      }
      default: {
        for (size_t i = 0; i < m; i++, root_index++) {
          if (i != 0) {
            offset += (gap << 1);
          }
          const std::complex<double> W = inv_root_of_unity_powers[root_index];
          std::complex<double>* X_r = result + offset;
          std::complex<double>* Y_r = X_r + gap;
          const std::complex<double>* X_op = X_r;
          const std::complex<double>* Y_op = Y_r;

          HEXL_LOOP_UNROLL_8
          for (size_t j = 0; j < gap; j += 8) {
            ComplexInvButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W);
            ComplexInvButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W);
            ComplexInvButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W);
            ComplexInvButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W);
            ComplexInvButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W);
            ComplexInvButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W);
            ComplexInvButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W);
            ComplexInvButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W);
          }
        }
      }
    }
    gap <<= 1;
  }

  if (m > 0) {
    const std::complex<double> W =
        *scalar * inv_root_of_unity_powers[root_index];
    std::complex<double>* X_r = result;
    std::complex<double>* Y_r = X_r + gap;
    const std::complex<double>* X_o = X_r;
    const std::complex<double>* Y_o = Y_r;

    switch (gap) {
      case 1: {
        ScaledComplexInvButterflyRadix2(X_r, Y_r, X_o, Y_o, W, scalar);
        break;
      }
      case 2: {
        ScaledComplexInvButterflyRadix2(X_r++, Y_r++, X_o++, Y_o++, W, scalar);
        ScaledComplexInvButterflyRadix2(X_r, Y_r, X_o, Y_o, W, scalar);
        break;
      }
      case 4: {
        ScaledComplexInvButterflyRadix2(X_r++, Y_r++, X_o++, Y_o++, W, scalar);
        ScaledComplexInvButterflyRadix2(X_r++, Y_r++, X_o++, Y_o++, W, scalar);
        ScaledComplexInvButterflyRadix2(X_r++, Y_r++, X_o++, Y_o++, W, scalar);
        ScaledComplexInvButterflyRadix2(X_r, Y_r, X_o, Y_o, W, scalar);
        break;
      }
      case 8: {
        ScaledComplexInvButterflyRadix2(X_r++, Y_r++, X_o++, Y_o++, W, scalar);
        ScaledComplexInvButterflyRadix2(X_r++, Y_r++, X_o++, Y_o++, W, scalar);
        ScaledComplexInvButterflyRadix2(X_r++, Y_r++, X_o++, Y_o++, W, scalar);
        ScaledComplexInvButterflyRadix2(X_r++, Y_r++, X_o++, Y_o++, W, scalar);
        ScaledComplexInvButterflyRadix2(X_r++, Y_r++, X_o++, Y_o++, W, scalar);
        ScaledComplexInvButterflyRadix2(X_r++, Y_r++, X_o++, Y_o++, W, scalar);
        ScaledComplexInvButterflyRadix2(X_r++, Y_r++, X_o++, Y_o++, W, scalar);
        ScaledComplexInvButterflyRadix2(X_r, Y_r, X_o, Y_o, W, scalar);
        break;
      }
      default: {
        HEXL_LOOP_UNROLL_8
        for (size_t j = 0; j < gap; j += 8) {
          ScaledComplexInvButterflyRadix2(X_r++, Y_r++, X_o++, Y_o++, W,
                                          scalar);
          ScaledComplexInvButterflyRadix2(X_r++, Y_r++, X_o++, Y_o++, W,
                                          scalar);
          ScaledComplexInvButterflyRadix2(X_r++, Y_r++, X_o++, Y_o++, W,
                                          scalar);
          ScaledComplexInvButterflyRadix2(X_r++, Y_r++, X_o++, Y_o++, W,
                                          scalar);
          ScaledComplexInvButterflyRadix2(X_r++, Y_r++, X_o++, Y_o++, W,
                                          scalar);
          ScaledComplexInvButterflyRadix2(X_r++, Y_r++, X_o++, Y_o++, W,
                                          scalar);
          ScaledComplexInvButterflyRadix2(X_r++, Y_r++, X_o++, Y_o++, W,
                                          scalar);
          ScaledComplexInvButterflyRadix2(X_r++, Y_r++, X_o++, Y_o++, W,
                                          scalar);
        }
      }
    }
  }

  // When M is too short it only needs the final stage butterfly. Copying here
  // in the case of out-of-place.
  if (result != operand && n == 2) {
    std::memcpy(result, operand, n * sizeof(std::complex<double>));
  }
}

}  // namespace hexl
}  // namespace intel
