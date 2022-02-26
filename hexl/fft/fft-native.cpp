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

  std::complex<double> Wg(1, 0);
  static constexpr double PI_ = 3.1415926535897932384626433832795028842;
  size_t m = 1;

  // In case of out-of-place operation do first pass and convert to in-place
  {
    const std::complex<double> W = root_of_unity_powers[1];
    std::complex<double>* X_r = result;
    std::complex<double>* Y_r = X_r + gap;
    const std::complex<double>* X_op = operand;
    const std::complex<double>* Y_op = X_op + gap;

    std::cout << "S = " << gap << " m = " << m << std::endl;
    int xc = 0;
    int yc = gap;
    // First pass for out-of-order case
    switch (m) {
      case 8: {
        std::cout << "J1 = " << j1 << " gap = " << gap << " step = 8 "
                  << std::endl;
        std::cout << "\t x = " << xc << " y = " << yc << std::endl;
        ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, Wg);
        xc++;
        yc++;
        std::cout << "\t x = " << xc << " y = " << yc << std::endl;
        ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, Wg);
        xc++;
        yc++;
        std::cout << "\t x = " << xc << " y = " << yc << std::endl;
        ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, Wg);
        xc++;
        yc++;
        std::cout << "\t x = " << xc << " y = " << yc << std::endl;
        ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, Wg);
        xc++;
        yc++;
        std::cout << "\t x = " << xc << " y = " << yc << std::endl;
        ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, Wg);
        xc++;
        yc++;
        std::cout << "\t x = " << xc << " y = " << yc << std::endl;
        ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, Wg);
        xc++;
        yc++;
        std::cout << "\t x = " << xc << " y = " << yc << std::endl;
        ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, Wg);
        xc++;
        yc++;
        std::cout << "\t x = " << xc << " y = " << yc << std::endl;
        ComplexFwdButterflyRadix2(X_r, Y_r, X_op, Y_op, Wg);
        break;
      }
      case 4: {
        std::cout << "J1 = " << j1 << " gap = " << gap << " step = 4 "
                  << std::endl;
        std::cout << "\t x = " << xc << " y = " << yc << std::endl;
        ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, Wg);
        xc++;
        yc++;
        std::cout << "\t x = " << xc << " y = " << yc << std::endl;
        ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, Wg);
        xc++;
        yc++;
        std::cout << "\t x = " << xc << " y = " << yc << std::endl;
        ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, Wg);
        xc++;
        yc++;
        ComplexFwdButterflyRadix2(X_r, Y_r, X_op, Y_op, Wg);
        break;
      }
      case 2: {
        std::cout << "J1 = " << j1 << " gap = " << gap << " step = 2 "
                  << std::endl;
        std::cout << "\t x = " << xc << " y = " << yc << std::endl;
        ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, Wg);
        xc++;
        yc++;
        ComplexFwdButterflyRadix2(X_r, Y_r, X_op, Y_op, Wg);
        break;
      }
      case 1: {
        std::cout << "J1 = " << j1 << " gap = " << gap << " step = 1 "
                  << std::endl;
        std::complex<double> scaled_W = W;
        if (scalar != nullptr) scaled_W = W * *scalar;
        ComplexFwdButterflyRadix2(X_r, Y_r, X_op, Y_op, Wg);
        break;
      }
      default: {
        std::cout << "J1 = " << j1 << " gap = " << gap << " step = 8 "
                  << std::endl;
        HEXL_LOOP_UNROLL_8
        for (size_t j = 0; j < m; j += 8) {
          std::cout << "\t x = " << xc << " y = " << yc << std::endl;
          ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, Wg);
          xc++;
          yc++;
          std::cout << "\t x = " << xc << " y = " << yc << std::endl;
          ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, Wg);
          xc++;
          yc++;
          std::cout << "\t x = " << xc << " y = " << yc << std::endl;
          ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, Wg);
          xc++;
          yc++;
          std::cout << "\t x = " << xc << " y = " << yc << std::endl;
          ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, Wg);
          xc++;
          yc++;
          std::cout << "\t x = " << xc << " y = " << yc << std::endl;
          ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, Wg);
          xc++;
          yc++;
          std::cout << "\t x = " << xc << " y = " << yc << std::endl;
          ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, Wg);
          xc++;
          yc++;
          std::cout << "\t x = " << xc << " y = " << yc << std::endl;
          ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, Wg);
          xc++;
          yc++;
          std::cout << "\t x = " << xc << " y = " << yc << std::endl;
          ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, Wg);
          xc++;
          yc++;
        }
      }
    }
    gap >>= 1;
    m <<= 1;
  }

  // Continue with in-place operation
  const std::complex<double> J(0, 1);
  for (; m < n; m <<= 1) {
    std::cout << "S = " << gap << " m = " << m << std::endl;
    std::complex<double> Wm = std::exp(J * (PI_ / static_cast<int> gap));
    size_t j1 = 0;
    switch (m) {
      case 8: {
        for (size_t i = 0; i < gap; i++) {
          if (i != 0) {
            j1 += (m << 1);
          }
          std::cout << "J1 = " << j1 << " gap = " << gap << " step = 8 "
                    << std::endl;
          const std::complex<double> W = root_of_unity_powers[m + i];
          std::complex<double>* X_r = result + j1;
          std::complex<double>* Y_r = X_r + m;
          const std::complex<double>* X_op = X_r;
          const std::complex<double>* Y_op = Y_r;
          int xc = j1;
          int yc = xc + m;

          std::cout << "\t x = " << xc << " y = " << yc << std::endl;
          ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, Wg);
          xc++;
          yc++;
          std::cout << "\t x = " << xc << " y = " << yc << std::endl;
          ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, Wg);
          xc++;
          yc++;
          std::cout << "\t x = " << xc << " y = " << yc << std::endl;
          ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, Wg);
          xc++;
          yc++;
          std::cout << "\t x = " << xc << " y = " << yc << std::endl;
          ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, Wg);
          xc++;
          yc++;
          std::cout << "\t x = " << xc << " y = " << yc << std::endl;
          ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, Wg);
          xc++;
          yc++;
          std::cout << "\t x = " << xc << " y = " << yc << std::endl;
          ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, Wg);
          xc++;
          yc++;
          std::cout << "\t x = " << xc << " y = " << yc << std::endl;
          ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, Wg);
          xc++;
          yc++;
          std::cout << "\t x = " << xc << " y = " << yc << std::endl;
          ComplexFwdButterflyRadix2(X_r, Y_r, X_op, Y_op, Wg);
        }
        break;
        Wg *= Wm;
      }
      case 4: {
        for (size_t i = 0; i < gap; i++) {
          if (i != 0) {
            j1 += (m << 1);
          }
          std::cout << "J1 = " << j1 << " gap = " << gap << " step = 4 "
                    << std::endl;
          const std::complex<double> W = root_of_unity_powers[m + i];
          std::complex<double>* X_r = result + j1;
          std::complex<double>* Y_r = X_r + m;
          const std::complex<double>* X_op = X_r;
          const std::complex<double>* Y_op = Y_r;
          int xc = j1;
          int yc = xc + m;

          std::cout << "\t x = " << xc << " y = " << yc << std::endl;
          ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, Wg);
          xc++;
          yc++;
          std::cout << "\t x = " << xc << " y = " << yc << std::endl;
          ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, Wg);
          xc++;
          yc++;
          std::cout << "\t x = " << xc << " y = " << yc << std::endl;
          ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, Wg);
          xc++;
          yc++;
          std::cout << "\t x = " << xc << " y = " << yc << std::endl;
          ComplexFwdButterflyRadix2(X_r, Y_r, X_op, Y_op, Wg);
          Wg *= Wm;
        }
        break;
      }
      case 2: {
        for (size_t i = 0; i < gap; i++) {
          if (i != 0) {
            j1 += (m << 1);
          }
          std::cout << "J1 = " << j1 << " gap = " << gap << " step = 2 "
                    << std::endl;
          const std::complex<double> W = root_of_unity_powers[m + i];
          std::complex<double>* X_r = result + j1;
          std::complex<double>* Y_r = X_r + m;
          const std::complex<double>* X_op = X_r;
          const std::complex<double>* Y_op = Y_r;
          int xc = j1;
          int yc = xc + m;

          std::cout << "\t x = " << xc << " y = " << yc << std::endl;
          ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, Wg);
          xc++;
          yc++;
          std::cout << "\t x = " << xc << " y = " << yc << std::endl;
          ComplexFwdButterflyRadix2(X_r, Y_r, X_op, Y_op, Wg);
          Wg *= Wm;
        }
        break;
      }
      case 1: {
        if (scalar == nullptr) {
          for (size_t i = 0; i < gap; i++) {
            if (i != 0) {
              j1 += (m << 1);
            }
            std::cout << "J1 = " << j1 << " gap = " << gap << " step = 1 "
                      << std::endl;
            const std::complex<double> W = root_of_unity_powers[m + i];
            std::complex<double>* X_r = result + j1;
            std::complex<double>* Y_r = X_r + m;
            const std::complex<double>* X_op = X_r;
            const std::complex<double>* Y_op = Y_r;
            int xc = j1;
            int yc = xc + m;
            std::cout << "\t x = " << xc << " y = " << yc << std::endl;
            ComplexFwdButterflyRadix2(X_r, Y_r, X_op, Y_op, Wg);
            Wg *= Wm;
          }
        } else {
          for (size_t i = 0; i < gap; i++) {
            if (i != 0) {
              j1 += (m << 1);
            }
            std::cout << "J1 = " << j1 << " gap = " << gap << " step = 1 "
                      << std::endl;
            const std::complex<double> W =
                *scalar * root_of_unity_powers[m + i];
            std::complex<double>* X_r = result + j1;
            std::complex<double>* Y_r = X_r + m;
            *X_r = (*scalar) * (*X_r);
            const std::complex<double>* X_op = X_r;
            const std::complex<double>* Y_op = Y_r;
            int xc = j1;
            int yc = xc + m;
            std::cout << "\t x = " << xc << " y = " << yc << std::endl;
            ComplexFwdButterflyRadix2(X_r, Y_r, X_op, Y_op, Wg);
            Wg *= Wm;
          }
        }
        break;
      }
      default: {
        for (size_t i = 0; i < m; i++) {
          if (i != 0) {
            j1 += (gap << 1);
          }
          const std::complex<double> W = root_of_unity_powers[m + i];
          std::complex<double>* X_r = result + j1;
          std::complex<double>* Y_r = X_r + gap;
          const std::complex<double>* X_op = X_r;
          const std::complex<double>* Y_op = Y_r;
          int xc = j1;
          int yc = xc + gap;
          std::cout << "J1 = " << j1 << " gap = " << gap << " step = 8 "
                    << std::endl;

          HEXL_LOOP_UNROLL_8
          for (size_t j = 0; j < gap; j += 8) {
            std::cout << "\t x = " << xc << " y = " << yc << std::endl;
            ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, Wg);
            xc++;
            yc++;
            std::cout << "\t x = " << xc << " y = " << yc << std::endl;
            ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, Wg);
            xc++;
            yc++;
            std::cout << "\t x = " << xc << " y = " << yc << std::endl;
            ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, Wg);
            xc++;
            yc++;
            std::cout << "\t x = " << xc << " y = " << yc << std::endl;
            ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, Wg);
            xc++;
            yc++;
            std::cout << "\t x = " << xc << " y = " << yc << std::endl;
            ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, Wg);
            xc++;
            yc++;
            std::cout << "\t x = " << xc << " y = " << yc << std::endl;
            ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, Wg);
            xc++;
            yc++;
            std::cout << "\t x = " << xc << " y = " << yc << std::endl;
            ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, Wg);
            xc++;
            yc++;
            std::cout << "\t x = " << xc << " y = " << yc << std::endl;
            ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, Wg);
            xc++;
            yc++;
          }
          Wg *= Wm;
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

  std::complex<double> Wg(1, 0);
  static constexpr double PI_ = 3.1415926535897932384626433832795028842;

  for (size_t i = 0; i < n; ++i) {
    result[i] = operand[ReverseBits(i, 4)];
  }

  uint64_t n_div_2 = (n >> 1);
  size_t gap = 1;
  size_t root_index = 0;

  size_t stop_loop = (scalar == nullptr) ? 0 : 1;
  size_t m = n_div_2;

  std::complex<double> root_of_unity_powers[n];
  const std::complex<double> J(0, 1);
  for (; m > 0; m >>= 1) {
    std::complex<double> wO(1, 0);
    std::complex<double> wmO = std::exp(J * (PI_ / gap));
    for (int j = 0; j < gap; ++j) {
      for (int k = 0; k < m; k++) {
        root_of_unity_powers[root_index] = wO;
        std::cout << root_index << " " << wO << std::endl;
      }
      wO *= wmO;
      root_index++;
    }
    gap <<= 1;
  }

  root_index = 0;
  gap = 1;
  m = n_div_2;

  for (; m > stop_loop; m >>= 1) {
    std::cout << "S = " << gap << " m = " << m << std::endl;
    std::complex<double> Wm = std::exp(J * (PI_ / static_cast<int> gap));
    size_t j1 = 0;

    switch (gap) {
      case 1: {
        std::cout << "Loop J1 = " << j1 << " gap = " << gap << " step = 1 "
                  << std::endl;
        std::complex<double>* W = &root_of_unity_powers[root_index++];
        for (size_t i = 0; i < m; i++) {
          if (i != 0) {
            j1 += (gap << 1);
          }

          std::complex<double>* X_r = result + j1;
          std::complex<double>* Y_r = X_r + gap;
          const std::complex<double>* X_op = result + j1;
          const std::complex<double>* Y_op = X_op + gap;

          int xc = j1;
          int yc = xc + gap;

          std::cout << "\t x[" << xc << "] = " << *X_r << " y[" << yc
                    << "] = " << *Y_r << " Wg[" << root_index << "] = " << *W
                    << std::endl;
          ComplexFwdButterflyRadix2(X_r, Y_r, X_op, Y_op, *W);
        }
        break;
      }
      case 2: {
        std::cout << "Loop J1 = " << j1 << " gap = " << gap << " step = 2 "
                  << std::endl;
        const std::complex<double>* W = &root_of_unity_powers[root_index];
        root_index += 2;
        for (size_t i = 0; i < m; i++) {
          if (i != 0) {
            j1 += (gap << 1);
          }

          std::complex<double>* X_r = result + j1;
          std::complex<double>* Y_r = X_r + gap;
          const std::complex<double>* X_op = X_r;
          const std::complex<double>* Y_op = Y_r;
          int xc = j1;
          int yc = xc + gap;

          std::cout << "\t x[" << xc << "] = " << *X_r << " y[" << yc
                    << "] = " << *Y_r << " Wg[" << root_index << "] = " << *W
                    << std::endl;
          ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, *W);
          xc++;
          yc++;
          std::cout << "\t x[" << xc << "] = " << *X_r << " y[" << yc
                    << "] = " << *Y_r << " Wg[" << root_index
                    << "] = " << *(W + 1) << std::endl;
          ComplexFwdButterflyRadix2(X_r, Y_r, X_op, Y_op, *(W + 1));
        }
        break;
      }
      case 4: {
        std::cout << "Loop J1 = " << j1 << " gap = " << gap << " step = 4 "
                  << std::endl;
        const std::complex<double>* W = &root_of_unity_powers[root_index];
        root_index += 4;
        for (size_t i = 0; i < m; i++) {
          if (i != 0) {
            j1 += (gap << 1);
          }

          std::complex<double>* X_r = result + j1;
          std::complex<double>* Y_r = X_r + gap;
          const std::complex<double>* X_op = X_r;
          const std::complex<double>* Y_op = Y_r;
          int xc = j1;
          int yc = xc + gap;
          std::cout << "\t x[" << xc << "] = " << *X_r << " y[" << yc
                    << "] = " << *Y_r << " Wg[" << root_index << "] = " << *W
                    << std::endl;
          ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, *W);
          xc++;
          yc++;
          std::cout << "\t x[" << xc << "] = " << *X_r << " y[" << yc
                    << "] = " << *Y_r << " Wg[" << root_index
                    << "] = " << *(W + 1) << std::endl;
          ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, *(W + 1));
          xc++;
          yc++;
          std::cout << "\t x[" << xc << "] = " << *X_r << " y[" << yc
                    << "] = " << *Y_r << " Wg[" << root_index
                    << "] = " << *(W + 2) << std::endl;
          ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, *(W + 2));
          xc++;
          yc++;
          std::cout << "\t x[" << xc << "] = " << *X_r << " y[" << yc
                    << "] = " << *Y_r << " Wg[" << root_index
                    << "] = " << *(W + 3) << std::endl;
          ComplexFwdButterflyRadix2(X_r, Y_r, X_op, Y_op, *(W + 3));
        }
        break;
      }
      case 8: {
        std::cout << "Loop J1 = " << j1 << " gap = " << gap << " step = 8 "
                  << std::endl;
        const std::complex<double>* W = &root_of_unity_powers[root_index];
        root_index += 8;
        for (size_t i = 0; i < m; i++) {
          if (i != 0) {
            j1 += (gap << 1);
          }

          std::complex<double>* X_r = result + j1;
          std::complex<double>* Y_r = X_r + gap;
          const std::complex<double>* X_op = X_r;
          const std::complex<double>* Y_op = Y_r;
          int xc = j1;
          int yc = xc + gap;
          std::cout << "\t x[" << xc << "] = " << *X_r << " y[" << yc
                    << "] = " << *Y_r << " Wg[" << root_index << "] = " << *W
                    << std::endl;
          ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, *W++);
          xc++;
          yc++;
          std::cout << "\t x[" << xc << "] = " << *X_r << " y[" << yc
                    << "] = " << *Y_r << " Wg[" << root_index << "] = " << *W
                    << std::endl;
          ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, *W++);
          xc++;
          yc++;
          std::cout << "\t x[" << xc << "] = " << *X_r << " y[" << yc
                    << "] = " << *Y_r << " Wg[" << root_index << "] = " << *W
                    << std::endl;
          ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, *W++);
          xc++;
          yc++;
          std::cout << "\t x[" << xc << "] = " << *X_r << " y[" << yc
                    << "] = " << *Y_r << " Wg[" << root_index << "] = " << *W
                    << std::endl;
          ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, *W++);
          xc++;
          yc++;
          std::cout << "\t x[" << xc << "] = " << *X_r << " y[" << yc
                    << "] = " << *Y_r << " Wg[" << root_index << "] = " << *W
                    << std::endl;
          ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, *W++);
          xc++;
          yc++;
          std::cout << "\t x[" << xc << "] = " << *X_r << " y[" << yc
                    << "] = " << *Y_r << " Wg[" << root_index << "] = " << *W
                    << std::endl;
          ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, *W++);
          xc++;
          yc++;
          std::cout << "\t x[" << xc << "] = " << *X_r << " y[" << yc
                    << "] = " << *Y_r << " Wg[" << root_index << "] = " << *W
                    << std::endl;
          ComplexFwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, *W++);
          xc++;
          yc++;
          std::cout << "\t x[" << xc << "] = " << *X_r << " y[" << yc
                    << "] = " << *Y_r << " Wg[" << root_index << "] = " << *W
                    << std::endl;
          ComplexFwdButterflyRadix2(X_r, Y_r, X_op, Y_op, *W);
        }
        break;
      }
      default: {
        for (size_t i = 0; i < m; i++, root_index++) {
          if (i != 0) {
            j1 += (gap << 1);
          }
          std::cout << "Loop J1 = " << j1 << " gap = " << gap << " step = 8 "
                    << std::endl;
          const std::complex<double> W = inv_root_of_unity_powers[root_index];
          std::complex<double>* X_r = result + j1;
          std::complex<double>* Y_r = X_r + gap;
          const std::complex<double>* X_op = X_r;
          const std::complex<double>* Y_op = Y_r;
          int xc = j1;
          int yc = xc + gap;
          HEXL_LOOP_UNROLL_8
          for (size_t j = 0; j < gap; j += 8) {
            std::cout << "\t x = " << xc << " y = " << yc << " Wg = " << Wg
                      << std::endl;
            ComplexInvButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, Wg);
            xc++;
            yc++;
            std::cout << "\t x = " << xc << " y = " << yc << " Wg = " << Wg
                      << std::endl;
            ComplexInvButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, Wg);
            xc++;
            yc++;
            std::cout << "\t x = " << xc << " y = " << yc << " Wg = " << Wg
                      << std::endl;
            ComplexInvButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, Wg);
            xc++;
            yc++;
            std::cout << "\t x = " << xc << " y = " << yc << " Wg = " << Wg
                      << std::endl;
            ComplexInvButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, Wg);
            xc++;
            yc++;
            std::cout << "\t x = " << xc << " y = " << yc << " Wg = " << Wg
                      << std::endl;
            ComplexInvButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, Wg);
            xc++;
            yc++;
            std::cout << "\t x = " << xc << " y = " << yc << " Wg = " << Wg
                      << std::endl;
            ComplexInvButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, Wg);
            xc++;
            yc++;
            std::cout << "\t x = " << xc << " y = " << yc << " Wg = " << Wg
                      << std::endl;
            ComplexInvButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, Wg);
            xc++;
            yc++;
            std::cout << "\t x = " << xc << " y = " << yc << " Wg = " << Wg
                      << std::endl;
            ComplexInvButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, Wg);
            xc++;
            yc++;
          }
          Wg *= Wm;
        }
      }
    }
    gap <<= 1;
  }

  if (m > 0) {
    std::cout << "S = " << gap << std::endl;
    const std::complex<double> W =
        *scalar * inv_root_of_unity_powers[root_index];
    std::complex<double>* X_r = result;
    std::complex<double>* Y_r = X_r + gap;
    const std::complex<double>* X_o = X_r;
    const std::complex<double>* Y_o = Y_r;
    int xc = 0;
    int yc = xc + gap;
    switch (gap) {
      case 1: {
        std::cout << "J1 = " << j1 << " gap = " << gap << " step = 1 "
                  << std::endl;
        std::cout << "\t x = " << xc << " y = " << yc << std::endl;
        ScaledComplexInvButterflyRadix2(X_r, Y_r, X_o, Y_o, Wg, scalar);
        break;
      }
      case 2: {
        std::cout << "J1 = " << j1 << " gap = " << gap << " step = 2 "
                  << std::endl;
        std::cout << "\t x = " << xc << " y = " << yc << " Wg = " << Wg
                  << std::endl;
        ScaledComplexInvButterflyRadix2(X_r++, Y_r++, X_o++, Y_o++, Wg, scalar);
        xc++;
        yc++;
        std::cout << "\t x = " << xc << " y = " << yc << " Wg = " << Wg
                  << std::endl;
        ScaledComplexInvButterflyRadix2(X_r, Y_r, X_o, Y_o, Wg, scalar);
        break;
      }
      case 4: {
        std::cout << "J1 = " << j1 << " gap = " << gap << " step = 4 "
                  << std::endl;
        std::cout << "\t x = " << xc << " y = " << yc << " Wg = " << Wg
                  << std::endl;
        ScaledComplexInvButterflyRadix2(X_r++, Y_r++, X_o++, Y_o++, Wg, scalar);
        xc++;
        yc++;
        std::cout << "\t x = " << xc << " y = " << yc << " Wg = " << Wg
                  << std::endl;
        ScaledComplexInvButterflyRadix2(X_r++, Y_r++, X_o++, Y_o++, Wg, scalar);
        xc++;
        yc++;
        std::cout << "\t x = " << xc << " y = " << yc << " Wg = " << Wg
                  << std::endl;
        ScaledComplexInvButterflyRadix2(X_r++, Y_r++, X_o++, Y_o++, Wg, scalar);
        xc++;
        yc++;
        std::cout << "\t x = " << xc << " y = " << yc << " Wg = " << Wg
                  << std::endl;
        ScaledComplexInvButterflyRadix2(X_r, Y_r, X_o, Y_o, Wg, scalar);
        break;
      }
      case 8: {
        std::cout << "J1 = " << j1 << " gap = " << gap << " step = 8 "
                  << std::endl;
        std::cout << "\t x = " << xc << " y = " << yc << " Wg = " << Wg
                  << std::endl;
        ScaledComplexInvButterflyRadix2(X_r++, Y_r++, X_o++, Y_o++, Wg, scalar);
        xc++;
        yc++;
        std::cout << "\t x = " << xc << " y = " << yc << " Wg = " << Wg
                  << std::endl;
        ScaledComplexInvButterflyRadix2(X_r++, Y_r++, X_o++, Y_o++, Wg, scalar);
        xc++;
        yc++;
        std::cout << "\t x = " << xc << " y = " << yc << " Wg = " << Wg
                  << std::endl;
        ScaledComplexInvButterflyRadix2(X_r++, Y_r++, X_o++, Y_o++, Wg, scalar);
        xc++;
        yc++;
        std::cout << "\t x = " << xc << " y = " << yc << " Wg = " << Wg
                  << std::endl;
        ScaledComplexInvButterflyRadix2(X_r++, Y_r++, X_o++, Y_o++, Wg, scalar);
        xc++;
        yc++;
        std::cout << "\t x = " << xc << " y = " << yc << " Wg = " << Wg
                  << std::endl;
        ScaledComplexInvButterflyRadix2(X_r++, Y_r++, X_o++, Y_o++, Wg, scalar);
        xc++;
        yc++;
        std::cout << "\t x = " << xc << " y = " << yc << " Wg = " << Wg
                  << std::endl;
        ScaledComplexInvButterflyRadix2(X_r++, Y_r++, X_o++, Y_o++, Wg, scalar);
        xc++;
        yc++;
        std::cout << "\t x = " << xc << " y = " << yc << " Wg = " << Wg
                  << std::endl;
        ScaledComplexInvButterflyRadix2(X_r++, Y_r++, X_o++, Y_o++, Wg, scalar);
        xc++;
        yc++;
        std::cout << "\t x = " << xc << " y = " << yc << " Wg = " << Wg
                  << std::endl;
        ScaledComplexInvButterflyRadix2(X_r, Y_r, X_o, Y_o, Wg, scalar);
        break;
      }
      default: {
        std::cout << "J1 = " << j1 << " gap = " << gap << " step = 8 "
                  << std::endl;
        HEXL_LOOP_UNROLL_8
        for (size_t j = 0; j < gap; j += 8) {
          std::cout << "\t x = " << xc << " y = " << yc << " Wg = " << Wg
                    << std::endl;
          ScaledComplexInvButterflyRadix2(X_r++, Y_r++, X_o++, Y_o++, Wg,
                                          scalar);
          xc++;
          yc++;
          std::cout << "\t x = " << xc << " y = " << yc << " Wg = " << Wg
                    << std::endl;
          ScaledComplexInvButterflyRadix2(X_r++, Y_r++, X_o++, Y_o++, Wg,
                                          scalar);
          xc++;
          yc++;
          std::cout << "\t x = " << xc << " y = " << yc << " Wg = " << Wg
                    << std::endl;
          ScaledComplexInvButterflyRadix2(X_r++, Y_r++, X_o++, Y_o++, Wg,
                                          scalar);
          xc++;
          yc++;
          std::cout << "\t x = " << xc << " y = " << yc << " Wg = " << Wg
                    << std::endl;
          ScaledComplexInvButterflyRadix2(X_r++, Y_r++, X_o++, Y_o++, Wg,
                                          scalar);
          xc++;
          yc++;
          std::cout << "\t x = " << xc << " y = " << yc << " Wg = " << Wg
                    << std::endl;
          ScaledComplexInvButterflyRadix2(X_r++, Y_r++, X_o++, Y_o++, Wg,
                                          scalar);
          xc++;
          yc++;
          std::cout << "\t x = " << xc << " y = " << yc << " Wg = " << Wg
                    << std::endl;
          ScaledComplexInvButterflyRadix2(X_r++, Y_r++, X_o++, Y_o++, Wg,
                                          scalar);
          xc++;
          yc++;
          std::cout << "\t x = " << xc << " y = " << yc << " Wg = " << Wg
                    << std::endl;
          ScaledComplexInvButterflyRadix2(X_r++, Y_r++, X_o++, Y_o++, Wg,
                                          scalar);
          xc++;
          yc++;
          std::cout << "\t x = " << xc << " y = " << yc << " Wg = " << Wg
                    << std::endl;
          ScaledComplexInvButterflyRadix2(X_r++, Y_r++, X_o++, Y_o++, Wg,
                                          scalar);
          xc++;
          yc++;
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
