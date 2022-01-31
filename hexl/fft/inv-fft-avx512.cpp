// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "hexl/fft/inv-fft-avx512.hpp"

#include "hexl/fft/fft-avx512-util.hpp"
#include "hexl/logging/logging.hpp"

namespace intel {
namespace hexl {

#ifdef HEXL_HAS_AVX512DQ

/// @brief Final butterfly step for the Inverse FFT.
/// @param[in,out] X_real Double precision (DP) values in SIMD form representing
/// the real part of 8 complex numbers.
/// @param[in,out] X_imag DP values in SIMD form representing the
/// imaginary part of the forementioned complex numbers.
/// @param[in,out] Y_real DP values in SIMD form representing the
/// real part of 8 complex numbers.
/// @param[in,out] Y_imag DP values in SIMD form representing the
/// imaginary part of the forementioned complex numbers.
/// @param[in] W_real DP values in SIMD form representing the real part of the
/// Inverse Complex Roots of unity.
/// @param[in] W_imag DP values in SIMD form representing the imaginary part of
/// the Inverse Complex Roots of unity.
void ComplexInvButterfly(__m512d* X_real, __m512d* X_imag, __m512d* Y_real,
                         __m512d* Y_imag, __m512d W_real, __m512d W_imag,
                         const double_t* scalar = nullptr) {
  // U = X,
  __m512d U_real = *X_real;
  __m512d U_imag = *X_imag;

  // X = U + Y
  *X_real = _mm512_add_pd(U_real, *Y_real);
  *X_imag = _mm512_add_pd(U_imag, *Y_imag);

  if (scalar != nullptr) {
    __m512d v_scalar = _mm512_set1_pd(*scalar);
    *X_real = _mm512_mul_pd(*X_real, v_scalar);
    *X_imag = _mm512_mul_pd(*X_imag, v_scalar);
  }

  // V = U - Y
  __m512d V_real = _mm512_sub_pd(U_real, *Y_real);
  __m512d V_imag = _mm512_sub_pd(U_imag, *Y_imag);

  // Y = V*W. Complex multiplication:
  // (v_r + iv_b)*(w_a + iw_b) = (v_a*w_a - v_b*w_b) + i(v_a*w_b + v_b*w_a)
  *Y_real = _mm512_mul_pd(V_real, W_real);
  __m512d tmp = _mm512_mul_pd(V_imag, W_imag);
  *Y_real = _mm512_sub_pd(*Y_real, tmp);

  *Y_imag = _mm512_mul_pd(V_real, W_imag);
  tmp = _mm512_mul_pd(V_imag, W_real);
  *Y_imag = _mm512_add_pd(*Y_imag, tmp);
}

void ComplexInvT1(double_t* result_8C_intrlvd,
                  const double_t* operand_1C_intrlvd,
                  const double_t* W_1C_intrlvd, uint64_t m) {
  size_t offset = 0;

  // 8 | m guaranteed by n >= 16
  HEXL_LOOP_UNROLL_4

  for (size_t i = 0; i < (m >> 1); i += 8) {
    // Referencing operand
    const double_t* X_op_real = operand_1C_intrlvd + offset;

    // Referencing result
    double_t* X_r_real = result_8C_intrlvd + offset;
    double_t* X_r_imag = result_8C_intrlvd + 8 + offset;
    __m512d* v_X_r_pt_real = reinterpret_cast<__m512d*>(X_r_real);
    __m512d* v_X_r_pt_imag = reinterpret_cast<__m512d*>(X_r_imag);

    __m512d v_X_real;
    __m512d v_X_imag;
    __m512d v_Y_real;
    __m512d v_Y_imag;

    ComplexLoadInvInterleavedT1(X_op_real, &v_X_real, &v_X_imag, &v_Y_real,
                                &v_Y_imag);

    // Weights
    // x =  (14r, 10r, 6r, 2r, 12r, 8r, 4r, 0r);
    // y =  (15r, 11r, 7r, 3r, 13r, 9r, 5r, 1r);
    __m512d v_W_real = _mm512_set_pd(
        W_1C_intrlvd[14], W_1C_intrlvd[10], W_1C_intrlvd[6], W_1C_intrlvd[2],
        W_1C_intrlvd[12], W_1C_intrlvd[8], W_1C_intrlvd[4], W_1C_intrlvd[0]);
    __m512d v_W_imag = _mm512_set_pd(
        W_1C_intrlvd[15], W_1C_intrlvd[11], W_1C_intrlvd[7], W_1C_intrlvd[3],
        W_1C_intrlvd[13], W_1C_intrlvd[9], W_1C_intrlvd[5], W_1C_intrlvd[1]);
    W_1C_intrlvd += 16;

    ComplexInvButterfly(&v_X_real, &v_X_imag, &v_Y_real, &v_Y_imag, v_W_real,
                        v_W_imag);

    _mm512_storeu_pd(v_X_r_pt_real, v_X_real);
    _mm512_storeu_pd(v_X_r_pt_imag, v_X_imag);
    v_X_r_pt_real += 2;
    v_X_r_pt_imag += 2;
    _mm512_storeu_pd(v_X_r_pt_real, v_Y_real);
    _mm512_storeu_pd(v_X_r_pt_imag, v_Y_imag);

    offset += 32;
  }
}

void ComplexInvT2(double_t* operand_8C_intrlvd, const double_t* W_1C_intrlvd,
                  uint64_t m) {
  size_t offset = 0;

  // 4 | m guaranteed by n >= 16
  HEXL_LOOP_UNROLL_4
  for (size_t i = 0; i < (m >> 1); i += 4) {
    double_t* X_real = operand_8C_intrlvd + offset;
    double_t* X_imag = operand_8C_intrlvd + 8 + offset;

    __m512d* v_X_pt_real = reinterpret_cast<__m512d*>(X_real);
    __m512d* v_X_pt_imag = reinterpret_cast<__m512d*>(X_imag);

    __m512d v_X_real;
    __m512d v_X_imag;

    __m512d v_Y_real;
    __m512d v_Y_imag;

    ComplexLoadInvInterleavedT2(X_real, &v_X_real, &v_Y_real);
    ComplexLoadInvInterleavedT2(X_imag, &v_X_imag, &v_Y_imag);

    // Weights
    // x =  (13,  9, 5, 1, 12,  8, 4, 0)
    // y =  (15, 11, 7, 3, 14, 10, 6, 2)
    __m512d v_W_real = _mm512_set_pd(
        W_1C_intrlvd[6], W_1C_intrlvd[4], W_1C_intrlvd[2], W_1C_intrlvd[0],
        W_1C_intrlvd[6], W_1C_intrlvd[4], W_1C_intrlvd[2], W_1C_intrlvd[0]);
    __m512d v_W_imag = _mm512_set_pd(
        W_1C_intrlvd[7], W_1C_intrlvd[5], W_1C_intrlvd[3], W_1C_intrlvd[1],
        W_1C_intrlvd[7], W_1C_intrlvd[5], W_1C_intrlvd[3], W_1C_intrlvd[1]);
    W_1C_intrlvd += 8;

    ComplexInvButterfly(&v_X_real, &v_X_imag, &v_Y_real, &v_Y_imag, v_W_real,
                        v_W_imag);

    _mm512_storeu_pd(v_X_pt_real, v_X_real);
    _mm512_storeu_pd(v_X_pt_imag, v_X_imag);
    v_X_pt_real += 2;
    v_X_pt_imag += 2;
    _mm512_storeu_pd(v_X_pt_real, v_Y_real);
    _mm512_storeu_pd(v_X_pt_imag, v_Y_imag);

    offset += 32;
  }
}

void ComplexInvT4(double_t* operand_8C_intrlvd, const double_t* W_1C_intrlvd,
                  uint64_t m) {
  size_t offset = 0;

  // 2 | m guaranteed by n >= 16
  HEXL_LOOP_UNROLL_4
  for (size_t i = 0; i < (m >> 1); i += 2) {
    double_t* X_real = operand_8C_intrlvd + offset;
    double_t* X_imag = operand_8C_intrlvd + 8 + offset;

    __m512d* v_X_pt_real = reinterpret_cast<__m512d*>(X_real);
    __m512d* v_X_pt_imag = reinterpret_cast<__m512d*>(X_imag);

    __m512d v_X_real;
    __m512d v_X_imag;
    __m512d v_Y_real;
    __m512d v_Y_imag;

    ComplexLoadInvInterleavedT4(X_real, &v_X_real, &v_Y_real);
    ComplexLoadInvInterleavedT4(X_imag, &v_X_imag, &v_Y_imag);

    // Weights
    // x =  (11,  9, 3, 1, 10,  8, 2, 0)
    // y =  (15, 13, 7, 5, 14, 12, 6, 4)
    __m512d v_W_real = _mm512_set_pd(
        W_1C_intrlvd[2], W_1C_intrlvd[2], W_1C_intrlvd[0], W_1C_intrlvd[0],
        W_1C_intrlvd[2], W_1C_intrlvd[2], W_1C_intrlvd[0], W_1C_intrlvd[0]);
    __m512d v_W_imag = _mm512_set_pd(
        W_1C_intrlvd[3], W_1C_intrlvd[3], W_1C_intrlvd[1], W_1C_intrlvd[1],
        W_1C_intrlvd[3], W_1C_intrlvd[3], W_1C_intrlvd[1], W_1C_intrlvd[1]);

    W_1C_intrlvd += 4;

    ComplexInvButterfly(&v_X_real, &v_X_imag, &v_Y_real, &v_Y_imag, v_W_real,
                        v_W_imag);

    ComplexWriteInvInterleavedT4(v_X_real, v_Y_real, v_X_pt_real);
    ComplexWriteInvInterleavedT4(v_X_imag, v_Y_imag, v_X_pt_imag);

    offset += 32;
  }
}

void ComplexInvT8(double_t* operand_8C_intrlvd, const double_t* W_1C_intrlvd,
                  uint64_t gap, uint64_t m) {
  size_t offset = 0;

  HEXL_LOOP_UNROLL_4
  for (size_t i = 0; i < (m >> 1); i++) {
    // Referencing operand
    double_t* X_real = operand_8C_intrlvd + offset;
    double_t* X_imag = operand_8C_intrlvd + 8 + offset;

    double_t* Y_real = X_real + gap;
    double_t* Y_imag = X_imag + gap;

    __m512d* v_X_pt_real = reinterpret_cast<__m512d*>(X_real);
    __m512d* v_X_pt_imag = reinterpret_cast<__m512d*>(X_imag);

    __m512d* v_Y_pt_real = reinterpret_cast<__m512d*>(Y_real);
    __m512d* v_Y_pt_imag = reinterpret_cast<__m512d*>(Y_imag);

    // Weights
    __m512d v_W_real = _mm512_set1_pd(*W_1C_intrlvd++);
    __m512d v_W_imag = _mm512_set1_pd(*W_1C_intrlvd++);

    // assume 8 | t
    for (size_t j = 0; j < gap; j += 16) {
      __m512d v_X_real = _mm512_loadu_pd(v_X_pt_real);
      __m512d v_X_imag = _mm512_loadu_pd(v_X_pt_imag);

      __m512d v_Y_real = _mm512_loadu_pd(v_Y_pt_real);
      __m512d v_Y_imag = _mm512_loadu_pd(v_Y_pt_imag);

      ComplexInvButterfly(&v_X_real, &v_X_imag, &v_Y_real, &v_Y_imag, v_W_real,
                          v_W_imag);

      _mm512_storeu_pd(v_X_pt_real, v_X_real);
      _mm512_storeu_pd(v_X_pt_imag, v_X_imag);

      _mm512_storeu_pd(v_Y_pt_real, v_Y_real);
      _mm512_storeu_pd(v_Y_pt_imag, v_Y_imag);

      // Increase operand & result pointers
      v_X_pt_real += 2;
      v_X_pt_imag += 2;
      v_Y_pt_real += 2;
      v_Y_pt_imag += 2;
    }
    offset += (gap << 1);
  }
}

// Takes operand as 8 complex interleaved: This is 8 real parts followed by
// its 8 imaginary parts.
// Returns operand as 1 complex interleaved: One real part followed by its
// imaginary part.
void ComplexFinalInvT8(double_t* operand_8C_intrlvd,
                       const double_t* W_1C_intrlvd, uint64_t gap, uint64_t m,
                       const double_t* scalar = nullptr) {
  size_t offset = 0;

  __m512d v_scalar;
  if (scalar != nullptr) {
    v_scalar = _mm512_set1_pd(*scalar);
  }

  HEXL_LOOP_UNROLL_4
  for (size_t i = 0; i < (m >> 1); i++) {
    // Referencing operand
    double_t* X_real = operand_8C_intrlvd + offset;
    double_t* X_imag = operand_8C_intrlvd + 8 + offset;

    double_t* Y_real = X_real + gap;
    double_t* Y_imag = X_imag + gap;

    __m512d* v_X_pt_real = reinterpret_cast<__m512d*>(X_real);
    __m512d* v_X_pt_imag = reinterpret_cast<__m512d*>(X_imag);

    __m512d* v_Y_pt_real = reinterpret_cast<__m512d*>(Y_real);
    __m512d* v_Y_pt_imag = reinterpret_cast<__m512d*>(Y_imag);

    // Weights
    __m512d v_W_real = _mm512_set1_pd(*W_1C_intrlvd++);
    __m512d v_W_imag = _mm512_set1_pd(*W_1C_intrlvd++);

    if (scalar != nullptr) {
      v_W_real = _mm512_mul_pd(v_W_real, v_scalar);
      v_W_imag = _mm512_mul_pd(v_W_imag, v_scalar);
    }

    // assume 8 | t
    for (size_t j = 0; j < gap; j += 16) {
      __m512d v_X_real = _mm512_loadu_pd(v_X_pt_real);
      __m512d v_X_imag = _mm512_loadu_pd(v_X_pt_imag);
      __m512d v_Y_real = _mm512_loadu_pd(v_Y_pt_real);
      __m512d v_Y_imag = _mm512_loadu_pd(v_Y_pt_imag);

      ComplexInvButterfly(&v_X_real, &v_X_imag, &v_Y_real, &v_Y_imag, v_W_real,
                          v_W_imag, scalar);

      ComplexWriteInvInterleavedT8(&v_X_real, &v_X_imag, &v_Y_real, &v_Y_imag,
                                   v_X_pt_real, v_Y_pt_real);

      // Increase operand & result pointers
      v_X_pt_real += 2;
      v_X_pt_imag += 2;
      v_Y_pt_real += 2;
      v_Y_pt_imag += 2;
    }
    offset += (gap << 1);
  }
}

void Inverse_FFT_FromBitReverseAVX512(
    double_t* result_cmplx_intrlvd, const double_t* operand_cmplx_intrlvd,
    const double_t* inv_root_of_unity_cmplx_intrlvd, const uint64_t n,
    const double_t* scale) {
  HEXL_CHECK(IsPowerOfTwo(n), "n " << n << " is not a power of 2");
  HEXL_CHECK(n > 2, "n " << n << " is not bigger than 2");

  size_t gap = 2;    // Interleaved complex values requires twice the size
  size_t m = n;      // (2*n >> 1);
  size_t W_idx = 2;  // 2*1

  {
    // T1
    const double_t* W_cmplx_intrlvd = &inv_root_of_unity_cmplx_intrlvd[W_idx];
    ComplexInvT1(result_cmplx_intrlvd, operand_cmplx_intrlvd, W_cmplx_intrlvd,
                 m);
    W_idx += m;
    gap <<= 1;
    m >>= 1;

    // T2
    W_cmplx_intrlvd = &inv_root_of_unity_cmplx_intrlvd[W_idx];
    ComplexInvT2(result_cmplx_intrlvd, W_cmplx_intrlvd, m);
    W_idx += m;
    gap <<= 1;
    m >>= 1;

    // T4
    W_cmplx_intrlvd = &inv_root_of_unity_cmplx_intrlvd[W_idx];
    ComplexInvT4(result_cmplx_intrlvd, W_cmplx_intrlvd, m);
    W_idx += m;
    gap <<= 1;
    m >>= 1;

    for (; m > 2;) {
      W_cmplx_intrlvd = &inv_root_of_unity_cmplx_intrlvd[W_idx];
      ComplexInvT8(result_cmplx_intrlvd, W_cmplx_intrlvd, gap, m);
      W_idx += m;
      gap <<= 1;
      m >>= 1;
    }

    W_cmplx_intrlvd = &inv_root_of_unity_cmplx_intrlvd[W_idx];
    ComplexFinalInvT8(result_cmplx_intrlvd, W_cmplx_intrlvd, gap, m, scale);
    W_idx += m;
    gap <<= 1;
    m >>= 1;
  }

  HEXL_VLOG(5, "AVX512 returning INV FFT result " << std::vector<double_t>(
                   result_cmplx_intrlvd, result_cmplx_intrlvd + n));
}

#endif  // HEXL_HAS_AVX512DQ

}  // namespace hexl
}  // namespace intel