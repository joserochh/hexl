// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cstring>
#include <iostream>
#include <memory>
#include <utility>

#include "hexl/logging/logging.hpp"
#include "hexl/ntt/ntt.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "hexl/util/aligned-allocator.hpp"
#include "hexl/util/check.hpp"
#include "ntt/ntt-default.hpp"
#include "util/cpu-features.hpp"

namespace intel {
namespace hexl {

void ForwardTransformToBitReverseRadix4(
    uint64_t* operand, uint64_t n, uint64_t modulus,
    const uint64_t* root_of_unity_powers,
    const uint64_t* precon_root_of_unity_powers, uint64_t input_mod_factor,
    uint64_t output_mod_factor) {
  HEXL_CHECK(NTT::CheckArguments(n, modulus), "");
  HEXL_CHECK_BOUNDS(operand, n, modulus * input_mod_factor,
                    "operand exceeds bound " << modulus * input_mod_factor);
  HEXL_CHECK(root_of_unity_powers != nullptr,
             "root_of_unity_powers == nullptr");
  HEXL_CHECK(precon_root_of_unity_powers != nullptr,
             "precon_root_of_unity_powers == nullptr");
  HEXL_CHECK(
      input_mod_factor == 1 || input_mod_factor == 2 || input_mod_factor == 4,
      "input_mod_factor must be 1, 2, or 4; got " << input_mod_factor);
  HEXL_UNUSED(input_mod_factor);
  HEXL_CHECK(output_mod_factor == 1 || output_mod_factor == 4,
             "output_mod_factor must be 1 or 4; got " << output_mod_factor);

  HEXL_VLOG(3, "modulus " << modulus);
  HEXL_VLOG(3, "n " << n);

  HEXL_VLOG(3, "operand " << std::vector<uint64_t>(operand, operand + n));

  HEXL_VLOG(3, "root_of_unity_powers " << std::vector<uint64_t>(
                   root_of_unity_powers, root_of_unity_powers + n));

  uint64_t root_of_unity = root_of_unity_powers[1];

  uint64_t ldn = Log2(n);
  HEXL_VLOG(3, "ldn " << ldn);

  bool is_power_of_4 = IsPowerOfFour(n);

  LOG(INFO) << "radix 4";
  // auto n = n;
  // ReverseVectorBits(operand, n);
  // HEXL_VLOG(3, "bit-reversed inputs "
  //                  << std::vector<uint64_t>(operand, operand + n));

  // Radix-2 step for non-powers of 4
  // if (!is_power_of_4) {
  //   HEXL_VLOG(3, "Radix 2 step");
  //   // TODO(fboemer): more efficient
  //   for (size_t i = 0; i < n; i += 2) {
  //     // sumdiff(f[i], f[i + 1]);
  //     uint64_t idx1 = ReverseBitsUInt(i, ldn);
  //     uint64_t idx2 = ReverseBitsUInt(i + 1, ldn);

  //     uint64_t sum = AddUIntMod(operand[idx1], operand[idx2], modulus);
  //     uint64_t diff = SubUIntMod(operand[idx1], operand[idx2], modulus);

  //     HEXL_VLOG(3, "loaded operand[" << idx1 << "] = " << operand[idx1]);
  //     HEXL_VLOG(3, "loaded operand[" << idx2 << "] = " << operand[idx2]);

  //     operand[idx1] = sum;
  //     operand[idx2] = diff;

  //     HEXL_VLOG(3, "wrote operand[" << idx1 << "] = " << operand[idx1]);
  //     HEXL_VLOG(3, "wrote operand[" << idx2 << "] = " << operand[idx2]);
  //   }
  // }
  // HEXL_VLOG(3, "after radix 2 outputs "
  //              << std::vector<uint64_t>(operand, operand + n));

  /*
  size_t t = (n >> 1);
for (size_t m = 1; m < n; m <<= 1) {
HEXL_VLOG(3, "m " << m);
size_t j1 = 0;
for (size_t i = 0; i < m; i++) {
HEXL_VLOG(3, "i " << i);
size_t j2 = j1 + t;
const uint64_t W_op = root_of_unity_powers[m + i];
*/

  for (size_t ldm = ldn; ldm >= 2; ldm -= 2) {
    // for (size_t ldm = 2 + size_t(!is_power_of_4); ldm <= ldn; ldm += 2) {
    size_t m = 1UL << ldm;
    size_t m4 = m >> 2;  // 4;
    HEXL_VLOG(3, "m " << m);
    HEXL_VLOG(3, "m4 " << m4);

    // uint64_t W_op1 = 1;
    // uint64_t W_op2 = 1;
    // uint64_t W_op3 = 1;
    // uint64_t dw = 18;

    uint64_t imag = root_of_unity_powers[1];
    HEXL_VLOG(3, "imag " << imag);

    for (size_t j = 0; j < m4; j++) {
      HEXL_VLOG(3, "j " << j);
      // LOG(INFO) << "r1 " << r1;

#pragma GCC unroll 4
#pragma clang loop unroll_count(4)
      for (size_t r = 0; r < n; r += m) {
        HEXL_VLOG(3, "r " << r);

        // 4-point NTT butterfly

        uint64_t X0_ind = r + j;
        uint64_t X1_ind = X0_ind + m4;
        uint64_t X2_ind = X0_ind + 2 * m4;
        uint64_t X3_ind = X0_ind + 3 * m4;

        const uint64_t W_op0 = root_of_unity_powers[X0_ind];
        const uint64_t W_op1 = root_of_unity_powers[X1_ind];
        const uint64_t W_op2 = root_of_unity_powers[X2_ind];
        const uint64_t W_op3 = root_of_unity_powers[X3_ind];
        HEXL_VLOG(
            3, "W_ops " << (std::vector<uint64_t>{W_op0, W_op1, W_op2, W_op3}));

        HEXL_VLOG(3, "Xinds " << (std::vector<uint64_t>{X0_ind, X1_ind, X2_ind,
                                                        X3_ind}));

        HEXL_VLOG(3, "Xs " << (std::vector<uint64_t>{
                         operand[X0_ind], operand[X1_ind], operand[X2_ind],
                         operand[X3_ind]}));

        uint64_t X0 = operand[X0_ind];
        uint64_t X1 = operand[X1_ind];
        uint64_t X2 = operand[X2_ind];
        uint64_t X3 = operand[X3_ind];

        // continue;

        uint64_t a0 = operand[X0_ind];
        uint64_t a1 = MultiplyMod(operand[X1_ind], W_op1, modulus);
        uint64_t a2 = MultiplyMod(operand[X2_ind], W_op2, modulus);
        uint64_t a3 = MultiplyMod(operand[X3_ind], W_op3, modulus);

        HEXL_VLOG(3, "as " << (std::vector<uint64_t>{a0, a1, a2, a3}));

        uint64_t W1_x_x2 = MultiplyMod(W_op1, X2, modulus);
        uint64_t tmp0 = AddUIntMod(X0, W1_x_x2, modulus);
        uint64_t tmp2 = SubUIntMod(X0, W1_x_x2, modulus);
        HEXL_VLOG(3, "W1_x_x2 " << W1_x_x2);
        HEXL_VLOG(3, "tmp0 " << tmp0);
        HEXL_VLOG(3, "tmp2 " << tmp2);

        uint64_t W1_x_x3 = MultiplyMod(W_op1, X3, modulus);
        HEXL_VLOG(3, "W1_x_x4 " << W1_x_x3);

        uint64_t tmp1 = AddUIntMod(X1, W1_x_x3, modulus);
        uint64_t tmp3 = SubUIntMod(X1, W1_x_x3, modulus);
        HEXL_VLOG(3, "tmp1 " << tmp1);
        HEXL_VLOG(3, "tmp3 " << tmp3);

        uint64_t tmp1_x_W2 = MultiplyMod(tmp1, W_op2, modulus);
        uint64_t Y0 = AddUIntMod(tmp0, tmp1_x_W2, modulus);
        uint64_t Y1 = SubUIntMod(tmp0, tmp1_x_W2, modulus);

        uint64_t tmp3_x_W2 = MultiplyMod(tmp3, W_op3, modulus);
        uint64_t Y2 = AddUIntMod(tmp2, tmp3_x_W2, modulus);
        uint64_t Y3 = SubUIntMod(tmp2, tmp3_x_W2, modulus);

        HEXL_VLOG(3, "Ys " << (std::vector<uint64_t>{Y0, Y1, Y2, Y3}));

        operand[X0_ind] = Y0;
        operand[X1_ind] = Y1;
        operand[X2_ind] = Y2;
        operand[X3_ind] = Y3;
      }

      HEXL_VLOG(3, "inner Intermediate values "
                       << std::vector<uint64_t>(operand, operand + n));

      // W_op1 = (W_op1 * dw) % mod;
      // W_op2 = (W_op1 * W_op1) % mod;
      // W_op3 = (W_op1 * W_op2) % mod;
    }
    HEXL_VLOG(3, "outer Intermediate values "
                     << std::vector<uint64_t>(operand, operand + n));
  }

  HEXL_VLOG(3, "outputs " << std::vector<uint64_t>(operand, operand + n));
}

}  // namespace hexl
}  // namespace intel
