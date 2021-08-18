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

  uint64_t log_n = Log2(n);
  HEXL_VLOG(3, "log_n " << log_n);

  bool is_power_of_4 = IsPowerOfFour(n);

  LOG(INFO) << "radix 4";
  // auto n = n;
  // ReverseVectorBits(operand, n);
  // HEXL_VLOG(3, "bit-reversed inputs "
  //                  << std::vector<uint64_t>(operand, operand + n));

  // Radix-2 step for non-powers of 4
  if (!is_power_of_4) {
    HEXL_VLOG(3, "Radix 2 step");
    uint64_t twice_modulus = modulus << 1;
    size_t t = (n >> 1);
    size_t root_index = 1;

    const uint64_t W = root_of_unity_powers[1];
    const uint64_t W_precon = precon_root_of_unity_powers[1];

    uint64_t* X = operand;
    uint64_t* Y = X + t;
    for (size_t j = 0; j < t; j++) {
      FwdButterfly(X++, Y++, W, W_precon, modulus, twice_modulus);
    }
    // Data in [0, 4q)
  }

  HEXL_VLOG(3, "after radix 2 outputs "
                   << std::vector<uint64_t>(operand, operand + n));

  uint64_t w_idx = is_power_of_4 ? 1 : 2;
  size_t t = (n >> w_idx) >> 1;
  for (size_t m = is_power_of_4 ? 1 : 2; m < n; m <<= 2) {
    HEXL_VLOG(3, "m " << m);

    size_t gap = n >> Log2(m);
    gap >>= 2;
    HEXL_VLOG(3, "gap " << gap);

    for (size_t i = 0; i < m; i++) {
      HEXL_VLOG(3, "i " << i << " up to " << m);

      HEXL_VLOG(3, "t " << t);

      for (size_t j = 0; j < t; j++) {
        HEXL_VLOG(3, "j " << j << " up to " << t);
        HEXL_VLOG(3, "j1 " << j1);

        // 4-point NTT butterfly
        uint64_t X0_ind = j + 4 * i * t;
        uint64_t X1_ind = X0_ind + gap;
        uint64_t X2_ind = X0_ind + 2 * gap;
        uint64_t X3_ind = X0_ind + 3 * gap;

        HEXL_VLOG(3, "Xinds " << (std::vector<uint64_t>{X0_ind, X1_ind, X2_ind,
                                                        X3_ind}));

        uint64_t W1_ind = m + i;
        uint64_t W2_ind = 2 * W1_ind;
        uint64_t W3_ind = 2 * W1_ind + 1;
        ++w_idx;

        HEXL_VLOG(3,
                  "Winds " << (std::vector<uint64_t>{W1_ind, W2_ind, W3_ind}));

        const uint64_t W1 = root_of_unity_powers[W1_ind];
        const uint64_t W2 = root_of_unity_powers[W2_ind];
        const uint64_t W3 = root_of_unity_powers[W3_ind];

        uint64_t X0 = operand[X0_ind] % modulus;
        uint64_t X1 = operand[X1_ind] % modulus;
        uint64_t X2 = operand[X2_ind] % modulus;
        uint64_t X3 = operand[X3_ind] % modulus;

        FwdButterflyRadix4(&X0, &X1, &X2, &X3, W1, W2, W3, modulus,
                           2 * modulus);

        operand[X0_ind] = X0;
        operand[X1_ind] = X1;
        operand[X2_ind] = X2;
        operand[X3_ind] = X3;
      }
      w_idx = w_idx * 2;
      // HEXL_VLOG(3, "inner Intermediate values "
      //                  << std::vector<uint64_t>(operand, operand + n));
    }
    t >>= 2;
    // HEXL_VLOG(3, "outer Intermediate values "
    //                  << std::vector<uint64_t>(operand, operand + n));
  }

  uint64_t twice_modulus = modulus * 2;

  if (output_mod_factor == 1) {
    for (size_t i = 0; i < n; ++i) {
      if (operand[i] >= twice_modulus) {
        operand[i] -= twice_modulus;
      }
      if (operand[i] >= modulus) {
        operand[i] -= modulus;
      }
      HEXL_CHECK(operand[i] < modulus, "Incorrect modulus reduction in NTT "
                                           << operand[i] << " >= " << modulus);
    }
  }

  HEXL_VLOG(3, "outputs " << std::vector<uint64_t>(operand, operand + n));
}

}  // namespace hexl
}  // namespace intel
