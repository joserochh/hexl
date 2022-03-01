// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <complex>
#include <iostream>

#include "hexl/fft/fft-native.hpp"
#include "hexl/fft/fft.hpp"
#include "hexl/logging/logging.hpp"
#include "hexl/util/defines.hpp"
#include "test-util.hpp"
#include "util/cpu-features.hpp"
#include "util/util-internal.hpp"
namespace intel {
namespace hexl {

TEST(FFT, ForwardFFTNative) {
  /*
    {  // Single Unscaled
      const uint64_t n = 64;

      FFT fft(n, nullptr);
      AlignedVector64<std::complex<double>> root_powers =
        fft.GetComplexRootsOfUnity();

      const double data_bound = (1 << 30);
      AlignedVector64<std::complex<double>> operand(n);
      AlignedVector64<std::complex<double>> result(n);

      operand[0] = std::complex<double>(
          GenerateInsecureUniformRealRandomValue(0, data_bound),
          GenerateInsecureUniformRealRandomValue(0, data_bound));

      Forward_FFT_ToBitReverseRadix2(result.data(), operand.data(),
                                     root_powers.data(), n);

      for (size_t i = 0; i < n; ++i) {
        double tmp = abs(operand[0].real() - result[i].real());
        ASSERT_TRUE(tmp < 0.5);
        tmp = abs(operand[0].imag() - result[i].imag());
        ASSERT_TRUE(tmp < 0.5);
      }
    }

    {  // Single Scaled
      const uint64_t n = 64;

      FFT fft(n, nullptr);
      AlignedVector64<std::complex<double>> root_powers =
        fft.GetInvComplexRootsOfUnity();

      const double scale = 1 << 16;
      const double inv_scale = static_cast<double>(1.0) / scale;
      const double data_bound = (1 << 30);
      AlignedVector64<std::complex<double>> operand(n);
      AlignedVector64<std::complex<double>> result(n);

      std::complex<double> value(
          GenerateInsecureUniformRealRandomValue(0, data_bound),
          GenerateInsecureUniformRealRandomValue(0, data_bound));
      operand[0] = value;
      value *= inv_scale;

      Forward_FFT_ToBitReverseRadix2(result.data(), operand.data(),
                                     root_powers.data(), n, &inv_scale);

      for (size_t i = 0; i < n; ++i) {
        double tmp = abs(value.real() - result[i].real());
        ASSERT_TRUE(tmp < 0.5);
        tmp = abs(value.imag() - result[i].imag());
        ASSERT_TRUE(tmp < 0.5);
      }
    }*/

  {
    const uint64_t n = 16;
    FFT fft(n, nullptr);
    AlignedVector64<std::complex<double>> root_powers =
        fft.GetComplexRootsOfUnity();

    std::vector<std::complex<double>> operand = {
        8.0,  10.0, 10.0, 8.0,  10.0, 10.0, 10.0, 10.0,
        10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0};
    std::vector<std::complex<double>> expected = {
        {73, 73},
        {-6.2032150945452065, -4.2416445337387456},
        {-13.135501078993816, -10.251661790966025},
        {-4.9522280719094312, -6.1940103592659312},
        {-13.989360298368769, -20.936557238542498},
        {-12.041504662418394, -11.538103686627512},
        {3.1145318688906216, -6.1377091711734213},
        {2.8190362866168104, 0.94252895584742813},
        {0.78569495838710224, -1.1758756024193586},
        {-2.3937707912728081, 16.926460686272904},
        {-8.2973196713709836, 13.235852212450261},
        {3.9512417817615444, -11.907983525397228},
        {-17.116634736998645, 17.116634736998645},
        {-2.4155730294210045, 5.7101840294886106},
        {0.55814605451906485, 3.8269354264526543},
        {6.4603733686324603, -14.136903384391978}};

    AlignedVector64<std::complex<double>> result(n);

    Fwd_FFT(result.data(), operand.data(), root_powers.data(), n);

    for (size_t i = 0; i < n; ++i) {
      std::cout.precision(17);
      std::cout << "{" << result[i].real() << "," << result[i].imag() << "},"
                << std::endl;
    }

    Inv_FFT(result.data(), result.data(), root_powers.data(), n);
    for (size_t i = 0; i < n; ++i) {
      std::cout.precision(17);
      std::cout << "{" << result[i].real() << "," << result[i].imag() << "},"
                << std::endl;
      double tmp = abs(expected[i].real() - result[i].real());
      // ASSERT_TRUE(tmp < 0.5);
      tmp = abs(expected[i].imag() - result[i].imag());
      // ASSERT_TRUE(tmp < 0.5);
    }
  }
}

TEST(FFT, ForwardInverseFFTNative) {
  const uint64_t n = 64;

  FFT fft(n, nullptr);
  AlignedVector64<std::complex<double>> root_powers =
      fft.GetComplexRootsOfUnity();
  AlignedVector64<std::complex<double>> inv_root_powers =
      fft.GetInvComplexRootsOfUnity();

  {  // Zeros test
    const double scale = 1 << 16;
    const double scalar = scale / static_cast<double>(n);
    const double inv_scale = static_cast<double>(1.0) / scale;

    AlignedVector64<std::complex<double>> operand(n, {0, 0});
    AlignedVector64<std::complex<double>> transformed(n);
    AlignedVector64<std::complex<double>> result(n);

    Inverse_FFT_FromBitReverseRadix2(transformed.data(), operand.data(),
                                     inv_root_powers.data(), n, &scalar);

    Forward_FFT_ToBitReverseRadix2(result.data(), transformed.data(),
                                   root_powers.data(), n, &inv_scale);

    for (size_t i = 0; i < n; ++i) {
      auto tmp = abs(operand[i].real() - result[i].real());
      ASSERT_TRUE(tmp < 0.5);
      tmp = abs(operand[i].imag() - result[i].imag());
      ASSERT_TRUE(tmp < 0.5);
    }
  }

  {                                      // Large Scaled
    const double scale = 1099511627776;  // (1 << 40)
    const double scalar = scale / static_cast<double>(n);
    const double inv_scale = static_cast<double>(1.0) / scale;
    const double data_bound = (1 << 30);

    AlignedVector64<std::complex<double>> operand(n);
    AlignedVector64<std::complex<double>> transformed(n);
    AlignedVector64<std::complex<double>> result(n);

    for (size_t i = 0; i < n; i++) {
      operand[i] = std::complex<double>(
          GenerateInsecureUniformRealRandomValue(0, data_bound),
          GenerateInsecureUniformRealRandomValue(0, data_bound));
    }

    Inverse_FFT_FromBitReverseRadix2(transformed.data(), operand.data(),
                                     inv_root_powers.data(), n, &scalar);

    Forward_FFT_ToBitReverseRadix2(result.data(), transformed.data(),
                                   root_powers.data(), n, &inv_scale);

    for (size_t i = 0; i < n; ++i) {
      double tmp = abs(operand[i].real() - result[i].real());
      ASSERT_TRUE(tmp < 0.5);
      tmp = abs(operand[i].imag() - result[i].imag());
      ASSERT_TRUE(tmp < 0.5);
    }
  }

  {                                               // Very Large Scale
    const double scale = 1.2980742146337069e+33;  // (1 << 110)
    const double scalar = scale / static_cast<double>(n);
    const double inv_scale = static_cast<double>(1.0) / scale;
    const double data_bound = (1 << 20);

    AlignedVector64<std::complex<double>> operand(n);
    AlignedVector64<std::complex<double>> transformed(n);
    AlignedVector64<std::complex<double>> result(n);

    for (size_t i = 0; i < n; i++) {
      operand[i] = std::complex<double>(
          GenerateInsecureUniformRealRandomValue(0, data_bound),
          GenerateInsecureUniformRealRandomValue(0, data_bound));
    }

    AlignedVector64<std::complex<double>> expected = operand;

    Inverse_FFT_FromBitReverseRadix2(transformed.data(), operand.data(),
                                     inv_root_powers.data(), n, &scalar);

    Forward_FFT_ToBitReverseRadix2(result.data(), transformed.data(),
                                   root_powers.data(), n, &inv_scale);

    for (size_t i = 0; i < n; ++i) {
      double tmp = abs(expected[i].real() - result[i].real());
      ASSERT_TRUE(tmp < 0.5);
      tmp = abs(expected[i].imag() - result[i].imag());
      ASSERT_TRUE(tmp < 0.5);
    }
  }

  {                                               // Over 128 bits Scale
    const double scale = 1.3611294676837539e+39;  // (1 << 130)
    const double scalar = scale / static_cast<double>(n);
    const double inv_scale = static_cast<double>(1.0) / scale;
    const double data_bound = (1 << 20);

    AlignedVector64<std::complex<double>> operand(n);
    AlignedVector64<std::complex<double>> transformed(n);
    AlignedVector64<std::complex<double>> result(n);

    for (size_t i = 0; i < n; i++) {
      operand[i] = std::complex<double>(
          GenerateInsecureUniformRealRandomValue(0, data_bound),
          GenerateInsecureUniformRealRandomValue(0, data_bound));
    }

    AlignedVector64<std::complex<double>> expected = operand;

    Inverse_FFT_FromBitReverseRadix2(transformed.data(), operand.data(),
                                     inv_root_powers.data(), n, &scalar);

    Forward_FFT_ToBitReverseRadix2(result.data(), transformed.data(),
                                   root_powers.data(), n, &inv_scale);

    for (size_t i = 0; i < n; ++i) {
      double tmp = abs(expected[i].real() - result[i].real());
      ASSERT_TRUE(tmp < 0.5);
      tmp = abs(expected[i].imag() - result[i].imag());
      ASSERT_TRUE(tmp < 0.5);
    }
  }

  {                                               // Inplace
    const double scale = 1.3611294676837539e+39;  // (1 << 130)
    const double scalar = scale / static_cast<double>(n);
    const double inv_scale = static_cast<double>(1.0) / scale;
    const double data_bound = (1 << 20);

    AlignedVector64<std::complex<double>> operand(n);

    for (size_t i = 0; i < n; i++) {
      operand[i] = std::complex<double>(
          GenerateInsecureUniformRealRandomValue(0, data_bound),
          GenerateInsecureUniformRealRandomValue(0, data_bound));
    }

    AlignedVector64<std::complex<double>> expected = operand;

    Inverse_FFT_FromBitReverseRadix2(operand.data(), operand.data(),
                                     inv_root_powers.data(), n, &scalar);

    Forward_FFT_ToBitReverseRadix2(operand.data(), operand.data(),
                                   root_powers.data(), n, &inv_scale);

    for (size_t i = 0; i < n; ++i) {
      double tmp = abs(expected[i].real() - operand[i].real());
      ASSERT_TRUE(tmp < 0.5);
      tmp = abs(expected[i].imag() - operand[i].imag());
      ASSERT_TRUE(tmp < 0.5);
    }
  }
}

}  // namespace hexl
}  // namespace intel
