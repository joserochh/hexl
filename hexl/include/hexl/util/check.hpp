// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#include "hexl/util/types.hpp"

// Create logging/debug macros with no run-time overhead unless HEXL_DEBUG is
// enabled
#ifdef HEXL_DEBUG

#define HEXL_CHECK(cond, expr) \
  {}
#define HEXL_CHECK_BOUNDS(...) \
  {}

#else  // HEXL_DEBUG=OFF

#define HEXL_CHECK(cond, expr) \
  {}
#define HEXL_CHECK_BOUNDS(...) \
  {}

#endif  // HEXL_DEBUG
