// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_MATH_FUNCTIONS_HIP_H
#define EIGEN_MATH_FUNCTIONS_HIP_H

namespace Eigen {

namespace internal {

// Make sure this is only available when targeting a GPU: we don't want to
// introduce conflicts between these packet_traits definitions and the ones
// we'll use on the host side (SSE, AVX, ...)
#if defined(__HIPCC__) && defined(EIGEN_USE_GPU)
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
float4 plog(const float4& a)
{
  return make_float4(logf(a.x), logf(a.y), logf(a.z), logf(a.w));
}

EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
double2 plog(const double2& a)
{
  using ::log;
  return make_double2(log(a.x), log(a.y));
}

EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
float4 plog1p(const float4& a)
{
  return make_float4(log1pf(a.x), log1pf(a.y), log1pf(a.z), log1pf(a.w));
}

EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
double2 plog1p(const double2& a)
{
  return make_double2(log1p(a.x), log1p(a.y));
}

EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
float4 pexp(const float4& a)
{
  return make_float4(expf(a.x), expf(a.y), expf(a.z), expf(a.w));
}

EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
double2 pexp(const double2& a)
{
  using ::exp;
  return make_double2(exp(a.x), exp(a.y));
}

EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
float4 psqrt(const float4& a)
{
  return make_float4(sqrtf(a.x), sqrtf(a.y), sqrtf(a.z), sqrtf(a.w));
}

EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
double2 psqrt(const double2& a)
{
  using ::sqrt;
  return make_double2(sqrt(a.x), sqrt(a.y));
}

EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
float4 prsqrt(const float4& a)
{
  return make_float4(rsqrtf(a.x), rsqrtf(a.y), rsqrtf(a.z), rsqrtf(a.w));
}

EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
double2 prsqrt(const double2& a)
{
  return make_double2(rsqrt(a.x), rsqrt(a.y));
}


#endif

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_MATH_FUNCTIONS_HIP_H
