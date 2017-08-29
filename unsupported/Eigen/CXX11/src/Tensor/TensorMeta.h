// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2015 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_META_H
#define EIGEN_CXX11_TENSOR_TENSOR_META_H

// Custom serializers / deserializers for Eigen::array with
// Eigen::IndexPair<int> or std::pair<int, int> as elements
#ifdef __HCC__
#include "../util/EmulateArray.h"
#endif

namespace Eigen {

template<bool cond> struct Cond {};

template<typename T1, typename T2> EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
const T1& choose(Cond<true>, const T1& first, const T2&) {
  return first;
}

template<typename T1, typename T2> EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
const T2& choose(Cond<false>, const T1&, const T2& second) {
  return second;
}


template <typename T, typename X, typename Y>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
T divup(const X x, const Y y) {
  return static_cast<T>((x + y - 1) / y);
}

template <typename T>
EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
T divup(const T x, const T y) {
  return static_cast<T>((x + y - 1) / y);
}

template <size_t n> struct max_n_1 {
  static const size_t size = n;
};
template <> struct max_n_1<0> {
  static const size_t size = 1;
};


// Default packet types
template <typename Scalar, typename Device>
struct PacketType : internal::packet_traits<Scalar> {
  typedef typename internal::packet_traits<Scalar>::type type;
};

// For HIP packet types when using a GpuDevice
#if defined(EIGEN_USE_GPU) && defined(__HIPCC__) && defined(EIGEN_HAS_HIP_FP16)
template <>
struct PacketType<half, GpuDevice> {
  typedef half2 type;
  static const int size = 2;
  enum {
    HasAdd    = 1,
    HasSub    = 1,
    HasMul    = 1,
    HasNegate = 1,
    HasAbs    = 1,
    HasArg    = 0,
    HasAbs2   = 0,
    HasMin    = 1,
    HasMax    = 1,
    HasConj   = 0,
    HasSetLinear = 0,
    HasBlend  = 0,

    HasDiv    = 1,
    HasSqrt   = 1,
    HasRsqrt  = 1,
    HasExp    = 1,
    HasExpm1  = 0,
    HasLog    = 1,
    HasLog1p  = 0,
    HasLog10  = 0,
    HasPow    = 1,
  };
};
#endif

#if defined(EIGEN_USE_SYCL)
template <typename T>
  struct PacketType<T, SyclDevice> {
  typedef T type;
  static const int size = 1;
  enum {
    HasAdd    = 0,
    HasSub    = 0,
    HasMul    = 0,
    HasNegate = 0,
    HasAbs    = 0,
    HasArg    = 0,
    HasAbs2   = 0,
    HasMin    = 0,
    HasMax    = 0,
    HasConj   = 0,
    HasSetLinear = 0,
    HasBlend  = 0
  };
};
#endif


// Tuple mimics std::pair but works on e.g. hipcc.
template <typename U, typename V> struct Tuple {
 public:
  U first;
  V second;

  typedef U first_type;
  typedef V second_type;

  EIGEN_CONSTEXPR EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  Tuple() : first(), second() {}

  EIGEN_CONSTEXPR EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  Tuple(const U& f, const V& s) : first(f), second(s) {}

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  Tuple& operator= (const Tuple& rhs) {
    if (&rhs == this) return *this;
    first = rhs.first;
    second = rhs.second;
    return *this;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  void swap(Tuple& rhs) {
    using numext::swap;
    swap(first, rhs.first);
    swap(second, rhs.second);
  }
};

template <typename U, typename V>
EIGEN_CONSTEXPR EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
bool operator==(const Tuple<U, V>& x, const Tuple<U, V>& y) {
  return (x.first == y.first && x.second == y.second);
}

template <typename U, typename V>
EIGEN_CONSTEXPR EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
bool operator!=(const Tuple<U, V>& x, const Tuple<U, V>& y) {
  return !(x == y);
}


// Can't use std::pairs on hip devices
template <typename Idx> struct IndexPair {
  EIGEN_CONSTEXPR EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE IndexPair() : first(0), second(0) {}
  EIGEN_CONSTEXPR EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE IndexPair(Idx f, Idx s) : first(f), second(s) {}

  EIGEN_DEVICE_FUNC void set(IndexPair<Idx> val) {
    first = val.first;
    second = val.second;
  }

  Idx first;
  Idx second;
};
 

// Custom serializers / deserializers for Eigen::array with
// Eigen::IndexPair<int> or std::pair<int, int> as elements
#ifdef __HCC__
// Specialize array for size 1 and T = Eigen::IndexPair<int>
template<>
class array<Eigen::IndexPair<int>, 1> {
 public:
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Eigen::IndexPair<int>& operator[] (size_t index) { return values[index]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const Eigen::IndexPair<int>& operator[] (size_t index) const { return values[index]; }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Eigen::IndexPair<int>& front() { return values[0]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const Eigen::IndexPair<int>& front() const { return values[0]; }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Eigen::IndexPair<int>& back() { return values[0]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const Eigen::IndexPair<int>& back() const { return values[0]; }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  static std::size_t size() { return 1; }

  Eigen::IndexPair<int> values[1];

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE array() { }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE ~array() { }

#if EIGEN_HAS_VARIADIC_TEMPLATES
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE array(std::initializer_list<Eigen::IndexPair<int> > l) {
    eigen_assert(l.size() == 1);
    internal::smart_copy(l.begin(), l.end(), values);
  }
#endif

  __attribute__((annotate("user_deserialize")))
  array(int first, int second) [[cpu]][[hc]] {
    values[0] = Eigen::IndexPair<int>(first, second);
  }
  __attribute__((annotate("serialize")))
  void __cxxamp_serialize(Kalmar::Serialize &s) const {
    s.Append(sizeof(int), &values[0].first);
    s.Append(sizeof(int), &values[0].second);
  }

  array(Eigen::IndexPair<int> v0) {
    values[0].set(v0);
  }
};

// Specialize array for size 1 and T = Eigen::IndexPair<long>
template<>
class array<Eigen::IndexPair<long>, 1> {
 public:
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Eigen::IndexPair<long>& operator[] (size_t index) { return values[index]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const Eigen::IndexPair<long>& operator[] (size_t index) const { return values[index]; }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Eigen::IndexPair<long>& front() { return values[0]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const Eigen::IndexPair<long>& front() const { return values[0]; }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Eigen::IndexPair<long>& back() { return values[0]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const Eigen::IndexPair<long>& back() const { return values[0]; }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  static std::size_t size() { return 1; }

  Eigen::IndexPair<long> values[1];

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE array() { }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE ~array() { }

#if EIGEN_HAS_VARIADIC_TEMPLATES
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE array(std::initializer_list<Eigen::IndexPair<long> > l) {
    eigen_assert(l.size() == 1);
    internal::smart_copy(l.begin(), l.end(), values);
  }
#endif

  __attribute__((annotate("user_deserialize")))
  array(long first, long second) [[cpu]][[hc]] {
    values[0] = Eigen::IndexPair<long>(first, second);
  }
  __attribute__((annotate("serialize")))
  void __cxxamp_serialize(Kalmar::Serialize &s) const {
    s.Append(sizeof(long), &values[0].first);
    s.Append(sizeof(long), &values[0].second);
  }

  array(Eigen::IndexPair<long> v0) {
    values[0].set(v0);
  }
};

// Specialize array for size 1 and T = std::pair<int, int>
template<>
class array<std::pair<int, int>, 1> {
 public:
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE std::pair<int, int>& operator[] (size_t index) { return values[index]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const std::pair<int, int>& operator[] (size_t index) const { return values[index]; }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE std::pair<int, int> front() { return values[0]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const std::pair<int, int>& front() const { return values[0]; }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE std::pair<int, int>& back() { return values[0]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const std::pair<int, int>& back() const { return values[0]; }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  static std::size_t size() { return 1; }

  std::pair<int, int> values[1];

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE array() { }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE ~array() { }

#if EIGEN_HAS_VARIADIC_TEMPLATES
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE array(std::initializer_list<std::pair<int, int> > l) {
    eigen_assert(l.size() == 1);
    internal::smart_copy(l.begin(), l.end(), values);
  }
#endif

  __attribute__((annotate("user_deserialize")))
  array(int first, int second) [[cpu]][[hc]] {
    values[0] = std::pair<int, int>(first, second);
  }
  __attribute__((annotate("serialize")))
  void __cxxamp_serialize(Kalmar::Serialize &s) const {
    s.Append(sizeof(int), &values[0].first);
    s.Append(sizeof(int), &values[0].second);
  }

  array(std::pair<int, int> v0) {
    values[0] = v0;
  }
};

// Specialize array for size 2 and T = Eigen::IndexPair<int>
template <>
class array<Eigen::IndexPair<int>, 2> {
 public:
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Eigen::IndexPair<int>& operator[] (size_t index) { return values[index]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const Eigen::IndexPair<int>& operator[] (size_t index) const { return values[index]; }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Eigen::IndexPair<int>& front() { return values[0]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const Eigen::IndexPair<int>& front() const { return values[0]; }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Eigen::IndexPair<int>& back() { return values[1]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const Eigen::IndexPair<int>& back() const { return values[1]; }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  static std::size_t size() { return 2; }

  Eigen::IndexPair<int> values[2];

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE array() { }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE ~array() { }

#if EIGEN_HAS_VARIADIC_TEMPLATES
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE array(std::initializer_list<Eigen::IndexPair<int>> l) {
    eigen_assert(l.size() == 2);
    internal::smart_copy(l.begin(), l.end(), values);
  }
#endif

  __attribute__((annotate("user_deserialize")))
  array(int v0_first, int v0_second,
        int v1_first, int v1_second) [[cpu]][[hc]] {
    values[0] = Eigen::IndexPair<int>(v0_first, v0_second);
    values[1] = Eigen::IndexPair<int>(v1_first, v1_second);
  }

  __attribute__((annotate("serialize")))
  void __cxxamp_serialize(Kalmar::Serialize &s) const {
    s.Append(sizeof(int), &values[0].first);
    s.Append(sizeof(int), &values[0].second);
    s.Append(sizeof(int), &values[1].first);
    s.Append(sizeof(int), &values[1].second);
  }

  array(Eigen::IndexPair<int> v0,
        Eigen::IndexPair<int> v1) {
    values[0].set(v0);
    values[1].set(v1);
  }
};

// Specialize array for size 2 and T = Eigen::IndexPair<long>
template <>
class array<Eigen::IndexPair<long>, 2> {
 public:
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Eigen::IndexPair<long>& operator[] (size_t index) { return values[index]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const Eigen::IndexPair<long>& operator[] (size_t index) const { return values[index]; }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Eigen::IndexPair<long>& front() { return values[0]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const Eigen::IndexPair<long>& front() const { return values[0]; }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Eigen::IndexPair<long>& back() { return values[1]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const Eigen::IndexPair<long>& back() const { return values[1]; }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  static std::size_t size() { return 2; }

  Eigen::IndexPair<long> values[2];

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE array() { }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE ~array() { }

#if EIGEN_HAS_VARIADIC_TEMPLATES
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE array(std::initializer_list<Eigen::IndexPair<long>> l) {
    eigen_assert(l.size() == 2);
    internal::smart_copy(l.begin(), l.end(), values);
  }
#endif

  __attribute__((annotate("user_deserialize")))
  array(long v0_first, long v0_second,
        long v1_first, long v1_second) [[cpu]][[hc]] {
    values[0] = Eigen::IndexPair<long>(v0_first, v0_second);
    values[1] = Eigen::IndexPair<long>(v1_first, v1_second);
  }

  __attribute__((annotate("serialize")))
  void __cxxamp_serialize(Kalmar::Serialize &s) const {
    s.Append(sizeof(long), &values[0].first);
    s.Append(sizeof(long), &values[0].second);
    s.Append(sizeof(long), &values[1].first);
    s.Append(sizeof(long), &values[1].second);
  }

  array(Eigen::IndexPair<long> v0,
        Eigen::IndexPair<long> v1) {
    values[0].set(v0);
    values[1].set(v1);
  }
};

// Specialize array for size 2 and T = std::pair<int, int>
template<>
class array<std::pair<int, int>, 2> {
 public:
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE std::pair<int, int>& operator[] (size_t index) { return values[index]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const std::pair<int, int>& operator[] (size_t index) const { return values[index]; }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE std::pair<int, int> front() { return values[0]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const std::pair<int, int>& front() const { return values[0]; }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE std::pair<int, int>& back() { return values[1]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const std::pair<int, int>& back() const { return values[1]; }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  static std::size_t size() { return 2; }

  std::pair<int, int> values[2];

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE array() { }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE ~array() { }

#if EIGEN_HAS_VARIADIC_TEMPLATES
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE array(std::initializer_list<std::pair<int, int> > l) {
    eigen_assert(l.size() == 2);
    internal::smart_copy(l.begin(), l.end(), values);
  }
#endif

  __attribute__((annotate("user_deserialize")))
  array(int v0_first, int v0_second,
        int v1_first, int v1_second) [[cpu]][[hc]] {
    values[0] = std::pair<int, int>(v0_first, v0_second);
    values[1] = std::pair<int, int>(v1_first, v1_second);
  }
  __attribute__((annotate("serialize")))
  void __cxxamp_serialize(Kalmar::Serialize &s) const {
    s.Append(sizeof(int), &values[0].first);
    s.Append(sizeof(int), &values[0].second);
    s.Append(sizeof(int), &values[1].first);
    s.Append(sizeof(int), &values[1].second);
  }

  array(std::pair<int, int> v0,
        std::pair<int, int> v1) {
    values[0] = v0;
    values[1] = v1;
  }
};

// Specialize array for size 3 and T = Eigen::IndexPair<int>
template <>
class array<Eigen::IndexPair<int>, 3> {
 public:
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Eigen::IndexPair<int>& operator[] (size_t index) { return values[index]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const Eigen::IndexPair<int>& operator[] (size_t index) const { return values[index]; }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Eigen::IndexPair<int>& front() { return values[0]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const Eigen::IndexPair<int>& front() const { return values[0]; }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Eigen::IndexPair<int>& back() { return values[2]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const Eigen::IndexPair<int>& back() const { return values[2]; }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  static std::size_t size() { return 3; }

  Eigen::IndexPair<int> values[3];

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE array() { }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE ~array() { }

#if EIGEN_HAS_VARIADIC_TEMPLATES
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE array(std::initializer_list<Eigen::IndexPair<int> > l) {
    eigen_assert(l.size() == 3);
    internal::smart_copy(l.begin(), l.end(), values);
  }
#endif

  __attribute__((annotate("user_deserialize")))
  array(int v0_first, int v0_second,
        int v1_first, int v1_second,
        int v2_first, int v2_second) [[cpu]][[hc]] {
    values[0] = Eigen::IndexPair<int>(v0_first, v0_second);
    values[1] = Eigen::IndexPair<int>(v1_first, v1_second);
    values[2] = Eigen::IndexPair<int>(v2_first, v2_second);
  }

  __attribute__((annotate("serialize")))
  void __cxxamp_serialize(Kalmar::Serialize &s) const {
    for (int i = 0; i < 3; ++i) {
      s.Append(sizeof(int), &values[i].first);
      s.Append(sizeof(int), &values[i].second);
    }
  }

  array(Eigen::IndexPair<int> v0,
        Eigen::IndexPair<int> v1,
        Eigen::IndexPair<int> v2) {
    values[0].set(v0);
    values[1].set(v1);
    values[2].set(v2);
  }
};

// Specialize array for size 3 and T = Eigen::IndexPair<long>
template <>
class array<Eigen::IndexPair<long>, 3> {
 public:
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Eigen::IndexPair<long>& operator[] (size_t index) { return values[index]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const Eigen::IndexPair<long>& operator[] (size_t index) const { return values[index]; }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Eigen::IndexPair<long>& front() { return values[0]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const Eigen::IndexPair<long>& front() const { return values[0]; }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Eigen::IndexPair<long>& back() { return values[2]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const Eigen::IndexPair<long>& back() const { return values[2]; }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  static std::size_t size() { return 3; }

  Eigen::IndexPair<long> values[3];

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE array() { }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE ~array() { }

#if EIGEN_HAS_VARIADIC_TEMPLATES
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE array(std::initializer_list<Eigen::IndexPair<long> > l) {
    eigen_assert(l.size() == 3);
    internal::smart_copy(l.begin(), l.end(), values);
  }
#endif

  __attribute__((annotate("user_deserialize")))
  array(long v0_first, long v0_second,
        long v1_first, long v1_second,
        long v2_first, long v2_second) [[cpu]][[hc]] {
    values[0] = Eigen::IndexPair<long>(v0_first, v0_second);
    values[1] = Eigen::IndexPair<long>(v1_first, v1_second);
    values[2] = Eigen::IndexPair<long>(v2_first, v2_second);
  }

  __attribute__((annotate("serialize")))
  void __cxxamp_serialize(Kalmar::Serialize &s) const {
    for (int i = 0; i < 3; ++i) {
      s.Append(sizeof(long), &values[i].first);
      s.Append(sizeof(long), &values[i].second);
    }
  }

  array(Eigen::IndexPair<long> v0,
        Eigen::IndexPair<long> v1,
        Eigen::IndexPair<long> v2) {
    values[0].set(v0);
    values[1].set(v1);
    values[2].set(v2);
  }
};

// Specialize array for size 3 and T = std::pair<int, int>
template<>
class array<std::pair<int, int>, 3> {
 public:
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE std::pair<int, int>& operator[] (size_t index) { return values[index]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const std::pair<int, int>& operator[] (size_t index) const { return values[index]; }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE std::pair<int, int> front() { return values[0]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const std::pair<int, int>& front() const { return values[0]; }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE std::pair<int, int>& back() { return values[2]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const std::pair<int, int>& back() const { return values[2]; }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  static std::size_t size() { return 3; }

  std::pair<int, int> values[3];

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE array() { }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE ~array() { }

#if EIGEN_HAS_VARIADIC_TEMPLATES
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE array(std::initializer_list<std::pair<int, int> > l) {
    eigen_assert(l.size() == 3);
    internal::smart_copy(l.begin(), l.end(), values);
  }
#endif

  __attribute__((annotate("user_deserialize")))
  array(int v0_first, int v0_second,
        int v1_first, int v1_second,
        int v2_first, int v2_second) [[cpu]][[hc]] {
    values[0] = std::pair<int, int>(v0_first, v0_second);
    values[1] = std::pair<int, int>(v1_first, v1_second);
    values[2] = std::pair<int, int>(v2_first, v2_second);
  }
  __attribute__((annotate("serialize")))
  void __cxxamp_serialize(Kalmar::Serialize &s) const {
    for (int i = 0; i < 3; ++i) {
      s.Append(sizeof(int), &values[i].first);
      s.Append(sizeof(int), &values[i].second);
    }
  }

  array(std::pair<int, int> v0,
        std::pair<int, int> v1,
        std::pair<int, int> v2) {
    values[0] = v0;
    values[1] = v1;
    values[2] = v2;
  }
};

// Specialize array for size 4 and T = Eigen::IndexPair<int>
template <>
class array<Eigen::IndexPair<int>, 4> {
 public:
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Eigen::IndexPair<int>& operator[] (size_t index) { return values[index]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const Eigen::IndexPair<int>& operator[] (size_t index) const { return values[index]; }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Eigen::IndexPair<int>& front() { return values[0]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const Eigen::IndexPair<int>& front() const { return values[0]; }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Eigen::IndexPair<int>& back() { return values[3]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const Eigen::IndexPair<int>& back() const { return values[3]; }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  static std::size_t size() { return 4; }

  Eigen::IndexPair<int> values[4];

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE array() { }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE ~array() { }

#if EIGEN_HAS_VARIADIC_TEMPLATES
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE array(std::initializer_list<Eigen::IndexPair<int> > l) {
    eigen_assert(l.size() == 4);
    internal::smart_copy(l.begin(), l.end(), values);
  }
#endif

  __attribute__((annotate("user_deserialize")))
  array(int v0_first, int v0_second,
        int v1_first, int v1_second,
        int v2_first, int v2_second,
        int v3_first, int v3_second) [[cpu]][[hc]] {
    values[0] = Eigen::IndexPair<int>(v0_first, v0_second);
    values[1] = Eigen::IndexPair<int>(v1_first, v1_second);
    values[2] = Eigen::IndexPair<int>(v2_first, v2_second);
    values[3] = Eigen::IndexPair<int>(v3_first, v3_second);
  }

  __attribute__((annotate("serialize")))
  void __cxxamp_serialize(Kalmar::Serialize &s) const {
    for (int i = 0; i < 4; ++i) {
      s.Append(sizeof(int), &values[i].first);
      s.Append(sizeof(int), &values[i].second);
    }
  }

  array(Eigen::IndexPair<int> v0,
        Eigen::IndexPair<int> v1,
        Eigen::IndexPair<int> v2,
        Eigen::IndexPair<int> v3) {
    values[0].set(v0);
    values[1].set(v1);
    values[2].set(v2);
    values[3].set(v3);
  }
};

// Specialize array for size 4 and T = Eigen::IndexPair<long>
template <>
class array<Eigen::IndexPair<long>, 4> {
 public:
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Eigen::IndexPair<long>& operator[] (size_t index) { return values[index]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const Eigen::IndexPair<long>& operator[] (size_t index) const { return values[index]; }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Eigen::IndexPair<long>& front() { return values[0]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const Eigen::IndexPair<long>& front() const { return values[0]; }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Eigen::IndexPair<long>& back() { return values[3]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const Eigen::IndexPair<long>& back() const { return values[3]; }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  static std::size_t size() { return 4; }

  Eigen::IndexPair<long> values[4];

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE array() { }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE ~array() { }

#if EIGEN_HAS_VARIADIC_TEMPLATES
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE array(std::initializer_list<Eigen::IndexPair<long> > l) {
    eigen_assert(l.size() == 4);
    internal::smart_copy(l.begin(), l.end(), values);
  }
#endif

  __attribute__((annotate("user_deserialize")))
  array(long v0_first, long v0_second,
        long v1_first, long v1_second,
        long v2_first, long v2_second,
        long v3_first, long v3_second) [[cpu]][[hc]] {
    values[0] = Eigen::IndexPair<long>(v0_first, v0_second);
    values[1] = Eigen::IndexPair<long>(v1_first, v1_second);
    values[2] = Eigen::IndexPair<long>(v2_first, v2_second);
    values[3] = Eigen::IndexPair<long>(v3_first, v3_second);
  }

  __attribute__((annotate("serialize")))
  void __cxxamp_serialize(Kalmar::Serialize &s) const {
    for (int i = 0; i < 4; ++i) {
      s.Append(sizeof(long), &values[i].first);
      s.Append(sizeof(long), &values[i].second);
    }
  }

  array(Eigen::IndexPair<long> v0,
        Eigen::IndexPair<long> v1,
        Eigen::IndexPair<long> v2,
        Eigen::IndexPair<long> v3) {
    values[0].set(v0);
    values[1].set(v1);
    values[2].set(v2);
    values[3].set(v3);
  }
};

// Specialize array for size 4 and T = std::pair<int, int>
template<>
class array<std::pair<int, int>, 4> {
 public:
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE std::pair<int, int>& operator[] (size_t index) { return values[index]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const std::pair<int, int>& operator[] (size_t index) const { return values[index]; }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE std::pair<int, int> front() { return values[0]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const std::pair<int, int>& front() const { return values[0]; }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE std::pair<int, int>& back() { return values[3]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const std::pair<int, int>& back() const { return values[3]; }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  static std::size_t size() { return 4; }

  std::pair<int, int> values[4];

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE array() { }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE ~array() { }

#if EIGEN_HAS_VARIADIC_TEMPLATES
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE array(std::initializer_list<std::pair<int, int> > l) {
    eigen_assert(l.size() == 4);
    internal::smart_copy(l.begin(), l.end(), values);
  }
#endif

  __attribute__((annotate("user_deserialize")))
  array(int v0_first, int v0_second,
        int v1_first, int v1_second,
        int v2_first, int v2_second,
        int v3_first, int v3_second) [[cpu]][[hc]] {
    values[0] = std::pair<int, int>(v0_first, v0_second);
    values[1] = std::pair<int, int>(v1_first, v1_second);
    values[2] = std::pair<int, int>(v2_first, v2_second);
    values[3] = std::pair<int, int>(v3_first, v3_second);
  }
  __attribute__((annotate("serialize")))
  void __cxxamp_serialize(Kalmar::Serialize &s) const {
    for (int i = 0; i < 4; ++i) {
      s.Append(sizeof(int), &values[i].first);
      s.Append(sizeof(int), &values[i].second);
    }
  }

  array(std::pair<int, int> v0,
        std::pair<int, int> v1,
        std::pair<int, int> v2,
        std::pair<int, int> v3) {
    values[0] = v0;
    values[1] = v1;
    values[2] = v2;
    values[3] = v3;
  }
};

// Specialize array for size 5 and T = Eigen::IndexPair<int>
template <>
class array<Eigen::IndexPair<int>, 5> {
 public:
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Eigen::IndexPair<int>& operator[] (size_t index) { return values[index]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const Eigen::IndexPair<int>& operator[] (size_t index) const { return values[index]; }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Eigen::IndexPair<int>& front() { return values[0]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const Eigen::IndexPair<int>& front() const { return values[0]; }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Eigen::IndexPair<int>& back() { return values[4]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const Eigen::IndexPair<int>& back() const { return values[4]; }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  static std::size_t size() { return 5; }

  Eigen::IndexPair<int> values[5];

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE array() { }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE ~array() { }

#if EIGEN_HAS_VARIADIC_TEMPLATES
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE array(std::initializer_list<Eigen::IndexPair<int> > l) {
    eigen_assert(l.size() == 5);
    internal::smart_copy(l.begin(), l.end(), values);
  }
#endif

  __attribute__((annotate("user_deserialize")))
  array(int v0_first, int v0_second,
        int v1_first, int v1_second,
        int v2_first, int v2_second,
        int v3_first, int v3_second,
        int v4_first, int v4_second) [[cpu]][[hc]] {
    values[0] = Eigen::IndexPair<int>(v0_first, v0_second);
    values[1] = Eigen::IndexPair<int>(v1_first, v1_second);
    values[2] = Eigen::IndexPair<int>(v2_first, v2_second);
    values[3] = Eigen::IndexPair<int>(v3_first, v3_second);
    values[4] = Eigen::IndexPair<int>(v4_first, v4_second);
  }

  __attribute__((annotate("serialize")))
  void __cxxamp_serialize(Kalmar::Serialize &s) const {
    for (int i = 0; i < 5; ++i) {
      s.Append(sizeof(int), &values[i].first);
      s.Append(sizeof(int), &values[i].second);
    }
  }

  array(Eigen::IndexPair<int> v0,
        Eigen::IndexPair<int> v1,
        Eigen::IndexPair<int> v2,
        Eigen::IndexPair<int> v3,
        Eigen::IndexPair<int> v4) {
    values[0].set(v0);
    values[1].set(v1);
    values[2].set(v2);
    values[3].set(v3);
    values[4].set(v4);
  }
};
 
// Specialize array for size 5 and T = Eigen::IndexPair<long>
template <>
class array<Eigen::IndexPair<long>, 5> {
 public:
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Eigen::IndexPair<long>& operator[] (size_t index) { return values[index]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const Eigen::IndexPair<long>& operator[] (size_t index) const { return values[index]; }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Eigen::IndexPair<long>& front() { return values[0]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const Eigen::IndexPair<long>& front() const { return values[0]; }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Eigen::IndexPair<long>& back() { return values[4]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const Eigen::IndexPair<long>& back() const { return values[4]; }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  static std::size_t size() { return 5; }

  Eigen::IndexPair<long> values[5];

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE array() { }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE ~array() { }

#if EIGEN_HAS_VARIADIC_TEMPLATES
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE array(std::initializer_list<Eigen::IndexPair<long> > l) {
    eigen_assert(l.size() == 5);
    internal::smart_copy(l.begin(), l.end(), values);
  }
#endif

  __attribute__((annotate("user_deserialize")))
  array(long v0_first, long v0_second,
        long v1_first, long v1_second,
        long v2_first, long v2_second,
        long v3_first, long v3_second,
        long v4_first, long v4_second) [[cpu]][[hc]] {
    values[0] = Eigen::IndexPair<long>(v0_first, v0_second);
    values[1] = Eigen::IndexPair<long>(v1_first, v1_second);
    values[2] = Eigen::IndexPair<long>(v2_first, v2_second);
    values[3] = Eigen::IndexPair<long>(v3_first, v3_second);
    values[4] = Eigen::IndexPair<long>(v4_first, v4_second);
  }

  __attribute__((annotate("serialize")))
  void __cxxamp_serialize(Kalmar::Serialize &s) const {
    for (int i = 0; i < 5; ++i) {
      s.Append(sizeof(long), &values[i].first);
      s.Append(sizeof(long), &values[i].second);
    }
  }

  array(Eigen::IndexPair<long> v0,
        Eigen::IndexPair<long> v1,
        Eigen::IndexPair<long> v2,
        Eigen::IndexPair<long> v3,
        Eigen::IndexPair<long> v4) {
    values[0].set(v0);
    values[1].set(v1);
    values[2].set(v2);
    values[3].set(v3);
    values[4].set(v4);
  }
};

// Specialize array for size 5 and T = std::pair<int, int>
template<>
class array<std::pair<int, int>, 5> {
 public:
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE std::pair<int, int>& operator[] (size_t index) { return values[index]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const std::pair<int, int>& operator[] (size_t index) const { return values[index]; }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE std::pair<int, int> front() { return values[0]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const std::pair<int, int>& front() const { return values[0]; }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE std::pair<int, int>& back() { return values[4]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const std::pair<int, int>& back() const { return values[4]; }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  static std::size_t size() { return 5; }

  std::pair<int, int> values[5];

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE array() { }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE ~array() { }

#if EIGEN_HAS_VARIADIC_TEMPLATES
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE array(std::initializer_list<std::pair<int, int> > l) {
    eigen_assert(l.size() == 5);
    internal::smart_copy(l.begin(), l.end(), values);
  }
#endif

  __attribute__((annotate("user_deserialize")))
  array(int v0_first, int v0_second,
        int v1_first, int v1_second,
        int v2_first, int v2_second,
        int v3_first, int v3_second,
        int v4_first, int v4_second) [[cpu]][[hc]] {
    values[0] = std::pair<int, int>(v0_first, v0_second);
    values[1] = std::pair<int, int>(v1_first, v1_second);
    values[2] = std::pair<int, int>(v2_first, v2_second);
    values[3] = std::pair<int, int>(v3_first, v3_second);
    values[4] = std::pair<int, int>(v4_first, v4_second);
  }
  __attribute__((annotate("serialize")))
  void __cxxamp_serialize(Kalmar::Serialize &s) const {
    for (int i = 0; i < 5; ++i) {
      s.Append(sizeof(int), &values[i].first);
      s.Append(sizeof(int), &values[i].second);
    }
  }

  array(std::pair<int, int> v0,
        std::pair<int, int> v1,
        std::pair<int, int> v2,
        std::pair<int, int> v3,
        std::pair<int, int> v4) {
    values[0] = v0;
    values[1] = v1;
    values[2] = v2;
    values[3] = v3;
    values[4] = v4;
  }
};

// Specialize array for size 6 and T = Eigen::IndexPair<int>
template <>
class array<Eigen::IndexPair<int>, 6> {
 public:
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Eigen::IndexPair<int>& operator[] (size_t index) { return values[index]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const Eigen::IndexPair<int>& operator[] (size_t index) const { return values[index]; }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Eigen::IndexPair<int>& front() { return values[0]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const Eigen::IndexPair<int>& front() const { return values[0]; }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Eigen::IndexPair<int>& back() { return values[5]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const Eigen::IndexPair<int>& back() const { return values[5]; }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  static std::size_t size() { return 6; }

  Eigen::IndexPair<int> values[6];

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE array() { }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE ~array() { }

#if EIGEN_HAS_VARIADIC_TEMPLATES
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE array(std::initializer_list<Eigen::IndexPair<int> > l) {
    eigen_assert(l.size() == 6);
    internal::smart_copy(l.begin(), l.end(), values);
  }
#endif

  __attribute__((annotate("user_deserialize")))
  array(int v0_first, int v0_second,
        int v1_first, int v1_second,
        int v2_first, int v2_second,
        int v3_first, int v3_second,
        int v4_first, int v4_second,
        int v5_first, int v5_second) [[cpu]][[hc]] {
    values[0] = Eigen::IndexPair<int>(v0_first, v0_second);
    values[1] = Eigen::IndexPair<int>(v1_first, v1_second);
    values[2] = Eigen::IndexPair<int>(v2_first, v2_second);
    values[3] = Eigen::IndexPair<int>(v3_first, v3_second);
    values[4] = Eigen::IndexPair<int>(v4_first, v4_second);
    values[5] = Eigen::IndexPair<int>(v5_first, v5_second);
  }

  __attribute__((annotate("serialize")))
  void __cxxamp_serialize(Kalmar::Serialize &s) const {
    for (int i = 0; i < 6; ++i) {
      s.Append(sizeof(int), &values[i].first);
      s.Append(sizeof(int), &values[i].second);
    }
  }

  array(Eigen::IndexPair<int> v0,
        Eigen::IndexPair<int> v1,
        Eigen::IndexPair<int> v2,
        Eigen::IndexPair<int> v3,
        Eigen::IndexPair<int> v4,
        Eigen::IndexPair<int> v5) {
    values[0].set(v0);
    values[1].set(v1);
    values[2].set(v2);
    values[3].set(v3);
    values[4].set(v4);
    values[5].set(v5);
  }
};

// Specialize array for size 6 and T = Eigen::IndexPair<long>
template <>
class array<Eigen::IndexPair<long>, 6> {
 public:
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Eigen::IndexPair<long>& operator[] (size_t index) { return values[index]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const Eigen::IndexPair<long>& operator[] (size_t index) const { return values[index]; }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Eigen::IndexPair<long>& front() { return values[0]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const Eigen::IndexPair<long>& front() const { return values[0]; }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Eigen::IndexPair<long>& back() { return values[5]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const Eigen::IndexPair<long>& back() const { return values[5]; }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  static std::size_t size() { return 6; }

  Eigen::IndexPair<long> values[6];

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE array() { }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE ~array() { }

#if EIGEN_HAS_VARIADIC_TEMPLATES
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE array(std::initializer_list<Eigen::IndexPair<long> > l) {
    eigen_assert(l.size() == 6);
    internal::smart_copy(l.begin(), l.end(), values);
  }
#endif

  __attribute__((annotate("user_deserialize")))
  array(long v0_first, long v0_second,
        long v1_first, long v1_second,
        long v2_first, long v2_second,
        long v3_first, long v3_second,
        long v4_first, long v4_second,
        long v5_first, long v5_second) [[cpu]][[hc]] {
    values[0] = Eigen::IndexPair<long>(v0_first, v0_second);
    values[1] = Eigen::IndexPair<long>(v1_first, v1_second);
    values[2] = Eigen::IndexPair<long>(v2_first, v2_second);
    values[3] = Eigen::IndexPair<long>(v3_first, v3_second);
    values[4] = Eigen::IndexPair<long>(v4_first, v4_second);
    values[5] = Eigen::IndexPair<long>(v5_first, v5_second);
  }

  __attribute__((annotate("serialize")))
  void __cxxamp_serialize(Kalmar::Serialize &s) const {
    for (int i = 0; i < 6; ++i) {
      s.Append(sizeof(long), &values[i].first);
      s.Append(sizeof(long), &values[i].second);
    }
  }

  array(Eigen::IndexPair<long> v0,
        Eigen::IndexPair<long> v1,
        Eigen::IndexPair<long> v2,
        Eigen::IndexPair<long> v3,
        Eigen::IndexPair<long> v4,
        Eigen::IndexPair<long> v5) {
    values[0].set(v0);
    values[1].set(v1);
    values[2].set(v2);
    values[3].set(v3);
    values[4].set(v4);
    values[5].set(v5);
  }
};

// Specialize array for size 6 and T = std::pair<int, int>
template<>
class array<std::pair<int, int>, 6> {
 public:
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE std::pair<int, int>& operator[] (size_t index) { return values[index]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const std::pair<int, int>& operator[] (size_t index) const { return values[index]; }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE std::pair<int, int> front() { return values[0]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const std::pair<int, int>& front() const { return values[0]; }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE std::pair<int, int>& back() { return values[5]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const std::pair<int, int>& back() const { return values[5]; }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  static std::size_t size() { return 6; }

  std::pair<int, int> values[6];

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE array() { }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE ~array() { }

#if EIGEN_HAS_VARIADIC_TEMPLATES
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE array(std::initializer_list<std::pair<int, int> > l) {
    eigen_assert(l.size() == 6);
    internal::smart_copy(l.begin(), l.end(), values);
  }
#endif

  __attribute__((annotate("user_deserialize")))
  array(int v0_first, int v0_second,
        int v1_first, int v1_second,
        int v2_first, int v2_second,
        int v3_first, int v3_second,
        int v4_first, int v4_second,
        int v5_first, int v5_second) [[cpu]][[hc]] {
    values[0] = std::pair<int, int>(v0_first, v0_second);
    values[1] = std::pair<int, int>(v1_first, v1_second);
    values[2] = std::pair<int, int>(v2_first, v2_second);
    values[3] = std::pair<int, int>(v3_first, v3_second);
    values[4] = std::pair<int, int>(v4_first, v4_second);
    values[5] = std::pair<int, int>(v5_first, v5_second);
  }
  __attribute__((annotate("serialize")))
  void __cxxamp_serialize(Kalmar::Serialize &s) const {
    for (int i = 0; i < 6; ++i) {
      s.Append(sizeof(int), &values[i].first);
      s.Append(sizeof(int), &values[i].second);
    }
  }

  array(std::pair<int, int> v0,
        std::pair<int, int> v1,
        std::pair<int, int> v2,
        std::pair<int, int> v3,
        std::pair<int, int> v4,
        std::pair<int, int> v5) {
    values[0] = v0;
    values[1] = v1;
    values[2] = v2;
    values[3] = v3;
    values[4] = v4;
    values[5] = v5;
  }
};

// Specialize array for size 7 and T = Eigen::IndexPair<int>
template <>
class array<Eigen::IndexPair<int>, 7> {
 public:
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Eigen::IndexPair<int>& operator[] (size_t index) { return values[index]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const Eigen::IndexPair<int>& operator[] (size_t index) const { return values[index]; }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Eigen::IndexPair<int>& front() { return values[0]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const Eigen::IndexPair<int>& front() const { return values[0]; }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Eigen::IndexPair<int>& back() { return values[6]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const Eigen::IndexPair<int>& back() const { return values[6]; }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  static std::size_t size() { return 7; }

  Eigen::IndexPair<int> values[7];

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE array() { }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE ~array() { }

#if EIGEN_HAS_VARIADIC_TEMPLATES
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE array(std::initializer_list<Eigen::IndexPair<int> > l) {
    eigen_assert(l.size() == 7);
    internal::smart_copy(l.begin(), l.end(), values);
  }
#endif

  __attribute__((annotate("user_deserialize")))
  array(int v0_first, int v0_second,
        int v1_first, int v1_second,
        int v2_first, int v2_second,
        int v3_first, int v3_second,
        int v4_first, int v4_second,
        int v5_first, int v5_second,
        int v6_first, int v6_second) [[cpu]][[hc]] {
    values[0] = Eigen::IndexPair<int>(v0_first, v0_second);
    values[1] = Eigen::IndexPair<int>(v1_first, v1_second);
    values[2] = Eigen::IndexPair<int>(v2_first, v2_second);
    values[3] = Eigen::IndexPair<int>(v3_first, v3_second);
    values[4] = Eigen::IndexPair<int>(v4_first, v4_second);
    values[5] = Eigen::IndexPair<int>(v5_first, v5_second);
    values[6] = Eigen::IndexPair<int>(v6_first, v6_second);
  }

  __attribute__((annotate("serialize")))
  void __cxxamp_serialize(Kalmar::Serialize &s) const {
    for (int i = 0; i < 7; ++i) {
      s.Append(sizeof(int), &values[i].first);
      s.Append(sizeof(int), &values[i].second);
    }
  }

  array(Eigen::IndexPair<int> v0,
        Eigen::IndexPair<int> v1,
        Eigen::IndexPair<int> v2,
        Eigen::IndexPair<int> v3,
        Eigen::IndexPair<int> v4,
        Eigen::IndexPair<int> v5,
        Eigen::IndexPair<int> v6) {
    values[0].set(v0);
    values[1].set(v1);
    values[2].set(v2);
    values[3].set(v3);
    values[4].set(v4);
    values[5].set(v5);
    values[6].set(v6);
  }
};
 
// Specialize array for size 7 and T = Eigen::IndexPair<long>
template <>
class array<Eigen::IndexPair<long>, 7> {
 public:
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Eigen::IndexPair<long>& operator[] (size_t index) { return values[index]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const Eigen::IndexPair<long>& operator[] (size_t index) const { return values[index]; }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Eigen::IndexPair<long>& front() { return values[0]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const Eigen::IndexPair<long>& front() const { return values[0]; }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Eigen::IndexPair<long>& back() { return values[6]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const Eigen::IndexPair<long>& back() const { return values[6]; }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  static std::size_t size() { return 7; }

  Eigen::IndexPair<long> values[7];

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE array() { }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE ~array() { }

#if EIGEN_HAS_VARIADIC_TEMPLATES
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE array(std::initializer_list<Eigen::IndexPair<long> > l) {
    eigen_assert(l.size() == 7);
    internal::smart_copy(l.begin(), l.end(), values);
  }
#endif

  __attribute__((annotate("user_deserialize")))
  array(long v0_first, long v0_second,
        long v1_first, long v1_second,
        long v2_first, long v2_second,
        long v3_first, long v3_second,
        long v4_first, long v4_second,
        long v5_first, long v5_second,
        long v6_first, long v6_second) [[cpu]][[hc]] {
    values[0] = Eigen::IndexPair<long>(v0_first, v0_second);
    values[1] = Eigen::IndexPair<long>(v1_first, v1_second);
    values[2] = Eigen::IndexPair<long>(v2_first, v2_second);
    values[3] = Eigen::IndexPair<long>(v3_first, v3_second);
    values[4] = Eigen::IndexPair<long>(v4_first, v4_second);
    values[5] = Eigen::IndexPair<long>(v5_first, v5_second);
    values[6] = Eigen::IndexPair<long>(v6_first, v6_second);
  }

  __attribute__((annotate("serialize")))
  void __cxxamp_serialize(Kalmar::Serialize &s) const {
    for (int i = 0; i < 7; ++i) {
      s.Append(sizeof(long), &values[i].first);
      s.Append(sizeof(long), &values[i].second);
    }
  }

  array(Eigen::IndexPair<long> v0,
        Eigen::IndexPair<long> v1,
        Eigen::IndexPair<long> v2,
        Eigen::IndexPair<long> v3,
        Eigen::IndexPair<long> v4,
        Eigen::IndexPair<long> v5,
        Eigen::IndexPair<long> v6) {
    values[0].set(v0);
    values[1].set(v1);
    values[2].set(v2);
    values[3].set(v3);
    values[4].set(v4);
    values[5].set(v5);
    values[6].set(v6);
  }
};

// Specialize array for size 7 and T = std::pair<int, int>
template<>
class array<std::pair<int, int>, 7> {
 public:
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE std::pair<int, int>& operator[] (size_t index) { return values[index]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const std::pair<int, int>& operator[] (size_t index) const { return values[index]; }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE std::pair<int, int> front() { return values[0]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const std::pair<int, int>& front() const { return values[0]; }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE std::pair<int, int>& back() { return values[6]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const std::pair<int, int>& back() const { return values[6]; }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  static std::size_t size() { return 7; }

  std::pair<int, int> values[7];

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE array() { }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE ~array() { }

#if EIGEN_HAS_VARIADIC_TEMPLATES
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE array(std::initializer_list<std::pair<int, int> > l) {
    eigen_assert(l.size() == 7);
    internal::smart_copy(l.begin(), l.end(), values);
  }
#endif

  __attribute__((annotate("user_deserialize")))
  array(int v0_first, int v0_second,
        int v1_first, int v1_second,
        int v2_first, int v2_second,
        int v3_first, int v3_second,
        int v4_first, int v4_second,
        int v5_first, int v5_second,
        int v6_first, int v6_second) [[cpu]][[hc]] {
    values[0] = std::pair<int, int>(v0_first, v0_second);
    values[1] = std::pair<int, int>(v1_first, v1_second);
    values[2] = std::pair<int, int>(v2_first, v2_second);
    values[3] = std::pair<int, int>(v3_first, v3_second);
    values[4] = std::pair<int, int>(v4_first, v4_second);
    values[5] = std::pair<int, int>(v5_first, v5_second);
    values[6] = std::pair<int, int>(v6_first, v6_second);
  }
  __attribute__((annotate("serialize")))
  void __cxxamp_serialize(Kalmar::Serialize &s) const {
    for (int i = 0; i < 7; ++i) {
      s.Append(sizeof(int), &values[i].first);
      s.Append(sizeof(int), &values[i].second);
    }
  }

  array(std::pair<int, int> v0,
        std::pair<int, int> v1,
        std::pair<int, int> v2,
        std::pair<int, int> v3,
        std::pair<int, int> v4,
        std::pair<int, int> v5,
        std::pair<int, int> v6) {
    values[0] = v0;
    values[1] = v1;
    values[2] = v2;
    values[3] = v3;
    values[4] = v4;
    values[5] = v5;
    values[6] = v6;
  }
};

// Specialize array for size 8 and T = Eigen::IndexPair<int>
template <>
class array<Eigen::IndexPair<int>, 8> {
 public:
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Eigen::IndexPair<int>& operator[] (size_t index) { return values[index]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const Eigen::IndexPair<int>& operator[] (size_t index) const { return values[index]; }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Eigen::IndexPair<int>& front() { return values[0]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const Eigen::IndexPair<int>& front() const { return values[0]; }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Eigen::IndexPair<int>& back() { return values[7]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const Eigen::IndexPair<int>& back() const { return values[7]; }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  static std::size_t size() { return 8; }

  Eigen::IndexPair<int> values[8];

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE array() { }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE ~array() { }

#if EIGEN_HAS_VARIADIC_TEMPLATES
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE array(std::initializer_list<Eigen::IndexPair<int> > l) {
    eigen_assert(l.size() == 8);
    internal::smart_copy(l.begin(), l.end(), values);
  }
#endif

  __attribute__((annotate("user_deserialize")))
  array(int v0_first, int v0_second,
        int v1_first, int v1_second,
        int v2_first, int v2_second,
        int v3_first, int v3_second,
        int v4_first, int v4_second,
        int v5_first, int v5_second,
        int v6_first, int v6_second,
        int v7_first, int v7_second) [[cpu]][[hc]] {
    values[0] = Eigen::IndexPair<int>(v0_first, v0_second);
    values[1] = Eigen::IndexPair<int>(v1_first, v1_second);
    values[2] = Eigen::IndexPair<int>(v2_first, v2_second);
    values[3] = Eigen::IndexPair<int>(v3_first, v3_second);
    values[4] = Eigen::IndexPair<int>(v4_first, v4_second);
    values[5] = Eigen::IndexPair<int>(v5_first, v5_second);
    values[6] = Eigen::IndexPair<int>(v6_first, v6_second);
    values[7] = Eigen::IndexPair<int>(v7_first, v7_second);
  }

  __attribute__((annotate("serialize")))
  void __cxxamp_serialize(Kalmar::Serialize &s) const {
    for (int i = 0; i < 8; ++i) {
      s.Append(sizeof(int), &values[i].first);
      s.Append(sizeof(int), &values[i].second);
    }
  }

  array(Eigen::IndexPair<int> v0,
        Eigen::IndexPair<int> v1,
        Eigen::IndexPair<int> v2,
        Eigen::IndexPair<int> v3,
        Eigen::IndexPair<int> v4,
        Eigen::IndexPair<int> v5,
        Eigen::IndexPair<int> v6,
        Eigen::IndexPair<int> v7) {
    values[0].set(v0);
    values[1].set(v1);
    values[2].set(v2);
    values[3].set(v3);
    values[4].set(v4);
    values[5].set(v5);
    values[6].set(v6);
    values[7].set(v7);
  }
};

// Specialize array for size 8 and T = Eigen::IndexPair<long>
template <>
class array<Eigen::IndexPair<long>, 8> {
 public:
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Eigen::IndexPair<long>& operator[] (size_t index) { return values[index]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const Eigen::IndexPair<long>& operator[] (size_t index) const { return values[index]; }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Eigen::IndexPair<long>& front() { return values[0]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const Eigen::IndexPair<long>& front() const { return values[0]; }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE Eigen::IndexPair<long>& back() { return values[7]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const Eigen::IndexPair<long>& back() const { return values[7]; }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  static std::size_t size() { return 8; }

  Eigen::IndexPair<long> values[8];

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE array() { }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE ~array() { }

#if EIGEN_HAS_VARIADIC_TEMPLATES
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE array(std::initializer_list<Eigen::IndexPair<long> > l) {
    eigen_assert(l.size() == 8);
    internal::smart_copy(l.begin(), l.end(), values);
  }
#endif

  __attribute__((annotate("user_deserialize")))
  array(long v0_first, long v0_second,
        long v1_first, long v1_second,
        long v2_first, long v2_second,
        long v3_first, long v3_second,
        long v4_first, long v4_second,
        long v5_first, long v5_second,
        long v6_first, long v6_second,
        long v7_first, long v7_second) [[cpu]][[hc]] {
    values[0] = Eigen::IndexPair<long>(v0_first, v0_second);
    values[1] = Eigen::IndexPair<long>(v1_first, v1_second);
    values[2] = Eigen::IndexPair<long>(v2_first, v2_second);
    values[3] = Eigen::IndexPair<long>(v3_first, v3_second);
    values[4] = Eigen::IndexPair<long>(v4_first, v4_second);
    values[5] = Eigen::IndexPair<long>(v5_first, v5_second);
    values[6] = Eigen::IndexPair<long>(v6_first, v6_second);
    values[7] = Eigen::IndexPair<long>(v7_first, v7_second);
  }

  __attribute__((annotate("serialize")))
  void __cxxamp_serialize(Kalmar::Serialize &s) const {
    for (int i = 0; i < 8; ++i) {
      s.Append(sizeof(long), &values[i].first);
      s.Append(sizeof(long), &values[i].second);
    }
  }

  array(Eigen::IndexPair<long> v0,
        Eigen::IndexPair<long> v1,
        Eigen::IndexPair<long> v2,
        Eigen::IndexPair<long> v3,
        Eigen::IndexPair<long> v4,
        Eigen::IndexPair<long> v5,
        Eigen::IndexPair<long> v6,
        Eigen::IndexPair<long> v7) {
    values[0].set(v0);
    values[1].set(v1);
    values[2].set(v2);
    values[3].set(v3);
    values[4].set(v4);
    values[5].set(v5);
    values[6].set(v6);
    values[7].set(v7);
  }
};

// Specialize array for size 8 and T = std::pair<int, int>
template<>
class array<std::pair<int, int>, 8> {
 public:
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE std::pair<int, int>& operator[] (size_t index) { return values[index]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const std::pair<int, int>& operator[] (size_t index) const { return values[index]; }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE std::pair<int, int> front() { return values[0]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const std::pair<int, int>& front() const { return values[0]; }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE std::pair<int, int>& back() { return values[7]; }
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE const std::pair<int, int>& back() const { return values[7]; }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  static std::size_t size() { return 8; }

  std::pair<int, int> values[8];

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE array() { }

  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE ~array() { }

#if EIGEN_HAS_VARIADIC_TEMPLATES
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE array(std::initializer_list<std::pair<int, int> > l) {
    eigen_assert(l.size() == 8);
    internal::smart_copy(l.begin(), l.end(), values);
  }
#endif

  __attribute__((annotate("user_deserialize")))
  array(int v0_first, int v0_second,
        int v1_first, int v1_second,
        int v2_first, int v2_second,
        int v3_first, int v3_second,
        int v4_first, int v4_second,
        int v5_first, int v5_second,
        int v6_first, int v6_second,
        int v7_first, int v7_second) [[cpu]][[hc]] {
    values[0] = std::pair<int, int>(v0_first, v0_second);
    values[1] = std::pair<int, int>(v1_first, v1_second);
    values[2] = std::pair<int, int>(v2_first, v2_second);
    values[3] = std::pair<int, int>(v3_first, v3_second);
    values[4] = std::pair<int, int>(v4_first, v4_second);
    values[5] = std::pair<int, int>(v5_first, v5_second);
    values[6] = std::pair<int, int>(v6_first, v6_second);
    values[7] = std::pair<int, int>(v7_first, v7_second);
  }
  __attribute__((annotate("serialize")))
  void __cxxamp_serialize(Kalmar::Serialize &s) const {
    for (int i = 0; i < 8; ++i) {
      s.Append(sizeof(int), &values[i].first);
      s.Append(sizeof(int), &values[i].second);
    }
  }

  array(std::pair<int, int> v0,
        std::pair<int, int> v1,
        std::pair<int, int> v2,
        std::pair<int, int> v3,
        std::pair<int, int> v4,
        std::pair<int, int> v5,
        std::pair<int, int> v6,
        std::pair<int, int> v7) {
    values[0] = v0;
    values[1] = v1;
    values[2] = v2;
    values[3] = v3;
    values[4] = v4;
    values[5] = v5;
    values[6] = v6;
    values[7] = v7;
  }
};

#endif // #ifdef __HCC__


#ifdef EIGEN_HAS_SFINAE
namespace internal {

  template<typename IndexType, Index... Is>
  EIGEN_CONSTEXPR EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  array<Index, sizeof...(Is)> customIndices2Array(IndexType& idx, numeric_list<Index, Is...>) {
    return { idx[Is]... };
  }
  template<typename IndexType>
  EIGEN_CONSTEXPR EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  array<Index, 0> customIndices2Array(IndexType&, numeric_list<Index>) {
    return array<Index, 0>();
  }

  /** Make an array (for index/dimensions) out of a custom index */
  template<typename Index, std::size_t NumIndices, typename IndexType>
  EIGEN_CONSTEXPR EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
  array<Index, NumIndices> customIndices2Array(IndexType& idx) {
    return customIndices2Array(idx, typename gen_numeric_list<Index, NumIndices>::type{});
  }


  template <typename B, typename D>
  struct is_base_of
  {

    typedef char (&yes)[1];
    typedef char (&no)[2];

    template <typename BB, typename DD>
    struct Host
    {
      operator BB*() const;
      operator DD*();
    };

    template<typename T>
    static yes check(D*, T);
    static no check(B*, int);

    static const bool value = sizeof(check(Host<B,D>(), int())) == sizeof(yes);
  };

}
#endif



}  // namespace Eigen

#endif  // EIGEN_CXX11_TENSOR_TENSOR_META_H
