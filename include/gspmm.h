#ifndef GSPMM
#define GSPMM

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <string.h>

#include <iostream>
#include <vector>

enum REDUCEOP { SUM, MAX, MIN, MEAN };
enum COMPUTEOP { ADD, SUB, MUL, DIV };

#define MIN(a, b) ((a < b) ? a : b)
#define MAX(a, b) ((a < b) ? b : a)

struct Sum {
  static __device__ __forceinline__ float reduce(const float &acc,
                                                 const float &val) {
    return acc + val;
  }
  // static __device__ __forceinline__ float atomic_reduce(float *addr,
  //                                                       float &val) {
  //   return atomicAdd(addr, val);
  // }
  const static enum REDUCEOP Op = REDUCEOP::SUM;
};

struct Max {
  static __device__ __forceinline__ float reduce(const float &acc,
                                                 const float &val) {
    return MAX(acc, val);
  }
  // static __device__ __forceinline__ float atomic_reduce(float *addr,
  //                                                       float &val) {
  //   return atomicAdd(addr, val);
  // }
  const static enum REDUCEOP Op = REDUCEOP::MAX;
};

struct Min {
  static __device__ __forceinline__ float reduce(const float &acc,
                                                 const float &val) {
    return MIN(acc, val);
  }
  // static __device__ __forceinline__ float atomic_reduce(float *addr,
  //                                                       float &val) {
  //   return atomicAdd(addr, val);
  // }
  const static enum REDUCEOP Op = REDUCEOP::MIN;
};

struct Mean {
  static __device__ __forceinline__ float reduce(const float &acc,
                                                 const float &val) {
    return acc + val;
  }
  // static __device__ __forceinline__ float atomic_reduce(float *addr,
  //                                                       float &val) {
  //   return atomicAdd(addr, val);
  // }
  const static enum REDUCEOP Op = REDUCEOP::MEAN;
};

struct Mul {
  static __device__ __forceinline__ float compute(const float &a,
                                                  const float &b) {
    return a * b;
  }
};

struct Add {
  static __device__ __forceinline__ float compute(const float &a,
                                                  const float &b) {
    return a + b;
  }
};

struct Sub {
  static __device__ __forceinline__ float compute(const float &a,
                                                  const float &b) {
    return b - a;
  }
};

struct Div {
  static __device__ __forceinline__ float compute(const float &a,
                                                  const float &b) {
    return b / a;
  }
};

#define SWITCH_REDUCEOP(op, Op, ...)                                           \
  ({                                                                           \
    if (op == SUM) {                                                           \
      typedef Sum Op;                                                          \
      { __VA_ARGS__ }                                                          \
    } else if (op == MAX) {                                                    \
      typedef Max Op;                                                          \
      { __VA_ARGS__ }                                                          \
    } else if (op == MIN) {                                                    \
      typedef Min Op;                                                          \
      { __VA_ARGS__ }                                                          \
    } else if (op == MEAN) {                                                   \
      typedef Mean Op;                                                         \
      { __VA_ARGS__ }                                                          \
    } else {                                                                   \
      printf("unsupported op \n");                                             \
    }                                                                          \
  })

#define SWITCH_COMPUTEOP(op, Op, ...)                                          \
  ({                                                                           \
    if (op == ADD) {                                                           \
      typedef Add Op;                                                          \
      { __VA_ARGS__ }                                                          \
    } else if (op == SUB) {                                                    \
      typedef Sub Op;                                                          \
      { __VA_ARGS__ }                                                          \
    } else if (op == MUL) {                                                    \
      typedef Mul Op;                                                          \
      { __VA_ARGS__ }                                                          \
    } else if (op == DIV) {                                                    \
      typedef Div Op;                                                          \
      { __VA_ARGS__ }                                                          \
    } else {                                                                   \
      printf("unsupported op \n");                                             \
    }                                                                          \
  })

__device__ __forceinline__ float init(REDUCEOP op) {
  switch (op) {
  case SUM:
    return 0;
  case MAX:
    return INT_MIN;
  case MIN:
    return INT_MAX;
  case MEAN:
    return 0;
  default:
    return 0;
  }
}

#endif
