#ifndef CPU_UTIL
#define CPU_UTIL

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define checkMKLError(a)                                                       \
  do {                                                                         \
    if (SPARSE_STATUS_SUCCESS != (a)) {                                        \
      fprintf(stderr, "MKL runTime error in line %d of file %s \n", __LINE__,  \
              __FILE__);                                                       \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#endif
