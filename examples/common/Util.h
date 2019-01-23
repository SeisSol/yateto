#ifndef NDEBUG
long long libxsmm_num_total_flops = 0;
#endif

#if REAL_SIZE == 8
typedef double real;
#elif REAL_SIZE == 4
typedef float real;
#else
#  error REAL_SIZE not supported.
#endif

real fillWithStuff(real* A, unsigned reals) {
  for (unsigned j = 0; j < reals; ++j) {
      A[j] = drand48();
  }
}

