#ifndef YATETO_MARKER_H_
#define YATETO_MARKER_H_

// Markers for the functions a consumer may call from device code.
//
// CUDA and HIP compile a translation unit once per side and reject a call
// into a function the current side does not own, so a header that is used
// from both has to say so on every function. Outside those compilers the
// markers are empty, which is also what a single-source model such as SYCL
// needs: there the same function is simply compiled for both sides.
//
// A marked function has to be callable on a device, which rules out the
// parts of the standard library that are host-only. That is the same
// constraint constexpr imposes before C++20, so the two go together.
// __CUDACC__ comes from a header rather than from the compiler -- nvcc sets
// it, and clang gets it from the runtime wrapper it force-includes -- so
// clang's own __CUDA__ is asked for as well.
#if defined(__CUDACC__) || defined(__CUDA__) || defined(__HIP__) || defined(__HIPCC__)
#define YATETO_HOST __host__
#define YATETO_DEVICE __device__
#else
#define YATETO_HOST
#define YATETO_DEVICE
#endif

#define YATETO_HOSTDEVICE YATETO_HOST YATETO_DEVICE

#endif // YATETO_MARKER_H_
