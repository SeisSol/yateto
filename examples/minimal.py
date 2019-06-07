#!/usr/bin/env python3

from yateto import *


def gemm_cfg(arch, variant):
  if variant == 'onlyblas':
    return GeneratorCollection([MKL(arch)])
  return GeneratorCollection([LIBXSMM(arch), MKL(arch)])

def add(g):

  N = 8
  A = Tensor('A', (N, N))
  B = Tensor('B', (N, N, N))
  w = Tensor('w', (N,))
  C = Tensor('C', (N, N))

  kernel = C['ij'] <= 2.0 * C['ij'] + A['lj'] * B['ikl'] * w['k']
  g.add('kernel', kernel)


if __name__ == "__main__":

  import sys
  sys.path.append('..')
  import trace


  arch = useArchitectureIdentifiedBy("dhsw")
  g = Generator(arch)

  mode = "debug"
  #mode = "trace"

  if mode == "trace":
    tracer = trace.Trace(
      ignoredirs=[sys.prefix, sys.exec_prefix],
      trace=1,
      count=1)
    tracer.run('add(g)')
  else:
    add(g)
  #results = tracer.results()