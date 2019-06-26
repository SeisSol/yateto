#!/usr/bin/env python3

#from yateto import *


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

  print("start")
  kernel = C['ij'] <= 2.0 * C['ij'] + A['lj'] * B['ikl'] * w['k']
  g.add('kernel', kernel)




if __name__ == "__main__":

  import sys
  sys.path.append('..')
  from yateto import *
  import trace
  from hunter import trace, Q, CallPrinter, CodePrinter, VarsPrinter




  mode = "debug"
  #mode = "trace"

  if mode == "trace":
    trace(Q(stdlib=False), action=CallPrinter)
    arch = useArchitectureIdentifiedBy("dhsw")
    g = Generator(arch)
    print("start generation xxxx")
    add(g)
    print("start generation yyyy")
    g.generate(outputDir="./rav-test", gemm_cfg=gemm_cfg(arch, "onlyblas"))

  else:
    arch = useArchitectureIdentifiedBy("dhsw")
    g = Generator(arch)
    add(g)
    g.generate(outputDir="./rav-test", gemm_cfg=gemm_cfg(arch, "onlyblas"))