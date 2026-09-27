#!/usr/bin/env python3

from yateto import *
from yateto.type import AddressingMode

import numpy as np
import yateto.functions as yf

# Kernels whose unit tests bind the constant pool, and what the pool must not
# take away from them. The pool holds a constant as the kernel reads it, so a
# test binds it after assigning its own buffers to the constants; it holds
# nothing for a member without values, and it knows nothing of the cases a
# test runs a condition through, so those are assigned after it. Built in
# Debug, a member the pool left null stops the kernel at its assert.

def add(g):
  N = 4
  A = Tensor('A', (N, N))
  B = Tensor('B', (N, N))
  C = Tensor('C', (N, N))
  out = Tensor('out', (N, N))

  values = np.zeros((N, N))
  values[0, 0] = 0.5
  values[1, 2] = -1.0
  values[3, 3] = 2.0

  # a constant passed as an argument, read from the pool
  D = Tensor('D', (N, N), spp=values)
  g.add('constant', out['ij'] <= D['ik'] * B['kj'])

  # a family whose member 0 has no values and member 1 does
  F = [Tensor('F(0)', (N, N)), Tensor('F(1)', (N, N), spp=values)]
  g.add('mixed', out['ij'] <= F[0]['ik'] * B['kj'] + F[1]['ik'] * C['kj'])

  # the same, over the variants of a kernel family
  H = [Tensor('H(0)', (N, N)), Tensor('H(1)', (N, N), spp=values)]
  G = Tensor('G', (N, N), spp=values)
  g.addFamily('mixedFamily', simpleParameterSpace(2),
              lambda i: out['ij'] <= H[i]['ik'] * G['kj'] if i == 0
                        else out['ij'] <= H[i]['ik'] * B['kj'])

  # a condition that carries a value: the test runs both cases of it
  X = Tensor('X', (), spp={(): 1}, datatype=Datatype.BOOL)
  g.add('constantCondition', yf.assignIf(X[''], A['ij'], yf.sqrt(B['ij'])))

  # an immediate the looped GEMM reads from memory, next to a family whose
  # member 0 has no values
  cube = np.zeros((N, N, 2))
  cube[0, 0, 0] = 0.5
  cube[1, 2, 1] = -1.0
  I = Tensor('I', (N, N, 2), spp=cube, addressing=AddressingMode.IMMEDIATE)
  A3 = Tensor('A3', (N, N, 2))
  P = [Tensor('P(0)', (N, N)), Tensor('P(1)', (N, N), spp=values)]
  g.add('immediateAndMixed', [A3['ijl'] <= I['ikl'] * B['kj'],
                              out['ij'] <= P[0]['ik'] * B['kj'] + P[1]['ik'] * C['kj']])
