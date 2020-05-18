#!/usr/bin/env python3

from yateto import *

def add(g):
  M = 32
  N = 32
  K = 32
  A = Tensor('A', (M, K))
  B = Tensor('B', (K, N))
  C = Tensor('C', (M, N))

  g.add('matmulAB', C['ij'] <= A['ik'] * B['kj'])
  g.add('matmulATB', C['ij'] <= A['ki'] * B['kj'])
  g.add('matmulABT', C['ij'] <= A['ik'] * B['jk'])
  g.add('matmulATBT', C['ij'] <= A['ki'] * B['jk'])
