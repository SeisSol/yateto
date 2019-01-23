#!/usr/bin/env python3

from yateto import *

def add(g):
  M = 16
  N = 24
  K = 48
  A = Tensor('A', (M, K))
  B = Tensor('B', (K, N))
  C = Tensor('C', (M, N))

  kernel = C['ij'] <= A['ik'] * B['kj']
  g.add('matmul', kernel)
