#!/usr/bin/env python3

from yateto import *

def add(g):
  M = 32
  N = 40
  K = 40
  A = Tensor('A', (M, K))
  # B = Tensor('B', (K, N))
  C = Tensor('C', (M, N))

  XA = Tensor('XA', (32, 32))
  XB = Tensor('XB', (32, 32))
  XC = Tensor('XC', (32, 32))

  g.add('slicing1', C['ij'].subslice('j', 4, 36) <= (A['ij']).subslice('j', 8, 40) * (A['ij']).subslice('j', 0, 32))
  g.add('slicing2', [
    XC['ij'].subslice('i', i*16, (i+1)*16).subslice('j', j*16, (j+1)*16) <= (XA['ik']).subslice('i', i*16, (i+1)*16) * XB['kj'].subslice('j', j*16, (j+1)*16)
    for i in range(2) for j in range(2)
  ])
