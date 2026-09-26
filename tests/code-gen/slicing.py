#!/usr/bin/env python3

from yateto import *

import numpy as np

def add(g):
  M = 32
  N = 40
  K = 40
  A = Tensor('A', (M, K))
  C = Tensor('C', (M, N))

  XA = Tensor('XA', (32, 32))
  XB = Tensor('XB', (32, 32))
  XC = Tensor('XC', (32, 32))

  class Counter:
    def __init__(self):
      self.counter = 0

  counter = Counter()

  def _(kernel):
    counter.counter += 1
    g.add(f'kernel{counter.counter}', kernel)

  _(C['ij'].subslice('j', 4, 36) <= (A['ij']).subslice('j', 8, 40) * (A['ij']).subslice('j', 0, 32))
  _([
    XC['ij'].subslice('i', i*16, (i+1)*16).subslice('j', j*16, (j+1)*16) <= (XA['ik']).subslice('i', i*16, (i+1)*16) * XB['kj'].subslice('j', j*16, (j+1)*16)
    for i in range(2) for j in range(2)
  ])

  # A product with a sparse factor is zero where the factor is, and the
  # operation narrows its other operands to that. Taken through a slice, the
  # narrowing has to survive the passes after it, or the dense operand
  # reaches past the result.
  N = 8
  table = np.zeros((N, N))
  table[0, 0] = 0.5
  table[1, 2] = -1.0
  table[7, 3] = 2.0
  table[4, 4] = 1.0
  T = Tensor('T', (N, N), spp=table)
  P = Tensor('P', (N, N))
  Q = Tensor('Q', (N, N))
  _(P['kc'].subslice('c', 5, 8) <= T['kc'].subslice('c', 2, 5) * Q['kc'].subslice('c', 5, 8))
  _(P['kc'].subslice('c', 0, 3) <= T['kc'].subslice('c', 2, 5) * Q['kc'].subslice('c', 0, 3))
  _(P['kc'] <= Q['kc'] * T['kc'])
