#!/usr/bin/env python3

from yateto import *
from yateto.type import AddressingMode

import numpy as np
import yateto.functions as yf

# Operands whose numbers are part of the generated code. The unit test computes
# the reference from a buffer filled with the same numbers, so every kernel
# here checks that the numbers the generator writes are the tensor's own, at
# the entries they belong to.

def add(g):
  N = 8
  A = Tensor('A', (N, N))
  B = Tensor('B', (N, N))
  v = Tensor('v', (N,))

  def immediate(name, values):
    return Tensor(name, values.shape, spp=values, addressing=AddressingMode.IMMEDIATE)

  table = np.zeros((N, N))
  table[0, 0] = 0.5
  table[1, 2] = -1.0
  table[7, 3] = 2.0
  table[4, 4] = 1.0
  T = immediate('T', table)

  unit = np.zeros(N)
  unit[0] = 1.0
  e0 = immediate('e0', unit)

  trace = immediate('trace', np.array([1.0] * 3 + [0.0] * (N - 3)))

  class Counter:
    def __init__(self):
      self.counter = 0

  counter = Counter()

  def _(kernel):
    counter.counter += 1
    g.add(f'kernel{counter.counter}', kernel)

  # element-wise: the numbers are written where they are read
  _(A['ij'] <= T['ij'] * B['ij'])
  _(A['ij'] <= yf.add(T['ij'], B['ij']))
  _(A['ij'] <= yf.mul(v['i'], trace['j']))
  _(A['ij'] <= yf.maximum(B['ij'], trace['j']))

  # into a window of a larger tensor: the columns outside the window are the
  # test's own pattern before and after, and the comparison covers them
  _(A['kc'].subslice('c', 1, 3) <= e0['k'] * v['c'].subslice('c', 0, 2))
  _(A['kc'].subslice('c', 5, 8) <= T['kc'].subslice('c', 2, 5) * v['k'])
