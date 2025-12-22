#!/usr/bin/env python3

from yateto import *

from yateto.memory import *

import numpy as np

def add(g):
  N = 4
  A = Tensor('A', (N,), spp=np.array([1,0,0,1]), memoryLayoutClass=PatternMemoryLayout)
  B = Tensor('B', (N,), spp=np.array([1,0,1,0]), memoryLayoutClass=PatternMemoryLayout)
  D = Tensor('D', (N,), spp=np.ones((N,)), memoryLayoutClass=PatternMemoryLayout)
  B2 = Tensor('B2', (N, N), spp=np.ones((N, N)), memoryLayoutClass=PatternMemoryLayout)
  B3 = Tensor('B3', (N, N), spp=np.ones((N, N)), memoryLayoutClass=CSCMemoryLayout)
  Z = Tensor('Z', (N, N, N), spp=np.ones((N, N, N)), memoryLayoutClass=PatternMemoryLayout)
  C = Tensor('C', (N, N))
  C2 = Tensor('C2', (N, N, N))
  C3 = Tensor('C3', (N, N, N, N, N, N))

  class Counter:
    def __init__(self):
      self.counter = 0

  counter = Counter()

  def _(kernel):
    counter.counter += 1
    g.add(f'kernel{counter.counter}', kernel)

  _(C['ab'] <= A['a'])
  _(C['ab'] <= A['a'] + B['a'])
  _(C['ab'] <= A['a'] + B['b'])
  _(C['ab'] <= A['a'] * B['b'])
  _(C2['abc'] <= A['a'] + B['b'])
  _(C['ij'] <= B2['ik'] * B3['kj'])
  _(C2['zij'] <= B2['ik'] * Z['kjz'])
