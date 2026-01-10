#!/usr/bin/env python3

from yateto import *

from yateto.memory import *

import numpy as np

def checkerboard(shape):
  # cf. https://stackoverflow.com/a/51715491
  return np.indices(tuple(shape)).sum(axis=0) % 2

def invert(pattern):
  return 1 - pattern

def add(g):
  N = 4
  M = 2
  A = Tensor('A', (N,), spp=np.array([1,0,0,1]), memoryLayoutClass=PatternMemoryLayout)
  B = Tensor('B', (N,), spp=np.array([1,0,1,0]), memoryLayoutClass=PatternMemoryLayout)
  D = Tensor('D', (N,), spp=np.ones((N,)), memoryLayoutClass=PatternMemoryLayout)
  E = Tensor('E', (N,), spp=checkerboard((N,)), memoryLayoutClass=PatternMemoryLayout)
  B2 = Tensor('B2', (N, N), spp=np.ones((N, N)), memoryLayoutClass=PatternMemoryLayout)
  B3 = Tensor('B3', (N, N), spp=np.ones((N, N)), memoryLayoutClass=CSCMemoryLayout)
  B4 = Tensor('B4', (N, N), spp=checkerboard((N, N)), memoryLayoutClass=PatternMemoryLayout)
  B5 = Tensor('B5', (N, N), spp=invert(checkerboard((N, N))), memoryLayoutClass=PatternMemoryLayout)
  Z = Tensor('Z', (N, N, N), spp=np.ones((N, N, N)), memoryLayoutClass=PatternMemoryLayout)
  C0 = Tensor('C0', (N,))
  C = Tensor('C', (N, N))
  C2 = Tensor('C2', (N, N, N))

  YA = Tensor('YA', (M, M, M), spp=checkerboard((M, M, M)), memoryLayoutClass=PatternMemoryLayout)
  YB = Tensor('YB', (M, M, M), spp=checkerboard((M, M, M)), memoryLayoutClass=PatternMemoryLayout)
  YC = Tensor('YC', (M, M, M))
  C4 = Tensor('C4', (M, M, M, M))

  ZA = Tensor('ZA', (M, M, M, M, M, M), spp=checkerboard((M, M, M, M, M, M)), memoryLayoutClass=PatternMemoryLayout)
  ZB = Tensor('ZB', (M, M, M, M, M, M), spp=invert(checkerboard((M, M, M, M, M, M))), memoryLayoutClass=PatternMemoryLayout)
  ZC = Tensor('ZC', (M, M, M, M, M, M))

  class Counter:
    def __init__(self):
      self.counter = 0

  counter = Counter()

  def _(kernel):
    counter.counter += 1
    g.add(f'kernel{counter.counter}', kernel)

  _(C['ab'] <= A['a'])
  _(C['ab'] <= A['a'] + B['b'])
  _(C['ab'] <= A['a'] * B['b'])
  _(C2['abc'] <= A['a'] + B['b'])
  _(C2['abc'] <= B4['ac'])
  _(C['ij'] <= B2['ik'] * B3['kj'])
  _(C['ij'] <= B4['ik'] * B3['kj'])
  _(C['ij'] <= B4['ik'] * B3['kj'] * B4['ij'])
  _(C2['zij'] <= B2['ik'] * Z['kjz'])
  _(C0['k'] <= B4['ik'] * A['i'])

  _(C4['ijXY'] <= YA['Zik'] * YB['kXY'] * YC['Zkj'])
  _(ZC['abcxyz'] <= ZA['abcijk'] * ZB['ijkxyz'])
  _(ZC['abcxyz'] <= ZA['aibjck'] * ZB['zkyjxi'])
