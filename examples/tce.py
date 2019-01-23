#!/usr/bin/env python3

from yateto import *

def add(g):
  N = 8
  A = Tensor('A', (N, N, N, N))
  B = Tensor('B', (N, N, N, N))
  C = Tensor('C', (N, N, N, N))
  D = Tensor('D', (N, N, N, N))
  S = Tensor('S', (N, N, N, N))

  V = 16
  C1 = Tensor('C1', (N, V))
  C2 = Tensor('C2', (N, V))
  C3 = Tensor('C3', (N, V))
  C4 = Tensor('C4', (N, V))
  B2 = Tensor('T', (V, V, V, V))

  kernel = S['abij'] <= A['acik'] * B['befl'] * C['dfjk'] * D['cdel']
  g.add('tce1', kernel)

  kernel = B2['abcd'] <= C1['sd'] * C2['rc'] * C3['qb'] * C4['pa'] * A['pqrs']
  g.add('tce2', kernel)
