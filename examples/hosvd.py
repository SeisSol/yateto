#!/usr/bin/env python3

from string import ascii_lowercase as alph
from functools import reduce
from yateto import *

def add(g):
  N = 4

  n = 64
  r = 16
  X = Tensor('X', tuple(n for i in range(N)))
  G = Tensor('G', tuple(r for i in range(N)))
  A = [Tensor('A({})'.format(i), (n,r) if i > 0 else (r,n)) for i in range(N)]

  Alist = [A[0][alph[0] + alph[13]]] + [A[i][alph[13+i] + alph[i]] for i in range(1,N)]
  hosvd = G[alph[0:N]] <= X[alph[13:13+N]] * reduce(lambda x, y: x * y, Alist)
  g.add('hosvd', hosvd)
