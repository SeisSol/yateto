#!/usr/bin/env python3

import sys
import os, errno
sys.path.append('..')
try:
  os.mkdir('generated_code')
except OSError as e:
  if e.errno == errno.EEXIST:
    pass

from yateto import *
from yateto.ast.visitor import PrettyPrinter
from yateto.ast.cost import ExactCost, ShapeCostEstimator, BoundingBoxCostEstimator

from string import ascii_lowercase as alph
from functools import reduce

arch = useArchitectureIdentifiedBy('dsnb')

N = 5

n = 8
r = 4
X = Tensor('X', tuple(n for i in range(N)))
G = Tensor('G', tuple(r for i in range(N)))
A = [Tensor('A({})'.format(i), (n,r)) for i in range(N)]

g = Generator(arch)

Alist = [A[i][alph[13+i] + alph[i]] for i in range(N)]
hosvd = G[alph[0:N]] <= X[alph[13:13+N]] * reduce(lambda x, y: x * y, Alist)
g.add('hosvd', hosvd)

g.generate('generated_code')

PrettyPrinter().visit(hosvd)
