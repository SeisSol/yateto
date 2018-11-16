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

arch = useArchitectureIdentifiedBy('dsnb')

N = 8
A = Tensor('A', (N, N))
B = Tensor('B', (N, N, N))
w = Tensor('w', (N,))
C = Tensor('C', (N, N))

g = Generator(arch)

kernel = C['ij'] <= 2.0 * C['ij'] + A['lj'] * B['ikl'] * w['k']
g.add('kernel', kernel)

g.generate('generated_code')

PrettyPrinter().visit(kernel)
