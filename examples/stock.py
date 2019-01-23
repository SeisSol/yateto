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

p = 16
q = 4
R = Tensor('R', (p,p,p))
S = Tensor('S', (p,p,p))
XL = Tensor('XL', (p,q))
XR = Tensor('XR', (q,p))
YL = Tensor('YL', (p,q))
YR = Tensor('YR', (q,p))
ZL = Tensor('ZL', (p,q))
ZR = Tensor('ZR', (q,p))

g = Generator(arch)

kernel = R['ijk'] <= S['xyz'] * XL['xl'] * XR['li'] * YL['ym'] * YR['mj'] * ZL['zn'] * ZR['nk']
g.add('kernel', kernel)

g.generate('generated_code')

PrettyPrinter().visit(kernel)
