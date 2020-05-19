#!/usr/bin/env python3

import math
from yateto import *

def gemm_cfg(arch, variant):
  if variant == 'onlyblas':
    return GeneratorCollection([MKL(arch)])
  return GeneratorCollection([LIBXSMM(arch), MKL(arch)])

def add(g):
  for ep in range(2,7):
    p = 8*ep
    px = '{}p'.format(p)
    qmax = p//2
    steps = min(5, qmax)
    delta = (qmax-1)/(steps-1)
    qset1 = set(int(1 + eq*delta) for eq in range(0,steps))
    qset2 = set(2**i for i in range(0, int(math.log(qmax,2)+1)))
    qset = qset1.union(qset2)
    for q in qset:
      pqx = '{}p{}q'.format(p,q)
      R = Tensor('R' + px, (p,p,p))
      S = Tensor('S' + px, (p,p,p))
      XL = Tensor('XL' + pqx, (p,q))
      XR = Tensor('XR' + pqx, (q,p))
      XLT = Tensor('XLT' + pqx, (q,p))
      XRT = Tensor('XRT' + pqx, (p,q))
      XLTP = Tensor('XLTP' + pqx, (q,p), alignStride=True)
      XRTP = Tensor('XRTP' + pqx, (p,q), alignStride=True)
      YL = Tensor('YL' + pqx, (p,q))
      YR = Tensor('YR' + pqx, (q,p))
      ZL = Tensor('ZL' + pqx, (p,q))
      ZR = Tensor('ZR' + pqx, (q,p))

      stock = R['ijk'] <= S['xyz'] * XL['xl'] * XR['li'] * YL['ym'] * YR['mj'] * ZL['zn'] * ZR['nk']
      g.add('stock' + pqx, stock)

      stock = R['ijk'] <= S['xyz'] * XLT['lx'] * XRT['il'] * YL['ym'] * YR['mj'] * ZL['zn'] * ZR['nk']
      g.add('stock{}_trans'.format(pqx), stock)

      stock = R['ijk'] <= S['xyz'] * XLTP['lx'] * XRTP['il'] * YL['ym'] * YR['mj'] * ZL['zn'] * ZR['nk']
      g.add('stock{}_trans_pad'.format(pqx), stock)

