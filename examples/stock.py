#!/usr/bin/env python3

from yateto import *

def gemm_cfg(arch, variant):
  if variant == 'onlyblas':
    return GeneratorCollection([MKL(arch)])
  return GeneratorCollection([LIBXSMM(arch), MKL(arch)])

def add(g):
  for ep in range(5):
    p = 4*(ep+2)
    px = '{}p'.format(p)
    for q in range(1,p//2+1):
      pqx = '{}p{}q'.format(p,q)
      R = Tensor('R' + px, (p,p,p))
      S = Tensor('S' + px, (p,p,p))
      XL = Tensor('XL' + pqx, (p,q))
      XR = Tensor('XR' + pqx, (q,p))
      XLT = Tensor('XLT' + pqx, (q,p))
      XRT = Tensor('XRT' + pqx, (p,q))
      YL = Tensor('YL' + pqx, (p,q))
      YR = Tensor('YR' + pqx, (q,p))
      ZL = Tensor('ZL' + pqx, (p,q))
      ZR = Tensor('ZR' + pqx, (q,p))

      stock = R['ijk'] <= S['xyz'] * XL['xl'] * XR['li'] * YL['ym'] * YR['mj'] * ZL['zn'] * ZR['nk']
      g.add('stock' + pqx, stock)

      stock = R['ijk'] <= S['xyz'] * XLT['lx'] * XRT['il'] * YL['ym'] * YR['mj'] * ZL['zn'] * ZR['nk']
      g.add('stock{}_trans'.format(pqx), stock)
