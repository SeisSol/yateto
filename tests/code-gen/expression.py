#!/usr/bin/env python3

from yateto import *

def add(g):
  n = 32
  A = Tensor('A', (n, n))
  B = Tensor('B', (n, n))
  C = Tensor('C', (n, n))
  a = Scalar('a')

  g.add('axpy', C['ij'] <= A['ij'] + a * B['ij'])
  g.add('axpyNeg', C['ij'] <= A['ij'] - a * B['ij'])

  # TODO: add some more simple expression tests here

