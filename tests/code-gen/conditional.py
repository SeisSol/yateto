#!/usr/bin/env python3

from yateto import *

import yateto.functions as yf

def add(g):
  N = 8
  A = Tensor('A', (N, N))
  B = Tensor('B', (N, N))
  C = Tensor('C', (N, N))

  AI = Tensor('AI', (N, N), datatype=Datatype.I32)
  BI = Tensor('BI', (N, N), datatype=Datatype.I32)
  CI = Tensor('CI', (N, N), datatype=Datatype.I32)

  AB = Tensor('AB', (N, N), datatype=Datatype.BOOL)

  X = Tensor('X', (), datatype=Datatype.BOOL)
  X1 = Tensor('X1', (), datatype=Datatype.BOOL)
  X2 = Tensor('X2', (), datatype=Datatype.BOOL)
  X3 = Tensor('X3', (), datatype=Datatype.BOOL)

  class Counter:
    def __init__(self):
      self.counter = 0

  counter = Counter()

  def _(kernel):
    counter.counter += 1
    g.add(f'kernel{counter.counter}', kernel)

  _(yf.assignIf(X[''], A['ij'], yf.sqrt(B['ij'])))
  _(yf.assignIf(yf.all(AB['ij'], 'ij'), A['ij'], yf.sqrt(B['ij'])))
  _([
    yf.assignIf(X[''], A['ij'], yf.sqrt(B['ij'])),
    yf.assignIf(X[''], AI['ij'], -BI['ij'])
    ])
  _([
    yf.assignIf(X1[''], A['ij'], B['ik'] * C['kj'] + C['ij']),
    yf.assignIf(X1[''], A['ij'], A['ij'] + B['ik'] * C['kj'] + C['ij']),
    yf.assignIf(X2[''], C['ij'], yf.sqrt(B['ij']))
  ])
