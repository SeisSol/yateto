#!/usr/bin/env python3

from yateto import *

import yateto.functions as yf

def add(g):
  N = 8
  A0 = Tensor('A0', ())
  A1 = Tensor('A1', (N,))
  A2 = Tensor('A2', (N, N))

  AI0 = Tensor('AI0', (), datatype=Datatype.I32)
  AI1 = Tensor('AI1', (N,), datatype=Datatype.I32)
  AI2 = Tensor('AI2', (N, N), datatype=Datatype.I32)

  AB0 = Tensor('AB0', (), datatype=Datatype.BOOL)
  AB1 = Tensor('AB1', (N,), datatype=Datatype.BOOL)
  AB2 = Tensor('AB2', (N, N), datatype=Datatype.BOOL)

  class Counter:
    def __init__(self):
      self.counter = 0

  counter = Counter()

  def _(kernel):
    counter.counter += 1
    g.add(f'kernel{counter.counter}', kernel)

  _(A0[''] <= yf.sum(A1['i'], 'i'))
  _(A0[''] <= yf.sum(A2['ij'], 'ij'))
  _(A0[''] <= yf.min(A2['ij'], 'ij'))

  _(AI0[''] <= yf.sum(AI2['ij'], 'ij'))
  _(AI0[''] <= yf.min(AI2['ij'], 'ij'))
  _(AI0[''] <= yf.all(AI2['ij'], 'ij'))
  _(AI0[''] <= yf.any(AI2['ij'], 'ij'))

  _(AB0[''] <= yf.all(AB2['ij'], 'ij'))
  _(AB0[''] <= yf.any(AB2['ij'], 'ij'))

