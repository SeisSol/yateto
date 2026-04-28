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

  class Counter:
    def __init__(self):
      self.counter = 0

  counter = Counter()

  def _(kernel):
    counter.counter += 1
    g.add(f'kernel{counter.counter}', kernel)

  _(A['ij'] <= yf.sqrt(B['ij']))
  _(A['ij'] <= yf.sqrt(B['ij']) + yf.sin(C['ij']))
  _(A['ij'] <= yf.sqrt(B['ij']) * yf.sin(C['ij']))
  _(A['ij'] <= yf.minimum(B['ij'], C['ij']))
  _(A['ij'] <= yf.minimum(B['ij'], C['ij'] + yf.atanh(B['ij'])))

  _(AI['ij'] <= BI['ij'] + CI['ij'])
  _(AI['ij'] <= yf.bitwise_and(BI['ij'], CI['ij']))

  _(AB['ij'] <= yf.greater_equal(BI['ij'], CI['ij']))
  _(A['ij'] <= yf.where(yf.greater_equal(BI['ij'], CI['ij']), B['ij'], C['ij']))

  _(AI['ij'] <= yf.cast(A['ij'], Datatype.I32))
