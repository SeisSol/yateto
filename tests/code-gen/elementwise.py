#!/usr/bin/env python3

from yateto import *

import numpy as np
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

  # A constant with zeros in it patterns narrower than the axis it sits on, and
  # so does everything computed from it. Where such an operand ends, the
  # generated code has to read a literal zero rather than its neighbour.
  trace = Tensor('trace', (N,), np.array([1.0] * 3 + [0.0] * (N - 3)))
  v = Tensor('v', (N,))

  _(A['ij'] <= yf.add(yf.mul(v['i'], trace['j']), yf.mul(v['i'], B['ij'])))
  _(A['ij'] <= yf.add(v['i'], trace['j']))
  _(A['ij'] <= yf.mul(v['i'], trace['j']))
  _(A['ij'] <= yf.maximum(B['ij'], trace['j']))
  _(A['ij'] <= yf.exp(trace['j']) * v['i'])
  _(A['ij'] <= yf.where(yf.greater(trace['j'], 0.5 * trace['j']), B['ij'], C['ij']))

  # A zero scale factor: a zero fill, and nothing at all under accumulation.
  _(A['ij'] <= 0.0 * B['ij'])
  _(A['ij'] <= yf.maximum(B['ij'], 0.0 * C['ij']))
  _(A['ij'] <= yf.where(yf.greater(B['ij'], C['ij']), B['ij'], 0.0 * C['ij']))
