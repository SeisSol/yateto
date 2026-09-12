#!/usr/bin/env python3

import numpy as np

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

  # Constant tensors, one per element type, so that the pool has to store
  # entries of more than one type and init has to bind a reference of the
  # matching type to each of them. Same numbers in two types on purpose:
  # that is the case where an entry keyed on the numbers alone would be
  # shared between two arrays that cannot share one.
  eye = np.eye(N)
  VF = Tensor('VF', (N, N), eye)
  VI = Tensor('VI', (N, N), eye, datatype=Datatype.I32)
  VB = Tensor('VB', (N, N), eye, datatype=Datatype.BOOL)

  class Counter:
    def __init__(self):
      self.counter = 0

  counter = Counter()

  def _(kernel):
    counter.counter += 1
    g.add(f'kernel{counter.counter}', kernel)

  _(AI['ij'] <= yf.cast(A['ij'], Datatype.I32))

  _(C['ij'] <= VF['ik'] * A['kj'])
  _(CI['ij'] <= VI['ik'] * AI['kj'])
  _(AB['ij'] <= yf.cast(VB['ij'], Datatype.BOOL))
