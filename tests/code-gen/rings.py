#!/usr/bin/env python3

# The same structural expression -- a reduction over a product -- on different
# rings. Only (*, +) is a contraction and reaches the GEMM backends; every other
# ring is generated as loops.

from yateto import *

import yateto.functions as yf

def add(g):
  N = 8

  A  = Tensor('A',  (N, N))
  B  = Tensor('B',  (N, N))
  C  = Tensor('C',  (N, N))
  v  = Tensor('v',  (N,))
  s  = Tensor('s',  ())
  alpha = Scalar('alpha')

  AI = Tensor('AI', (N, N), datatype=Datatype.I32)
  BI = Tensor('BI', (N, N), datatype=Datatype.I32)
  CI = Tensor('CI', (N, N), datatype=Datatype.I32)

  AB = Tensor('AB', (N, N), datatype=Datatype.BOOL)
  BB = Tensor('BB', (N, N), datatype=Datatype.BOOL)
  CB = Tensor('CB', (N, N), datatype=Datatype.BOOL)

  class Counter:
    def __init__(self):
      self.counter = 0

  counter = Counter()

  def _(kernel):
    counter.counter += 1
    g.add(f'kernel{counter.counter}', kernel)

  # 1: the arithmetic ring -- a contraction
  _(C['ij'] <= A['ik'] * B['kj'])

  # 2: the boolean semiring, or over and
  _(CB['ij'] <= yf.any(yf.bitwise_and(AB['ik'], BB['kj']), 'k'))

  # 3: the boolean semiring, and over or
  _(CB['ij'] <= yf.all(yf.bitwise_or(AB['ik'], BB['kj']), 'k'))

  # 4: xor over and, on integers
  _(CI['ij'] <= yf.reduction(ops.Xor(), yf.bitwise_and(AI['ik'], BI['kj']), 'k'))

  # 5: the tropical semiring, min over plus
  _(C['ij'] <= yf.min(yf.add(A['ik'], B['kj']), 'k'))

  # 6: max over times
  _(C['ij'] <= yf.max(yf.mul(A['ik'], B['kj']), 'k'))

  # 7: a contraction with a scale factor, which folds into the GEMM
  _(C['ij'] <= alpha * A['ik'] * B['kj'])

  # 8: a chain, so that the contraction search has something to choose from
  _(C['ij'] <= A['ik'] * B['kl'] * A['lj'])

  # 9: a contraction against a vector
  _(v['i'] <= A['ij'] * v['j'])

  # 10: a full contraction down to a scalar
  _(s[''] <= A['ij'] * B['ij'])

  # 11: an element-wise product without a reduction on top
  _(C['ij'] <= yf.mul(A['ij'], B['ij']))

  # 12: the arithmetic ring on integers -- a contraction, but not one BLAS runs
  _(CI['ij'] <= AI['ik'] * BI['kj'])
