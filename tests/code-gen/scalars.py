#!/usr/bin/env python3

# Scalar arithmetic. Everything that reads only scalars is computed once, in the
# kernel prologue, ahead of every kernel and external routine call.

from yateto import *

import yateto.functions as yf

def add(g):
  N = 8

  A = Tensor('A', (N, N))
  B = Tensor('B', (N, N))
  C = Tensor('C', (N, N))
  s = Tensor('s', ())

  alpha = Scalar('alpha')
  beta  = Scalar('beta')
  gamma = Scalar('gamma')

  class Counter:
    def __init__(self):
      self.counter = 0

  counter = Counter()

  def _(kernel):
    counter.counter += 1
    g.add(f'kernel{counter.counter}', kernel)

  # 1: a plain named factor
  _(C['ij'] <= alpha * A['ij'])

  # 2: two numbers collapse while the tree is built
  _(C['ij'] <= 2.0 * (3.0 * A['ij']))

  # 3: two named scalars become one derived scalar
  _(C['ij'] <= alpha * (beta * A['ij']))

  # 4: a factor on each side of a product
  _(C['ij'] <= (alpha * A['ik']) * (beta * B['kj']))

  # 5: an expression built before it meets a tensor
  _(C['ij'] <= (alpha * beta + gamma) * A['ij'])

  # 6: division and subtraction
  _(C['ij'] <= ((alpha - beta) / gamma) * A['ij'])

  # 7: negation of a scaled term
  _(C['ij'] <= -(alpha * A['ij']))

  # 8: a factor folded into a contraction
  _(C['ij'] <= (alpha * beta) * A['ik'] * B['kj'])

  # 9: the same derived value used twice in one kernel
  _([
    C['ij'] <= (alpha * beta) * A['ij'],
    C['ij'] <= C['ij'] + (alpha * beta) * B['ij'],
  ])

  # 10: a scaled reduction to a rank-0 tensor
  _(s[''] <= alpha * yf.sum(A['ij'], 'ij'))

  # 11: a scalar factor under a guard
  flag = Tensor('flag', (), datatype=Datatype.BOOL)
  _(yf.assignIf(flag[''], C['ij'], (alpha / beta) * A['ij']))
