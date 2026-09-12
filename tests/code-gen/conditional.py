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

  # a negated guard: the case that leaves the destination alone is the one
  # where the condition holds
  _(yf.assignIf(yf.logical_not(X['']), A['ij'], yf.sqrt(B['ij'])))

  # two conditions on one statement -- three of the four assignments have to
  # leave it alone
  _(yf.assignIf(yf.bitwise_and(X1[''], X2['']), A['ij'], yf.sqrt(B['ij'])))

  # one condition and its negation: exactly one of the two runs, whichever
  # way the condition falls, so the destination is written in both cases
  _([
    yf.assignIf(X[''], A['ij'], yf.sqrt(B['ij'])),
    yf.assignIf(yf.logical_not(X['']), A['ij'], yf.sin(B['ij'])),
  ])

  # three conditions, so eight assignments, and a statement under each pair
  _([
    yf.assignIf(X1[''], A['ij'], yf.sqrt(B['ij'])),
    yf.assignIf(X2[''], A['ij'], A['ij'] + C['ij']),
    yf.assignIf(X3[''], C['ij'], yf.sqrt(B['ij'])),
  ])

  # a guarded statement whose result a later, unguarded one reads: the
  # unguarded statement runs in every case, over whatever the guard left
  _([
    yf.assignIf(X[''], C['ij'], yf.sqrt(B['ij'])),
    A['ij'] <= C['ik'] * C['kj'],
  ])

  # a guard over a contraction that needs a temporary, so that the branch
  # not taken leaves a buffer untouched rather than a destination
  _(yf.assignIf(X[''], A['ij'], B['ik'] * C['kl'] * B['lj']))

  # the condition is written by the kernel, so the statements around the
  # write read two different values under one name
  _([
    X1[''] <= yf.all(yf.greater_equal(BI['ij'], CI['ij']), 'ij'),
    yf.assignIf(X1[''], A['ij'], yf.sqrt(B['ij'])),
  ])
