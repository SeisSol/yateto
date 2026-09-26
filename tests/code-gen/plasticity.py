#!/usr/bin/env python3

# The shape a plasticity kernel takes: a first computation produces the
# condition, a second one runs only where it holds. The condition tensor is
# written inside the kernel and may be rewritten, and it may itself be produced
# under an outer condition.

from yateto import *

import yateto.functions as yf

def add(g):
  N = 8

  S  = Tensor('S',  (N, N))
  Y  = Tensor('Y',  (N, N))
  Z  = Tensor('Z',  (N, N))
  o1 = Tensor('o1', (N, N))
  o2 = Tensor('o2', (N, N))
  o3 = Tensor('o3', (N, N))

  # global condition tensors
  yielded = Tensor('yielded', (), datatype=Datatype.BOOL)
  # temporary condition tensors: these compete for the same buffers as the
  # ordinary intermediates
  flag  = Tensor('flag',  (), datatype=Datatype.BOOL, temporary=True)
  inner = Tensor('inner', (), datatype=Datatype.BOOL, temporary=True)
  spare = Tensor('spare', (N, N), temporary=True)

  class Counter:
    def __init__(self):
      self.counter = 0

  counter = Counter()

  def _(kernel):
    counter.counter += 1
    g.add(f'kernel{counter.counter}', kernel)

  # 1: condition computed in the kernel, then used
  _([
    flag[''] <= yf.any(yf.greater(S['ij'], Y['ij']), 'ij'),
    yf.assignIf(flag[''], o1['ij'], yf.sqrt(S['ij'])),
  ])

  # 2: the condition tensor is rewritten between two uses, so the two guards
  #    are over different values
  _([
    flag[''] <= yf.any(yf.greater(S['ij'], Y['ij']), 'ij'),
    yf.assignIf(flag[''], o1['ij'], yf.sqrt(S['ij'])),
    flag[''] <= yf.any(yf.greater(S['ij'], Z['ij']), 'ij'),
    yf.assignIf(flag[''], o2['ij'], yf.sqrt(Z['ij'])),
  ])

  # 3: nested -- `inner` only has a value where `flag` holds
  _([
    flag[''] <= yf.any(yf.greater(S['ij'], Y['ij']), 'ij'),
    yf.assignIf(flag[''], inner[''], yf.all(yf.greater(Z['ij'], Y['ij']), 'ij')),
    yf.assignIf(inner[''], o3['ij'], yf.sqrt(Z['ij'])),
  ])

  # 4: buffer pressure between the definition of the condition and its use
  _([
    flag[''] <= yf.any(yf.greater(S['ij'], Y['ij']), 'ij'),
    spare['ij'] <= yf.minimum(S['ij'], Z['ij']),
    o1['ij'] <= spare['ij'],
    yf.assignIf(flag[''], o2['ij'], yf.sqrt(S['ij'])),
  ])

  # 5: accumulation into a conditionally written target
  _([
    yielded[''] <= yf.any(yf.greater(S['ij'], Y['ij']), 'ij'),
    yf.assignIf(yielded[''], o1['ij'], o1['ij'] + S['ik'] * Z['kj']),
  ])

  # 6: a guard around a GEMM, and an unguarded statement in between that must
  #    not inherit the guard
  _([
    yielded[''] <= yf.any(yf.greater(S['ij'], Y['ij']), 'ij'),
    yf.assignIf(yielded[''], o1['ij'], S['ik'] * Z['kj']),
    o2['ij'] <= yf.sqrt(Y['ij']),
    yf.assignIf(yielded[''], o3['ij'], yf.cos(Z['ij'])),
  ])
