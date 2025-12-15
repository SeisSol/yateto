#!/usr/bin/env python3

from yateto import *

def add(g):
  N = 8
  A = Tensor('A', (N,))
  B = Tensor('B', (N,))
  B2 = Tensor('B2', (N, N))
  C = Tensor('C', (N, N))
  C2 = Tensor('C2', (N, N, N))
  C3 = Tensor('C3', (N, N, N, N, N, N))

  class Counter:
    def __init__(self):
      self.counter = 0

  counter = Counter()

  def _(kernel):
    counter.counter += 1
    g.add(f'kernel{counter.counter}', kernel)

  _(C['ab'] <= A['a'])
  _(C['ab'] <= A['a'] + B['a'])
  _(C['ab'] <= A['a'] + B['b'])
  _(C2['abc'] <= A['a'] + B['b'])
  _(C2['abc'] <= A['a'] + B['b'] + B2['ba'])
  _(C2['abc'] <= B2['ba'] + B2['ab'])
  _(C2['abc'] <= B2['ab'] + B2['bc'] + B2['ca'])
  _(C3['abcdef'] <= C2['def'] + A['a'] + A['b'] + B2['ba'] + B2['cd'])
