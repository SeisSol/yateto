#!/usr/bin/env python3

from yateto import *
from yateto.ast.node import Add

def add(g):
  M = 32
  K = 40
  A = Tensor('A', (M, K))
  B = Tensor('B', (M, K))
  C = Tensor('C', (M, K))

  class Counter:
    def __init__(self):
      self.counter = 0

  counter = Counter()

  def _(kernel):
    counter.counter += 1
    g.add(f'kernel{counter.counter}', kernel)

  # list bugs with their PR solving them here

  # #103.1
  # allow one-element sum accumulations
  _(B['ij'] <= Add() + A['ij'])

  # #103.2
  # prevent overriding a global variable when action merging
  _([
    B['ij'] <= A['ij'],
    C['ij'] <= B['ij']
  ])
