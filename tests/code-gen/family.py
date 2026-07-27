#!/usr/bin/env python3

from yateto import *

def add(g):
  N = 4
  A = Tensor("A", (N, N))
  B = Tensor("B", (N, N))
  C = Tensor("C", (N, N))

  def build(i):
      return C["ij"] <= A["ik"] * B["kj"]

  g.addFamily("family0p", simpleParameterSpace(), build)
  g.addFamily("family1p", simpleParameterSpace(2), build)
  g.addFamily("family2p", simpleParameterSpace(2, 3), build)
  g.addFamily("family3p", simpleParameterSpace(2, 3, 4), build)

  g.addFamily("family1px", parameterSpaceFromRanges(range(1, 10, 2)), build)
  g.addFamily("family2px", parameterSpaceFromRanges(range(1, 10, 2), range(10, 20, 3)), build)
