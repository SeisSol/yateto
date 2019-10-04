#!/usr/bin/env python3

from yateto import *
from yateto.input import parseXMLMatrixFile
from yateto.ast.node import Add
from yateto.ast.transformer import DeduceIndices, EquivalentSparsityPattern

def printEqspp():
  return True

def add(g):
  db = parseXMLMatrixFile('seissol_matrices.xml')
  
  Q = Tensor('Q', (8, 20, 15))
  I = Tensor('I', (8, 20, 15))
  g.add('seissol_stiffness', Q['skp'] <= db.kXiTDivM['lk'] * I['slq'] * db.star['qp'])

  # Reproduces recursive generation of zero blocks in Cauchy-Kowalevski prodedure,
  # described in "Sustained Petascale Performance of Seismic Simulations with SeisSol on SuperMUC",
  # Breuer et al., ISC 2014.
  dQ_shape = (20, 9)
  dQ0 = Tensor('dQ(0)', dQ_shape)
  star_ela = Tensor('star_ela', (9,9), spp=db.star['qp'].spp().as_ndarray()[0:9,0:9])
  dQ_prev = dQ0
  for i in range(1,4):
    derivativeSum = Add()
    for j in range(3):
      derivativeSum += db.kDivMT[j]['kl'] * dQ_prev['lq'] * star_ela['qp']
    derivativeSum = DeduceIndices('kp').visit(derivativeSum)
    derivativeSum = EquivalentSparsityPattern().visit(derivativeSum)
    dQ = Tensor('dQ({})'.format(i), dQ_shape, spp=derivativeSum.eqspp())
    g.add('derivative({})'.format(i), dQ['kp'] <= derivativeSum)
    dQ_prev = dQ
