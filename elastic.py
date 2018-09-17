#!/usr/bin/env python3

from yateto import Generator, Tensor
from yateto.generator import simpleParameterSpace
from yateto.input import parseXMLMatrixFile
from yateto.ast.visitor import *
from yateto.ast.transformer import *
from yateto.ast.node import Add
from yateto.codegen.code import Cpp
from yateto.codegen.cache import RoutineCache
from yateto.codegen.visitor import *
from yateto.arch import getArchitectureByIdentifier
import yateto.controlflow.visitor as cfv
import yateto.controlflow.transformer as cft
import itertools
import numpy as np

maxDegree = 1
order = maxDegree+1
numberOf2DBasisFunctions = order*(order+1)//2
numberOf3DBasisFunctions = order*(order+1)*(order+2)//6
numberOfQuantities = 9

arch = getArchitectureByIdentifier('dsnb')
DenseMemoryLayout.setAlignmentArch(arch)

multipleSims = True
transpose = True
#~ multipleSims = False
#~ transpose = False

if multipleSims:
  qShape = (8, numberOf3DBasisFunctions, numberOfQuantities)
  qi = lambda x: 's' + x
  alignStride=False
else:
  qShape = (numberOf3DBasisFunctions, numberOfQuantities)
  qi = lambda x: x
  alignStride=True

t = (lambda x: x[::-1]) if transpose else (lambda x: x)

clones = {
  'star': ['star[0]', 'star[1]', 'star[2]'],
}
db = parseXMLMatrixFile('matrices_{}.xml'.format(numberOf3DBasisFunctions), transpose=transpose, alignStride=alignStride)
db.update( parseXMLMatrixFile('star.xml'.format(numberOf3DBasisFunctions), clones) )

# Quantities
Q = Tensor('Q', qShape, alignStride=True)
I = Tensor('I', qShape, alignStride=True)
D = [Q]

# Flux solver
AplusT = [Tensor('AplusT[{}]'.format(dim), (numberOfQuantities, numberOfQuantities)) for dim in range(4)]
AminusT = [Tensor('AminusT[{}]'.format(dim), (numberOfQuantities, numberOfQuantities)) for dim in range(4)]

g = Generator(arch)

volumeSum = Q[qi('kp')]
for i in range(3):
  volumeSum += db.kDivM[i][t('kl')] * I[qi('lq')] * db.star[i]['qp']
volume = (Q[qi('kp')] <= volumeSum)
g.add('volume', volume)

localFluxSum = Q[qi('kp')]
for i in range(4):
  localFluxSum += db.rDivM[i][t('km')] * db.fMrT[i][t('ml')] * I[qi('lq')] * AplusT[i]['qp']
localFlux = (Q[qi('kp')] <= localFluxSum)
g.add('localFlux', localFlux)

neighbourFlux = lambda h,j,i: Q[qi('kp')] <= Q[qi('kp')] + db.rDivM[i][t('km')] * db.fP[h][t('mn')] * db.rT[j][t('nl')] * I[qi('lq')] * AminusT[i]['qp']
#~ g.addFamily('neighboringFlux', simpleParameterSpace(3,4,4), neighbourFlux)

derivatives = list()
for i in range(maxDegree):
  derivativeSum = Add()
  for j in range(3):
    derivativeSum += db.kDivMT[j][t('kl')] * D[i][qi('lq')] * db.star[j]['qp']
  derivativeSum = DeduceIndices( Q[qi('kp')].indices ).visit(derivativeSum)
  derivativeSum = EquivalentSparsityPattern().visit(derivativeSum)
  D.append( Tensor('dQ[{0}]'.format(i+1), qShape, spp=derivativeSum.eqspp(), alignStride=True) )
  derivative = D[i+1][qi('kp')] <= derivativeSum

  derivatives.append(derivative)
  g.add('derivative[{}]'.format(i), derivative)
  #~ g.add('derivative{}'.format(i), derivative)
  
  #~ derivative = DeduceIndices().visit(derivative)
  #~ derivative = EquivalentSparsityPattern().visit(derivative)
  #~ PrintEquivalentSparsityPatterns('sparsityPatterns/derivative{}/'.format(i)).visit(derivative)

X = Tensor('X', qShape)
Y = Tensor('Y', qShape)
g.add('test1', Q[qi('kp')] <= I[qi('kp')] + Q[qi('kp')])
g.add('test2', Q[qi('kp')] <= I[qi('kp')] + I[qi('kp')] + I[qi('kp')])
g.add('test3', Q[qi('kp')] <= db.kDivM[0][t('kl')] * I[qi('lp')] + db.kDivM[1][t('kl')] * Q[qi('lp')])
g.add('test4', I[qi('kp')] + I[qi('kp')] + Q[qi('kp')] + db.kDivM[0][t('kl')] * (X[qi('lq')] + Y[qi('lq')] + Q[qi('lq')]) * db.star[0]['qp'] + db.kDivM[1][t('kl')] * (X[qi('lq')] + Y[qi('lq')]) * db.star[1]['qp'])

g.generate('test/generated_code', 'seissol')
exit()

#~ PrintEquivalentSparsityPatterns('sparsityPatterns/volume/').visit(volume)
#~ PrintEquivalentSparsityPatterns('sparsityPatterns/localFlux/').visit(localFlux)

#~ nDof = 6
#~ nVar = 40
#~ A = Tensor('A', (nVar, nDof, nDof, nDof))
#~ B = Tensor('B', (nVar, nDof, nDof, nDof))
#~ C1 = Tensor('C1', (nDof, nDof))
#~ C2 = Tensor('C2', (nDof, nDof))
#~ C3 = Tensor('C3', (nDof, nDof))
#~ test = A['nxyz'] <= B['nijk'] * C1['ix'] * C2['jy'] * C3['kz']

#~ test = neighbourFlux(0,0,0)
#~ test = Tensor('D', (24,24,24,24,24))['abckl'] <= Tensor('A', (24,24,24,24))['ijmc'] * Tensor('B', (24,24,24,24))['mkab'] * Tensor('C', (24,24,24))['ijl']
#~ test = Tensor('D', (24,24,4,4,4))['abckl'] <= Tensor('A', (24,24,4,4))['ijmc'] * Tensor('B', (4,4,24,24))['mkab'] * Tensor('C', (24,24,4))['ijl']
#~ test = Tensor('D', (4,4,4,4,4,4))['abcijk'] <= Tensor('A', (4,4,6,4))['ijmc'] * Tensor('B', (6,4,4,4))['mkab']
#~ test = Tensor('D', (14,14,14,14,14,14))['abcijk'] <= Tensor('A', (14,14,14,14))['ijmc'] * Tensor('B', (14,14,14,14))['mkab']
#~ test = Tensor('D', (24,24,24,24,24,24))['abcijk'] <= Tensor('A', (24,24,24,24))['ijmc'] * Tensor('B', (24,24,24,24))['mkab']
#~ 
#~ spp = np.ones(shape=(4,4,4,4))
#~ spp[0,:,:,:] = 0
#~ spp[:,0,:,:] = 0
#~ test = Tensor('D', (4,4,4,4,4,4))['abcijk'] <= Tensor('A', (4,4,4,4))['ijmc'] * Tensor('B', (4,4,4,4))['mkab']

#~ test = Tensor('D', (4,4,4))['mij'] <= Tensor('A', (4,4))['ik'] * Tensor('B', (4,4))['kj'] * Tensor('C', (4,4))['ms']
#~ test = Tensor('D', (4,4,4,4,4))['hmnyj'] <= Tensor('F', (4,4,4))['hiy'] * Tensor('A', (4,4))['ki'] * Tensor('B', (4,4,4))['zkj'] * Tensor('C', (4,4,4))['msn']

#~ test = Tensor('Q', (4,4), alignStride=True)['ij'] <= Tensor('Q', (4,4), alignStride=True)['ij'] + Tensor('A', (4,4), alignStride=True)['ik'] * Tensor('B', (4,4))['kj']
#~ spp = np.ones((4,4,4), order='F')
#~ spp[0,:,:] = 0
#~ spp[:,0,:] = 0
#~ print(spp)
#~ test = Tensor('D', (4,4,4))['zij'] <= Tensor('B', (4,4,4),spp=spp)['zik'] * Tensor('C', (4,4))['kj']
#~ test = Tensor('Q', (4,4))['ij'] <= Tensor('B', (4,4), spp=spp)['ij']
#~ test = derivatives[4]
#~ test = volume
#~ test = Q[qi('kp')] <= I[qi('kp')] + Q[qi('kp')]
#~ test = Q[qi('kp')] <= I[qi('kp')] + I[qi('kp')] + I[qi('kp')]
#~ test = Q[qi('kp')] <= db.kDivM[0][t('kl')] * I[qi('lp')] + db.kDivM[1][t('kl')] * Q[qi('lp')]
#~ test = Q[qi('kp')] <= I[qi('kp')] + I[qi('kp')] + Q[qi('kp')] + db.kDivM[0][t('kl')] * (Tensor('A', qShape)[qi('lq')] + Tensor('B', qShape)[qi('lq')] + Q[qi('lq')]) * db.star[0]['qp'] + db.kDivM[1][t('kl')] * (Tensor('A', qShape)[qi('lq')] + Tensor('B', qShape)[qi('lq')]) * db.star[1]['qp']
test = localFlux
PrettyPrinter().visit(test)

test = DeduceIndices().visit(test)
#~ unitTest = copy.deepcopy(test)

test = EquivalentSparsityPattern().visit(test)
#~ PrettyPrinter().visit(test)

test = StrengthReduction().visit(test)
#~ PrettyPrinter().visit(test)

test = FindContractions().visit(test)
#~ PrettyPrinter().visit(test)

test = ComputeMemoryLayout().visit(test)
#~ PrettyPrinter().visit(test)

permutationVariants = FindIndexPermutations().visit(test)
test = SelectIndexPermutations(permutationVariants).visit(test)
#~ PrettyPrinter().visit(test)

test = ImplementContractions().visit(test)
#~ PrettyPrinter().visit(test)

PrettyPrinter().visit(test)

exit()

print('Initial CF')
ast2cf = cfv.AST2ControlFlow()
ast2cf.visit(test)
cfg = ast2cf.cfg()
cfv.PrettyPrinter().visit(cfg)

print('Find living')
cfg = cft.FindLiving().visit(cfg)
cfv.PrettyPrinter(True).visit(cfg)

print('Substitute forward')
cfg = cft.SubstituteForward().visit(cfg)
cfv.PrettyPrinter().visit(cfg)

print('Substitute backward')
cfg = cft.SubstituteBackward().visit(cfg)
cfv.PrettyPrinter().visit(cfg)

print('Remove empty statements')
cfg = cft.RemoveEmptyStatements().visit(cfg)
cfv.PrettyPrinter().visit(cfg)

print('Merge actions')
cfg = cft.MergeActions().visit(cfg)
cfv.PrettyPrinter().visit(cfg)

print('Reuse temporaries')
cfg = cft.ReuseTemporaries().visit(cfg)
cfv.PrettyPrinter().visit(cfg)

print('Determine local initialization')
cfg = cft.DetermineLocalInitialization().visit(cfg)
cfv.PrettyPrinter(True).visit(cfg)

#~ cache = RoutineCache()
#~ with Cpp() as cpp:
  #~ KernelGenerator(cpp, arch, cache).generate('test', test)
  #~ InitializerGenerator(cpp, arch).generate([Q, db.kDivM[0], db.kDivM[2], D[1], db.star[1]])
  #~ UnitTestGenerator(cpp, arch).generate('test', unitTest)
#~ 
#~ cache.generate('test/routines.cpp')
