#!/usr/bin/env python3

from yateto import Generator, Tensor
from yateto.generator import simpleParameterSpace
from yateto.input import parseXMLMatrixFile
from yateto.ast.visitor import PrettyPrinter, PrintEquivalentSparsityPatterns
from yateto.ast.transformer import DeduceIndices, EquivalentSparsityPattern, StrengthReduction, FindContractions, ImplementContractions, FindIndexPermutations, SelectIndexPermutations
from yateto.ast.node import Add
import itertools

maxDegree = 5
order = maxDegree+1
numberOf2DBasisFunctions = order*(order+1)//2
numberOf3DBasisFunctions = order*(order+1)*(order+2)//6
numberOfQuantities = 9
multipleSims = True
#~ multipleSims = False

if multipleSims:
  qShape = (8, numberOf3DBasisFunctions, numberOfQuantities)
  qi = lambda x: 's' + x
else:
  qShape = (numberOf3DBasisFunctions, numberOfQuantities)
  qi = lambda x: x

clones = {
  'star': ['star[0]', 'star[1]', 'star[2]'],
}
db = parseXMLMatrixFile('matrices_{}.xml'.format(numberOf3DBasisFunctions), clones)

# Quantities
Q = Tensor('Q', qShape)
I = Tensor('I', qShape)
D = [Q]

# Flux solver
AplusT = [Tensor('AplusT[{}]'.format(dim), (numberOfQuantities, numberOfQuantities)) for dim in range(4)]
AminusT = [Tensor('AminusT[{}]'.format(dim), (numberOfQuantities, numberOfQuantities)) for dim in range(4)]

g = Generator()

volumeSum = Q[qi('kp')]
for i in range(3):
  volumeSum += db.kDivM[i]['kl'] * I[qi('lq')] * db.star[i]['qp']
volume = (Q[qi('kp')] <= volumeSum)
g.add('volume', volume)

localFluxSum = Q[qi('kp')]
for i in range(4):
  localFluxSum += db.rDivM[i]['km'] * db.fMrT[i]['ml'] * I[qi('lq')] * AplusT[i]['qp']
localFlux = (Q[qi('kp')] <= localFluxSum)
g.add('localFlux', localFlux)

neighbourFlux = lambda i,j,h: Q[qi('kp')] <= Q[qi('kp')] + db.rDivM[i]['km'] * db.fP[h]['mn'] * db.rT[j]['nl'] * I[qi('lq')] * AminusT[i]['qp']
g.addFamily('neighboringFlux', simpleParameterSpace(4,4,3), neighbourFlux)

for i in range(maxDegree):
  derivativeSum = Add()
  for j in range(3):
    derivativeSum += db.kDivMT[j]['kl'] * D[i][qi('lq')] * db.star[j]['qp']
  derivativeSum = DeduceIndices( Q[qi('kp')].indices ).visit(derivativeSum)
  derivativeSum = EquivalentSparsityPattern().visit(derivativeSum)
  D.append( Tensor('dQ[{0}]'.format(i+1), qShape, spp=derivativeSum.eqspp()) )
  derivative = D[i+1][qi('kp')] <= derivativeSum

  g.add('derivative[{}]'.format(i), derivative)
  
  #~ derivative = DeduceIndices().visit(derivative)
  #~ derivative = EquivalentSparsityPattern().visit(derivative)
  #~ PrintEquivalentSparsityPatterns('sparsityPatterns/derivative{}/'.format(i)).visit(derivative)

g.generate('test')

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
#~ test = Tensor('D', (4,4,4,4,4,4))['abcijk'] <= Tensor('A', (4,4,4,4))['ijmc'] * Tensor('B', (4,4,4,4))['mkab']

# TODO: Check strength reduction for this example
test = Tensor('D', (4,4,4))['mij'] <= Tensor('A', (4,4))['ik'] * Tensor('B', (4,4))['kj'] * Tensor('C', (4,4))['ms']
#~ test = volume
PrettyPrinter().visit(test)

test = DeduceIndices().visit(test)
PrettyPrinter().visit(test)

test = EquivalentSparsityPattern().visit(test)
PrettyPrinter().visit(test)

test = StrengthReduction().visit(test)
PrettyPrinter().visit(test)

test = FindContractions().visit(test)
PrettyPrinter().visit(test)

test = FindIndexPermutations().visit(test)
test = SelectIndexPermutations().visit(test)
PrettyPrinter().visit(test)

test = ImplementContractions().visit(test)
PrettyPrinter().visit(test)

exit()

#~ exit()
#~ equivalentSparsityPattern(test)
#~ pprint(test)
#~ exit()

#~ import yateto.ast
#~ c = yateto.ast.node.Contract()
#~ c.setChildren([AminusT[0]['qq']])
#~ optimalBinaryTree(c)
#~ exit()

#~ node = test[1][1]
#~ node = test[1]
#~ optimalBinaryTree(node)
#~ pprint(test)

#~ A = Tensor('A', (20,10))
#~ B = Tensor('B', (20,10))
#~ C = Tensor('C', ())
#~ test = C[''] <= A['ab'] * B['ab']
#~ simplify(test)
#~ evaluate(test)
#~ optimalBinaryTree(test[1])

def allSubstrings(s):
  L = len(s)
  return [s[i:j+1] for i in range(L) for j in range(i,L)]

def splitByDistance(p):
  L = len(p)
  splits = [i+1 for x,y,i in zip(p[:-1], p[1:], range(L)) if y-x != 1]
  return [p[i:j] for i,j in zip([0] + splits, splits + [L])]

def doTheStuff2(I, P, M, prune = False):
  D = list()
  indices = sorted([P[p] for p in I])
  groups = splitByDistance(indices)
  groupStrings = [''.join([M[p] for p in sorted(g)]) for g in groups]
  D = set([s for g in groupStrings for s in allSubstrings(g)])
  if prune:
    D = set([d for d in D if d[0] == M[0]])
  return D

def indexString(s, M):
  return ''.join([m if m in s else ':' for m in M])

def LoG(A, B, C):
  print(A,B,C)
  candidates = list()
  if set(C) != (set(A) | set(B)) - (set(A) & set(B)):
    return -1  
  requiredIndices = set([A[0], B[0], C[0]])
  if C[0] in set(B):
    B, A = A, B
  Im = set(A) & set(C)
  In = set(B) & set(C)
  Ik = set(A) & set(B)
  
  PA = {idx: pos for pos, idx in enumerate(A)}
  PB = {idx: pos for pos, idx in enumerate(B)}
  PC = {idx: pos for pos, idx in enumerate(C)}
  
  #~ doTheStuff(Im, In, PC, C)
  #~ doTheStuff(Im, Ik, PA, A)
  #~ doTheStuff(Ik, In, PB, B)
  CM = doTheStuff2(Im, PC, C, True)
  CN = doTheStuff2(In, PC, C)
  AM = doTheStuff2(Im, PA, A)
  AK = doTheStuff2(Ik, PA, A)
  BK = doTheStuff2(Ik, PB, B)
  BN = doTheStuff2(In, PB, B)
  
  MC = CM & AM
  NC = CN & BN
  KC = AK & BK
  
  for m in MC:
    for n in NC:
      for k in KC:
        Cstr = 'C_{}'.format(indexString(m+n, C))
        if A.find(m[0]) < A.find(k[0]):
          Astr = 'A_{}'.format(indexString(m+k, A))
          if A.find(m[0]) != 0:
            continue
        else:
          Astr = '(A_{})\''.format(indexString(m+k, A))
          if A.find(k[0]) != 0:
            continue
        if B.find(k[0]) < B.find(n[0]):
          Bstr = 'B_{}'.format(indexString(k+n, B))
          if B.find(k[0]) != 0:
            continue
        else:
          Bstr = '(B_{})\''.format(indexString(k+n, B))
          if B.find(n[0]) != 0:
            continue
        print(len(m) + len(n) + len(k), '{} = {} {}'.format(Cstr, Astr, Bstr))
  
  
    
  #~ if len(requiredIndices & Im) != 1 or len(requiredIndices & In) >= 2 or len(requiredIndices & Ik) >= 2:
    #~ return -1
  #~ allIndices = Im | In | Ik
  #~ mIdx = C[0]
  #~ nChoice = list(In) if len(requiredIndices & In) != 1 else list(requiredIndices & In)
  #~ kChoice = list(Ik) if len(requiredIndices & Ik) != 1 else list(requiredIndices & Ik)
  #~ for nIdx in nChoice:
    #~ for kIdx in kChoice:
      #~ gemmIndices = [mIdx, nIdx, kIdx]
      #~ loopIndices = allIndices - set(gemmIndices)
      #~ print(gemmIndices, loopIndices)
  
  return candidates
  
#~ LoG('slq','ln','snq')
#~ LoG('slq','qp','slp')
#~ print(LoG('l','slq','sq'))
#~ print(LoG('sab','sab','sab'))
#~ print(LoG('ijmc','mkab','abcijk'))
#~ LoG('qlosr','osrk','qlk')

#~ LoG('ijmc', 'ijl', 'mcl')
#~ LoG('ijmc', 'ijl', 'mlc')
#~ LoG('ijmc', 'ijl', 'clm')
#~ LoG('ijmc', 'ijl', 'cml')
#~ LoG('ijmc', 'ijl', 'lmc')
#~ LoG('ijmc', 'ijl', 'lcm')
#~ print('============================')
#~ LoG('mkab', 'mcl', 'abckl')
#~ LoG('mkab', 'mlc', 'abckl')
#~ LoG('mkab', 'clm', 'abckl')
#~ LoG('mkab', 'cml', 'abckl')
#~ LoG('mkab', 'lmc', 'abckl')
#~ LoG('mkab', 'lcm', 'abckl')

#~ print('1')
#~ LoG('nijk', 'ix', 'nxkj')
#~ print('2')
#~ LoG('nxkj', 'jy', 'nxyk')
#~ print('3')
#~ LoG('nxyk', 'kz', 'nxyz')
print('1')
LoG('ijmc', 'ijl', 'mlc')
print('2')
LoG('mkab', 'mlc', 'abckl')

import yateto.ast.log as log
log.LoG('ijmc', 'ijl', 'mlc')
