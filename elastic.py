#!/usr/bin/env python3

from yateto import Generator, Tensor
from yateto.generator import simpleParameterSpace
from yateto.input import parseXMLMatrixFile
from yateto.ast.node import Add
from yateto.ast.tools import evaluate, equivalentSparsityPattern, pprint, simplify
from yateto.tensorop.tools import optimalBinaryTree
import itertools

maxDegree = 5
order = maxDegree+1
numberOf2DBasisFunctions = order*(order+1)//2
numberOf3DBasisFunctions = order*(order+1)*(order+2)//6
numberOfQuantities = 9
multipleSims = True
#~ multipleSims = False

if multipleSims:
  qShape = (2, numberOf3DBasisFunctions, numberOfQuantities)
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
D.extend([Tensor('dQ[{0}]'.format(i), qShape) for i in range(1, order)])

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

def nextDerivative(i):
  derivativeSum = Add()
  for j in range(3):
    derivativeSum += db.kDivMT[j]['kl'] * D[i][qi('lq')] * db.star[j]['qp']
  return D[i+1][qi('kp')] <= derivativeSum

g.addFamily('derivative', simpleParameterSpace(maxDegree), nextDerivative)

#~ simplify(derivative[3])
#~ evaluate(derivative[3])
#~ pprint(derivative[3])

#~ simplify(volume)
#~ evaluate(volume)
#~ equivalentSparsityPattern(volume)
#~ pprint(volume)

g.generate('test')

test = neighbourFlux(0,0,0)
#~ test = nextDerivative(0)
#~ pprint(test)
simplify(test)
#~ pprint(test)
evaluate(test)
#~ pprint(test)
#~ equivalentSparsityPattern(test)
#~ pprint(test)
#~ exit()

#~ import yateto.ast
#~ c = yateto.ast.node.Contract()
#~ c.setChildren([AminusT[0]['qq']])
#~ optimalBinaryTree(c)
#~ exit()

node = test[1][1]
optimalBinaryTree(node)
pprint(test)

A = Tensor('A', (20,10))
B = Tensor('B', (20,10))
C = Tensor('C', ())
test = C[''] <= A['ab'] * B['ab']
simplify(test)
evaluate(test)
optimalBinaryTree(test[1])

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
        print(len(m)*len(n)*len(k), '{} = {} {}'.format(Cstr, Astr, Bstr))
  
  
    
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
LoG('slq','qp','slp')
#~ print(LoG('l','slq','sq'))
#~ print(LoG('sab','sab','sab'))
#~ print(LoG('ijmc','mkab','abcijk'))
#~ LoG('qlosr','osrk','qlk')


