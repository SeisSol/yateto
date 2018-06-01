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
#~ multipleSims = True
multipleSims = False

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
simplify(test)
evaluate(test)
#~ equivalentSparsityPattern(test)


node = test[1][1]
import time
#~ t1 = time.time()
#~ optimalBinaryTree(node)
#~ t2 = time.time()
#~ print(t2-t1)
#~ 
#~ from yateto.tensorop.tools import optimalBinaryTree2
#~ t1 = time.time()
#~ optimalBinaryTree2(node)
#~ t2 = time.time()
#~ print(t2-t1)


from yateto.tensorop.tools import optimalBinaryTree3
t1 = time.time()
optimalBinaryTree3(node)
t2 = time.time()
print(t2-t1)





