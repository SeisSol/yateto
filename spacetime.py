#!/usr/bin/env python3

from yateto import Generator, Tensor
from yateto.generator import simpleParameterSpace
from yateto.input import parseXMLMatrixFile
from yateto.arch import getArchitectureByIdentifier
from yateto.ast.visitor import PrettyPrinter
from yateto.codegen.code import Cpp
from yateto.codegen.visitor import *

maxDegree = 5
order = maxDegree+1
numberOf1DBasisFunctions = order
numberOf2DBasisFunctions = order*(order+1)//2
numberOf3DBasisFunctions = order*(order+1)*(order+2)//6
numberOfQuantities = 9
arch = getArchitectureByIdentifier('dsnb')
DenseMemoryLayout.setAlignmentArch(arch)

clones = {
  'star': ['star[0]', 'star[1]', 'star[2]'],
}
db = parseXMLMatrixFile('matrices_{}.xml'.format(numberOf3DBasisFunctions), alignStride=True)
db.update( parseXMLMatrixFile('star.xml'.format(numberOf3DBasisFunctions), clones) )

# Quantities
Q0 = Tensor('Q0', (numberOf3DBasisFunctions, numberOfQuantities), alignStride=True)
Q = Tensor('Q', (numberOf3DBasisFunctions, numberOfQuantities, numberOf1DBasisFunctions), alignStride=True)
I = Tensor('I', (numberOf3DBasisFunctions, numberOfQuantities, numberOf1DBasisFunctions), alignStride=True)

chi0 = Tensor('chi0', (numberOf1DBasisFunctions,))
Z = Tensor('Z', (numberOf1DBasisFunctions, numberOf1DBasisFunctions))

g = Generator(arch)

spaceTimeIteration = I['mou'] <= chi0['u'] * Q0['mo'] + Z['ku'] * (db.kDivM[0]['ml'] * Q['lqk'] * db.star[0]['qo'] + db.kDivM[1]['ml'] * Q['lqk'] * db.star[1]['qo'] + db.kDivM[2]['ml'] * Q['lqk'] * db.star[2]['qo'])
g.add('spaceTimeIteration', spaceTimeIteration)

nDof = 6
nVar = 40
A = Tensor('A', (nVar, nDof, nDof, nDof))
B = Tensor('B', (nVar, nDof, nDof, nDof))
C1 = Tensor('C1', (nDof, nDof))
C2 = Tensor('C2', (nDof, nDof))
C3 = Tensor('C3', (nDof, nDof))

gridProjection = A['pxyz'] <= B['pijk'] * C1['ix'] * C2['jy'] * C3['kz']
g.add('gridProjection', gridProjection)

g.generate('test/generated_code', 'seissol')

print('SeisSol space-time predictor')
PrettyPrinter().visit(spaceTimeIteration)
print()
print('Grid projection')
PrettyPrinter().visit(gridProjection)
