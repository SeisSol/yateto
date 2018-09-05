from ..ast.node import IndexedTensor
from .common import TensorDescription, IndexedTensorDescription
from . import copyscaleadd, indexsum, log, product

class Factory(object):
  def create(self, node, *args):
    method = 'create_' + node.__class__.__name__
    factory = getattr(self, method, self.generic_create)
    return factory(node, *args)
  
  def generic_create(self, node, *args):
    raise NotImplementedError

class KernelFactory(Factory):
  def __init__(self, cpp, arch):
    self._cpp = cpp
    self._arch = arch

  def create_LoopOverGEMM(self, node, result, resultName, argNames, add, routineCache):
    assert len(argNames) == 2
    description = log.Description(
      add = add,
      result = IndexedTensorDescription.fromNode(resultName, result),
      leftTerm = IndexedTensorDescription.fromNode(argNames[0], node.leftTerm()),
      rightTerm = IndexedTensorDescription.fromNode(argNames[1], node.rightTerm()),
      loopIndices = node.loopIndices(),
      transA = node.transA(),
      transB = node.transB()
    )
    generator = log.generator(self._arch, description)
    generator.generate(self._cpp, routineCache)
  
  def create_IndexSum(self, node, result, resultName, argNames, add, routineCache):
    assert len(argNames) == 1
    description = indexsum.Description(
      add = add,
      result = IndexedTensorDescription.fromNode(resultName, result),
      term = IndexedTensorDescription.fromNode(argNames[0], node.term())
    )
    generator = indexsum.generator(self._arch, description)
    generator.generate(self._cpp, routineCache)
  
  def create_Product(self, node, result, resultName, argNames, add, routineCache):
    assert len(argNames) == 2
    description = product.Description(
      add = add,
      result = IndexedTensorDescription.fromNode(resultName, result),
      leftTerm = IndexedTensorDescription.fromNode(argNames[0], node.leftTerm()),
      rightTerm = IndexedTensorDescription.fromNode(argNames[1], node.rightTerm())
    )
    generator = product.generator(self._arch, description)
    generator.generate(self._cpp, routineCache)
  
  def create_Add(self, node, result, resultName, argNames, add, routineCache):
    beta = 1.0 if add else 0.0
    for i,child in enumerate(node):
      if isinstance(child, IndexedTensor):
        description = copyscaleadd.Description(
          alpha = 1.0,
          beta = beta,
          result = IndexedTensorDescription.fromNode(resultName, result),
          term = IndexedTensorDescription.fromNode(argNames[i], child),
        )
        generator = copyscaleadd.generator(self._arch, description)
        generator.generate(self._cpp, routineCache)
      beta = 1.0

  def create_Assign(self, node, result, resultName, argNames, add, routineCache):
    description = copyscaleadd.Description(
      alpha = 1.0,
      beta = 0.0,
      result = IndexedTensorDescription.fromNode(self._addArgument(argNames[0]), node.leftTerm()),
      term = IndexedTensorDescription.fromNode(self._addArgument(argNames[1]), node.rightTerm()),
    )
    generator = copyscaleadd.generator(self._arch, description)
    generator.generate(self._cpp, routineCache)
