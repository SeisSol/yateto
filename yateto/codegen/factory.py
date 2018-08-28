from .common import TensorDescription, IndexedTensorDescription
from . import addition, indexsum, log, product

class Factory(object):
  def create(self, node, *args):
    method = 'create_' + node.__class__.__name__
    factory = getattr(self, method, self.generic_create)
    return factory(node, *args)
  
  def generic_create(self, node, *args):
    #~ raise NotImplementedError
    pass

class KernelFactory(Factory):
  def __init__(self, cpp, arch):
    self._cpp = cpp
    self._arch = arch

  def create_LoopOverGEMM(self, node, resultName, argNames, add):
    assert len(argNames) == 2
    description = log.Description(
      add = add,
      result = IndexedTensorDescription(resultName, node.indices, node.memoryLayout()),
      leftTerm = IndexedTensorDescription(argNames[0], node.leftTerm().indices, node.leftTerm().memoryLayout()),
      rightTerm = IndexedTensorDescription(argNames[1], node.rightTerm().indices, node.rightTerm().memoryLayout()),
      loopIndices = node.loopIndices(),
      transA = node.transA(),
      transB = node.transB()
    )
    generator = log.generator(self._arch, description)
    generator.generate(self._cpp)
  
  def create_IndexSum(self, node, resultName, argNames, add):
    assert len(argNames) == 1
    description = indexsum.Description(
      add = add,
      result = IndexedTensorDescription(resultName, node.indices, node.memoryLayout()),
      term = IndexedTensorDescription(argNames[0], node.term().indices, node.term().memoryLayout())
    )
    generator = indexsum.generator(self._arch, description)
    generator.generate(self._cpp)
  
  def create_Product(self, node, resultName, argNames, add):
    assert len(argNames) == 2
    description = product.Description(
      add = add,
      result = IndexedTensorDescription(resultName, node.indices, node.memoryLayout()),
      leftTerm = IndexedTensorDescription(argNames[0], node.leftTerm().indices, node.leftTerm().memoryLayout()),
      rightTerm = IndexedTensorDescription(argNames[1], node.rightTerm().indices, node.rightTerm().memoryLayout())
    )
    generator = product.generator(self._arch, description)
    generator.generate(self._cpp)
