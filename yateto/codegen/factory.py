from ..ast.indices import Range
from ..ast.node import IndexedTensor
from ..memory import DenseMemoryLayout
from .common import forLoops, TensorDescription, IndexedTensorDescription
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
    return generator.generate(self._cpp, routineCache)
  
  def create_IndexSum(self, node, result, resultName, argNames, add, routineCache):
    assert len(argNames) == 1
    description = indexsum.Description(
      add = add,
      result = IndexedTensorDescription.fromNode(resultName, result),
      term = IndexedTensorDescription.fromNode(argNames[0], node.term())
    )
    generator = indexsum.generator(self._arch, description)
    return generator.generate(self._cpp, routineCache)
  
  def create_Product(self, node, result, resultName, argNames, add, routineCache):
    assert len(argNames) == 2
    description = product.Description(
      add = add,
      result = IndexedTensorDescription.fromNode(resultName, result),
      leftTerm = IndexedTensorDescription.fromNode(argNames[0], node.leftTerm()),
      rightTerm = IndexedTensorDescription.fromNode(argNames[1], node.rightTerm())
    )
    generator = product.generator(self._arch, description)
    return generator.generate(self._cpp, routineCache)
  
  def create_Add(self, node, result, resultName, argNames, add, routineCache):
    beta = 1.0 if add else 0.0
    flops = 0
    for i,child in enumerate(node):
      if isinstance(child, IndexedTensor):
        description = copyscaleadd.Description(
          alpha = 1.0,
          beta = beta,
          result = IndexedTensorDescription.fromNode(resultName, result),
          term = IndexedTensorDescription.fromNode(argNames[i], child),
        )
        generator = copyscaleadd.generator(self._arch, description)
        flops += generator.generate(self._cpp, routineCache)
      beta = 1.0
    return flops

  def create_Assign(self, node, result, resultName, argNames, add, routineCache):
    description = copyscaleadd.Description(
      alpha = 1.0,
      beta = 0.0,
      result = IndexedTensorDescription.fromNode(self._addArgument(argNames[0]), node.leftTerm()),
      term = IndexedTensorDescription.fromNode(self._addArgument(argNames[1]), node.rightTerm()),
    )
    generator = copyscaleadd.generator(self._arch, description)
    return generator.generate(self._cpp, routineCache)

class UnitTestFactory(Factory):
  def __init__(self, cpp, arch):
    self._cpp = cpp
    self._arch = arch
  
  def _formatTerm(self, name, node):
    address = DenseMemoryLayout(node.indices.shape()).addressString(node.indices)
    return '{}[{}]'.format(name, address)
  
  def create_Einsum(self, node, resultName, argNames):
    g = node.indices
    for child in node:
      g = g.merged(child.indices - g)
    
    ranges = {idx: Range(0, g.indexSize(idx)) for idx in g}
    
    result = self._formatTerm(resultName, node)
    terms = [self._formatTerm(argNames[i], child) for i,child in enumerate(node)]
    
    class EinsumBody(object):
      def __call__(s):
        self._cpp( '{} += {};'.format(result, ' * '.join(terms)) )
        return 0

    forLoops(self._cpp, g, ranges, EinsumBody())

  def create_Add(self, node, resultName, argNames):
    g = node.indices
    ranges = {idx: Range(0, g.indexSize(idx)) for idx in g}
    
    result = self._formatTerm(resultName, node)
    terms = [self._formatTerm(argNames[i], child) for i,child in enumerate(node)]
    
    class AddBody(object):
      def __call__(s):
        self._cpp( '{} += {};'.format(result, ' + '.join(terms)) )
        return 0

    forLoops(self._cpp, g, ranges, AddBody())

  def create_Assign(self, node, resultName, argNames):
    g = node.indices
    ranges = {idx: Range(0, g.indexSize(idx)) for idx in g}
    
    result = self._formatTerm(argNames[0], node.leftTerm())
    term = self._formatTerm(argNames[1], node.rightTerm())
    
    class AssignBody(object):
      def __call__(s):
        self._cpp( '{} = {};'.format(result, term) )
        return 0

    forLoops(self._cpp, g, ranges, AssignBody())

    compareTerm = '{}[{}]'.format(resultName, node.leftTerm().memoryLayout().addressString(g))

    class CompareBody(object):
      def __call__(s):
        self._cpp( 'double diff = {} - {};'.format(result, compareTerm) )
        self._cpp( 'error += diff * diff;' )
        return 0

    targetBBox = node[0].memoryLayout().bbox()
    ranges = {idx: Range(targetBBox[i].start, min(targetBBox[i].stop, g.indexSize(idx))) for i,idx in enumerate(g)}
    self._cpp('double error = 0.0;')
    forLoops(self._cpp, g, ranges, CompareBody())
    self._cpp('TS_ASSERT_LESS_THAN(sqrt(error), {});'.format(self._arch.epsilon))

  def create_Tensor(self, node, resultName, argNames):
    ml = node.memoryLayout()
    size = ml.requiredReals()
    
    memory = ['0.0']*size
    nz = node.spp().nonzero()
    for entry in zip(*nz):
      addr = ml.address(entry)
      memory[addr] = str(float(addr))

    self._cpp('{} {}[{}] __attribute__((aligned({}))) = {{{}}};'.format(self._arch.typename, resultName, size, self._arch.alignment, ', '.join(memory)))
    #~ with self._cpp.For('int idx = 0; idx < {}; ++idx'.format(size)):
      #~ self._cpp( '{}[idx] = idx;'.format(resultName) )
