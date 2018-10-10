import string
from numpy import count_nonzero
from ..ast.indices import Indices, Range
from ..ast.node import IndexedTensor
from ..memory import DenseMemoryLayout
from .common import forLoops, TensorDescription, IndexedTensorDescription
from . import copyscaleadd, indexsum, log, product

class KernelFactory(object):
  def __init__(self, cpp, arch):
    self._cpp = cpp
    self._arch = arch
    
  def create(self, node, *args):
    method = 'create_' + node.__class__.__name__
    factory = getattr(self, method, self.generic_create)
    return factory(node, *args)
  
  def generic_create(self, node, *args):
    raise NotImplementedError

  def simple(self, result, term, add, scalar, routineCache):
    raise NotImplementedError

  def _indices(self, var):
    shape = var.memoryLayout().shape()
    return Indices(string.ascii_lowercase[:len(shape)], shape)

class OptimisedKernelFactory(KernelFactory):
  def __init__(self, cpp, arch):
    super().__init__(cpp, arch)

  def create_LoopOverGEMM(self, node, result, arguments, add, scalar, prefetchName, routineCache):
    assert len(arguments) == 2
    description = log.Description(
      alpha = scalar,
      add = add,
      result = IndexedTensorDescription.fromNode(result, node),
      leftTerm = IndexedTensorDescription.fromNode(arguments[0], node.leftTerm()),
      rightTerm = IndexedTensorDescription.fromNode(arguments[1], node.rightTerm()),
      loopIndices = node.loopIndices(),
      transA = node.transA(),
      transB = node.transB(),
      prefetchName = prefetchName
    )
    generator = log.generator(self._arch, description)
    return generator.generate(self._cpp, routineCache)
  
  def create_IndexSum(self, node, result, arguments, add, scalar, prefetchName, routineCache):
    assert len(arguments) == 1
    description = indexsum.Description(
      alpha = scalar,
      add = add,
      result = IndexedTensorDescription.fromNode(result, node),
      term = IndexedTensorDescription.fromNode(arguments[0], node.term())
    )
    generator = indexsum.generator(self._arch, description)
    return generator.generate(self._cpp, routineCache)
  
  def create_Product(self, node, result, arguments, add, scalar, prefetchName, routineCache):
    assert len(arguments) == 2
    description = product.Description(
      alpha = scalar,
      add = add,
      result = IndexedTensorDescription.fromNode(result, node),
      leftTerm = IndexedTensorDescription.fromNode(arguments[0], node.leftTerm()),
      rightTerm = IndexedTensorDescription.fromNode(arguments[1], node.rightTerm())
    )
    generator = product.generator(self._arch, description)
    return generator.generate(self._cpp, routineCache)
  
  def simple(self, result, term, add, scalar, routineCache):
    description = copyscaleadd.Description(
      alpha = scalar,
      beta = 1.0 if add else 0.0,
      result = IndexedTensorDescription(str(result), self._indices(result), result.memoryLayout(), result.eqspp()),
      term = IndexedTensorDescription(str(term), self._indices(term), term.memoryLayout(), term.eqspp())
    )
    generator = copyscaleadd.generator(self._arch, description)
    return generator.generate(self._cpp, routineCache)

class UnitTestFactory(KernelFactory):
  def __init__(self, cpp, arch, nameFun):
    super().__init__(cpp, arch)
    self._name = nameFun

  def _formatTerm(self, var, indices):
    address = var.memoryLayout().addressString(indices)
    return '{}[{}]'.format(self._name(var), address)
  
  def create_Einsum(self, node, result, arguments, add, scalar, prefetchName, routineCache):
    g = node.indices
    for child in node:
      g = g.merged(child.indices - g)
    
    ranges = {idx: Range(0, g.indexSize(idx)) for idx in g}
    
    resultTerm = self._formatTerm(result, node.indices)
    terms = [self._formatTerm(arguments[i], child.indices) for i,child in enumerate(node)]
    
    if scalar and scalar != 1.0:
      terms.insert(0, str(scalar))
    
    if not add:
      self._cpp.memset(self._name(result), result.memoryLayout().requiredReals(), self._arch.typename)
    
    class EinsumBody(object):
      def __call__(s):
        self._cpp( '{} += {};'.format(resultTerm, ' * '.join(terms)) )
        return len(terms)

    return forLoops(self._cpp, g, ranges, EinsumBody())
  
  def create_ScalarMultiplication(self, node, result, arguments, add, scalar, prefetchName, routineCache):
    return self.simple(result, arguments[0], add, scalar, routineCache)

  def simple(self, result, term, add, scalar, routineCache):
    g = self._indices(result)
    
    ranges = {idx: Range(0, g.indexSize(idx)) for idx in g}
    
    resultTerm = self._formatTerm(result, g)
    termTerm = self._formatTerm(term, g)

    if scalar and scalar != 1.0:
      termTerm = '{} * {}'.format(scalar, termTerm)
    
    class AssignBody(object):
      def __call__(s):
        self._cpp( '{} {} {};'.format(resultTerm, '+=' if add else '=', termTerm) )
        return 1 if add else 0

    return forLoops(self._cpp, g, ranges, AssignBody())

  def compare(self, ref, target, epsMult = 10.0):
    g = self._indices(ref)
    refTerm = self._formatTerm(ref, g)
    targetTerm = self._formatTerm(target, g)

    class CompareBody(object):
      def __call__(s):
        self._cpp( 'double ref = {};'.format(refTerm) )
        self._cpp( 'double diff = ref - {};'.format(targetTerm) )
        self._cpp( 'error += diff * diff;' )
        self._cpp( 'refNorm += ref * ref;' )
        return 0

    targetBBox = target.memoryLayout().bbox()
    ranges = {idx: Range(targetBBox[i].start, min(targetBBox[i].stop, g.indexSize(idx))) for i,idx in enumerate(g)}
    self._cpp('double error = 0.0;')
    self._cpp('double refNorm = 0.0;')
    forLoops(self._cpp, g, ranges, CompareBody())
    self._cpp('TS_ASSERT_LESS_THAN(sqrt(error/refNorm), {});'.format(epsMult*self._arch.epsilon))

  def tensor(self, node, resultName, maxValue = 512):
    ml = node.memoryLayout()
    size = ml.requiredReals()

    spp = node.spp()
    isDense = count_nonzero(spp) == size
    if isDense:
      memory = list()
    else:
      memory = ['0.0']*size
      nz = spp.nonzero()
      for entry in zip(*nz):
        addr = ml.address(entry)
        memory[addr] = str(float(addr % maxValue)+1.0)

    self._cpp('{} {}[{}] __attribute__((aligned({}))) = {{{}}};'.format(self._arch.typename, resultName, size, self._arch.alignment, ', '.join(memory)))
    if isDense:
      with self._cpp.For('int i = 0; i < {}; ++i'.format(size)):
        self._cpp('{}[i] = static_cast<{}>(i % {} + 1);'.format(resultName, self._arch.typename, maxValue))
