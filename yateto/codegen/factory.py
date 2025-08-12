import string
from ..ast.indices import Indices, Range
from ..ast.node import IndexedTensor
from ..memory import DenseMemoryLayout
from .common import forLoops, TensorDescription, IndexedTensorDescription, BatchedOperationsAux
from . import copyscaleadd, indexsum, log, product, fused_gemms, elementwise
from ..type import Datatype

class KernelFactory(object):
  ERROR_NAME = '_error'

  def __init__(self, cpp, arch, target):
    self._cpp = cpp
    self._arch = arch
    self._freeList = list()
    self._target = target
    
  def create(self, node, *args):
    method = 'create_' + node.__class__.__name__
    factory = getattr(self, method, self.generic_create)
    return factory(node, *args)
  
  def generic_create(self, node, *args):
    raise NotImplementedError

  def simple(self, result, term, condition, add, scalar, routineCache, gemm_cfg):
    raise NotImplementedError

  def temporary(self, bufname, size, datatype, iniZero=False, memory=list()):
    assert(iniZero == False or len(memory) == 0)

    if datatype is None:
      datatype = Datatype.I8

    if self._target == 'cpu':
      if self._arch.onHeap(size):
        if len(self._freeList) == 0:
          self._cpp(f'int {self.ERROR_NAME};')
        self._cpp(f'{datatype.ctype()}* {bufname};')
        self._cpp(f'{self.ERROR_NAME} = posix_memalign(reinterpret_cast<void**>(&{bufname}), {self._arch.alignment}, {size}*sizeof({datatype.ctype()}));')
        if iniZero:
          self._cpp.memset(bufname, size, datatype.ctype())
        if memory:
          for i, data in enumerate(memory):
            self._cpp(f'{bufname}[{i}] = {data};')
        self._freeList.append(bufname)
      else:
        ini = ''
        if iniZero:
          ini = ' = {}'
        elif memory:
          ini = ' = {{{}}}'.format(', '.join(memory))
        self._cpp(f'alignas({self._arch.alignment}) {datatype.ctype()} {bufname}[{size}] {ini};')
    else:
      declaration = f'{datatype.ctype()}* {bufname}'
      total_size = f'{BatchedOperationsAux.NUM_ELEMENTS_NAME} * {size}'
      self._cpp(f'{declaration} = linearAllocator.allocate({total_size});')

  def allocateTemporary(self):
    return True
  
  def post_generate(self, routine_cache):
    pass

  def freeTmp(self):
    if self._target == 'cpu':
      for free in self._freeList:
        self._cpp(f'free({free});')
    elif self._target == 'gpu':
      self._cpp('linearAllocator.free();')
    else:
      raise RuntimeError('unknown compute target')

    self._freeList = []

  def reset_stream(self):
    if self._target == 'cpu':
      pass
    elif self._target == 'gpu':
      self._cpp(f'{BatchedOperationsAux.STREAM_PTR_NAME} = {BatchedOperationsAux.FORBIDDEN_STREAM_PTR};')
    else:
      raise RuntimeError('unknown compute target')

  def reset_flags(self):
    if self._target == 'cpu':
      pass
    elif self._target == 'gpu':
      self._cpp(f'{BatchedOperationsAux.FLAGS_NAME} = nullptr;')
    else:
      raise RuntimeError('unknown compute target')

  def _indices(self, var):
    shape = var.memoryLayout().shape()
    return Indices(string.ascii_lowercase[:len(shape)], shape)
  
  def _conditional(self, condition, generate):
    if isinstance(condition, bool):
      if condition:
        return generate()
      else:
        return 0
    else:
      if condition.tautology():
        return generate()
      elif condition.unfulfillable():
        return 0
      else:
        with self._cpp.If(f'{condition.ccode()}'):
          return generate()

class OptimizedKernelFactory(KernelFactory):
  def __init__(self, cpp, arch, target):
    super().__init__(cpp, arch, target)

  def create_LoopOverGEMM(self, node, result, arguments, condition, add, scalar, prefetchName, routineCache, gemm_cfg):
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
    generator = log.generator(self._arch, description, self._target)
    return self._conditional(condition, lambda: generator.generate(self._cpp, routineCache, gemm_cfg))

  def create_FusedGEMMs(self, node, result, arguments, condition, add, scalar, prefetchName, routineCache, gemm_cfg):
    description = fused_gemms.Description(node, result, arguments, condition, add, scalar)
    generator = fused_gemms.generator(self._arch, description, gemm_cfg, self._target)
    return self._conditional(condition, lambda: generator.generate(self._cpp, routineCache, gemm_cfg))
  
  def create_IndexSum(self, node, result, arguments, condition, add, scalar, prefetchName, routineCache, gemm_cfg):
    assert len(arguments) == 1
    description = indexsum.Description(
      alpha = scalar,
      add = add,
      result = IndexedTensorDescription.fromNode(result, node),
      term = IndexedTensorDescription.fromNode(arguments[0], node.term())
    )
    generator = indexsum.generator(self._arch, description, self._target)
    return self._conditional(condition, lambda: generator.generate(self._cpp, routineCache))
  
  def create_Product(self, node, result, arguments, condition, add, scalar, prefetchName, routineCache, gemm_cfg):
    assert len(arguments) == 2
    description = product.Description(
      alpha = scalar,
      add = add,
      result = IndexedTensorDescription.fromNode(result, node),
      leftTerm = IndexedTensorDescription.fromNode(arguments[0], node.leftTerm()),
      rightTerm = IndexedTensorDescription.fromNode(arguments[1], node.rightTerm())
    )
    generator = product.generator(self._arch, description, self._target)
    return self._conditional(condition, lambda: generator.generate(self._cpp, routineCache))

  def create_Permute(self, node, result, arguments, condition, add, scalar, prefetchName, routineCache, gemm_cfg):
    term = arguments[0]
    description = copyscaleadd.Description(
      alpha = scalar,
      beta = 1.0 if add else 0.0,
      result = IndexedTensorDescription.fromVar(result, node.indices),
      term = IndexedTensorDescription.fromVar(term, node.term().indices)
    )
    generator = copyscaleadd.generator(self._arch, description, gemm_cfg, self._target)
    return self._conditional(condition, lambda: generator.generate(self._cpp, routineCache))
  
  def create_Elementwise(self, node, result, arguments, condition, add, scalar, prefetchName, routineCache, gemm_cfg):
    description = elementwise.Description(
      alpha = scalar,
      add = add,
      result = IndexedTensorDescription.fromNode(result, node),
      terms = [IndexedTensorDescription.fromNode(argument, term) for argument, term in zip(arguments, node)],
      optype = node.optype,
      termTemplate = node.termTemplate,
      nodeTermIndices = node.nodeTermIndices
    )
    generator = elementwise.generator(self._arch, description, self._target)
    return self._conditional(condition, lambda: generator.generate(self._cpp, routineCache))
  
  def create_Reduction(self, node, result, arguments, condition, add, scalar, prefetchName, routineCache, gemm_cfg):
    description = reduction.Description(
      alpha = scalar,
      add = add,
      result = IndexedTensorDescription.fromNode(result, node),
      term = IndexedTensorDescription.fromNode(arguments[0], node.term()),
      optype = node.optype,
    )
    generator = reduction.generator(self._arch, description, self._target)
    return self._conditional(condition, lambda: generator.generate(self._cpp, routineCache))

  def simple(self, result, term, condition, add, scalar, routineCache, gemm_cfg):
    description = copyscaleadd.Description(
      alpha = scalar,
      beta = 1.0 if add else 0.0,
      result = IndexedTensorDescription.fromVar(result, self._indices(result)),
      term = IndexedTensorDescription.fromVar(term, self._indices(term))
    )
    generator = copyscaleadd.generator(self._arch, description, gemm_cfg, self._target)
    return self._conditional(condition, lambda: generator.generate(self._cpp, routineCache))

class UnitTestFactory(KernelFactory):
  def __init__(self, cpp, arch, nameFun, testFramework):
    super().__init__(cpp, arch, target='cpu')
    self._name = nameFun
    self._rand = 0
    self._testFramework = testFramework

  def _formatTerm(self, var, indices):
    address = var.memoryLayout().addressString(indices)
    return f'{self._name(var)}[{address}]'
  
  def create_Einsum(self, node, result, arguments, condition, add, scalar, prefetchName, routineCache, gemm_cfg):
    g = node.indices
    for child in node:
      g = g.merged(child.indices - g)
    
    ranges = {idx: Range(0, g.indexSize(idx)) for idx in g}
    
    resultTerm = self._formatTerm(result, node.indices)
    terms = [self._formatTerm(arguments[i], child.indices) for i,child in enumerate(node)]
    
    if scalar and scalar != 1.0:
      terms.insert(0, str(scalar))
    
    if not add:
      self._cpp.memset(self._name(result), result.memoryLayout().requiredReals(), result.datatype.ctype())
    
    class EinsumBody(object):
      def __call__(s):
        self._cpp(f"{resultTerm} += {' * '.join(terms)};")
        return len(terms)

    return self._conditional(condition, lambda: forLoops(self._cpp, g, ranges, EinsumBody(), pragmaSimd=False))
  
  def create_ScalarMultiplication(self, node, result, arguments, condition, add, scalar, prefetchName, routineCache, gemm_cfg):
    return self._conditional(condition, lambda: self.simple(result, arguments[0], add, scalar, routineCache))

  def create_Permute(self, node, result, arguments, condition, add, scalar, prefetchName, routineCache, gemm_cfg):
    assert node.indices <= node.term().indices and node.term().indices <= node.indices
    resultTerm = self._formatTerm(result, node.indices)
    termTerm = self._formatTerm(arguments[0], node.term().indices)
    return self._conditional(condition, lambda: self._simpleBody(resultTerm, termTerm, add, scalar, node.indices))
  
  def create_Product(self, node, result, arguments, condition, add, scalar, prefetchName, routineCache, gemm_cfg):
    g = self._indices(result)
    resultTerm = self._formatTerm(result, node.indices)

    argTerms = [self._formatTerm(argument, term.indices) for argument, term in zip(arguments, node)]
    termTerm = f'({argTerms[0]}) * ({argTerms[1]})'

    return self._conditional(condition, lambda: self._simpleBody(resultTerm, termTerm, add, scalar, g))

  def create_Elementwise(self, node, result, arguments, condition, add, scalar, prefetchName, routineCache, gemm_cfg):
    g = self._indices(result)
    resultTerm = self._formatTerm(result, node.indices)

    argTerms = [self._formatTerm(argument, term.indices) for argument, term in zip(arguments, node)]
    termTerm = node.optype.callstr(*node.fillTerms(argTerms))

    return self._conditional(condition, lambda: self._simpleBody(resultTerm, termTerm, add, scalar, g))
  
  def create_Reduction(self, node, result, arguments, condition, add, scalar, prefetchName, routineCache, gemm_cfg):
    g = self._indices(result)
    resultTerm = self._formatTerm(result, node.indices)
    argTerm = self._formatTerm(arguments[0], node.term())

    termTerm = node.optype.callstr(*node.fillTerms(argTerms))

    return self._conditional(condition, lambda: self._simpleBody(resultTerm, termTerm, add, scalar, g))

  def create_IfThenElse(self, node, result, arguments, condition, add, scalar, prefetchName, routineCache, gemm_cfg):
    g = self._indices(result)
    resultTerm = self._formatTerm(result, node.indices)
    yesTerm = self._formatTerm(arguments[0], node.yesTerm().indices)
    noTerm = self._formatTerm(arguments[1], node.noTerm().indices)
    conditionTerm = self._formatTerm(arguments[2], node.condition().indices)

    termTerm = f'(({conditionTerm}) ? ({yesTerm}) : ({noTerm}))'

    return self._conditional(condition, lambda: self._simpleBody(resultTerm, termTerm, add, scalar, g))

  def _simpleBody(self, resultTerm, termTerm, add, scalar, indices, reduceIdx = None):
    ranges = {idx: Range(0, indices.indexSize(idx)) for idx in indices}

    if scalar and scalar != 1.0:
      termTerm = f'{scalar} * {termTerm}'

    class AssignBody(object):
      def __call__(s):
        self._cpp(f"{resultTerm} {'+=' if add else '='} {termTerm};")
        return 1 if add else 0

    return forLoops(self._cpp, indices, ranges, AssignBody(), pragmaSimd=False)

  def simple(self, result, term, condition, add, scalar, routineCache, gemm_cfg):
    g = self._indices(result)

    resultTerm = self._formatTerm(result, g)
    termTerm = self._formatTerm(term, g)

    return self._conditional(condition, lambda: self._simpleBody(resultTerm, termTerm, add, scalar, g))

  def compare(self, ref, target, epsMult = 100.0):
    g = self._indices(ref)
    refTerm = self._formatTerm(ref, g)
    targetTerm = self._formatTerm(target, g)

    class CompareBody(object):
      def __call__(s):
        self._cpp( f'double ref = {refTerm};' )
        self._cpp( f'double diff = ref - {targetTerm};' )
        self._cpp( 'error += diff * diff;' )
        self._cpp( 'refNorm += ref * ref;' )
        return 0

    targetBBox = target.memoryLayout().bbox()
    ranges = {idx: Range(targetBBox[i].start, min(targetBBox[i].stop, g.indexSize(idx))) for i,idx in enumerate(g)}
    with self._cpp.AnonymousScope():
      self._cpp('double error = 0.0;')
      self._cpp('double refNorm = 0.0;')
      forLoops(self._cpp, g, ranges, CompareBody(), pragmaSimd=False)
      self._cpp(self._testFramework.assertLessThan('sqrt(error/refNorm)', epsMult*self._arch.epsilon))

  def tensor(self, node, resultName, maxValue = 512):
    ml = node.memoryLayout()
    size = ml.requiredReals()

    datatype = node.getDatatype(self._arch)

    spp = node.spp()
    isDense = spp.count_nonzero() == size
    if isDense:
      self.temporary(resultName, size, node.getDatatype(self._arch))
      with self._cpp.For(f'int i = 0; i < {size}; ++i'):
        self._cpp(f'{resultName}[i] = static_cast<{datatype.ctype()}>((i + {self._rand}) % {maxValue} + 1);')
    else:
      memory = [datatype.literal(0)]*size
      nz = spp.nonzero()
      for entry in zip(*nz):
        addr = ml.address(entry)
        memory[addr] = str(float((addr + self._rand) % maxValue)+1.0)
      self.temporary(resultName, size, datatype, memory=memory)
    self._rand += 1

class ExportGenerator:
  INTERFACE_VERSION = 1

  def __init__(self, arch):
    self.arch = arch
  
  def generate(self, cpp, cache):
    pass
  
  def add_linear_operation(self, dest, ops, target, permute, add):
    pass
  
  def add_operation(self, description):
    pass

class ExportFactory(KernelFactory):
  @classmethod
  def makeFactory(cls, generator):
    return lambda cpp, arch, target: cls(generator(arch), cpp, arch, target)

  def __init__(self, generator, cpp, arch, target):
    super().__init__(cpp, arch, target)
    self.generator = generator
  
  def post_generate(self, routine_cache):
    self.generator.generate(self._cpp, routine_cache)

  def allocateTemporary(self):
    return False
  
  def create_Elementwise(self, node, result, arguments, condition, add, scalar, prefetchName, routineCache, gemm_cfg):
    result = IndexedTensorDescription.fromNode(result, node)
    preArgs = [IndexedTensorDescription.fromNode(argument, term) for argument, term in zip(arguments, node)]
    args = node.fillTerms(preArgs)

    description = {
      'type': 'elementwise',
      'result': result,
      'args': args,
      'linear': {
        'alpha': scalar,
        'add': add,
      },
      'optype': node.optype
    }
    return self.generator.add_operation(description)

  def create_Reduction(self, node, result, arguments, condition, add, scalar, prefetchName, routineCache, gemm_cfg):
    assert len(arguments) == 1
    makeNode = IndexedTensorDescription.fromNode
    result = makeNode(result, node)
    argnodes = [makeNode(arguments[0], node.term())]

    description = {
      'type': 'reduction',
      'result': result,
      'args': argnodes,
      'linear': {
        'alpha': scalar,
        'add': add,
      },
      'optype': node.optype
    }
    return self.generator.add_operation(description)

  def create_LoopOverGEMM(self, node, result, arguments, condition, add, scalar, prefetchName, routineCache, gemm_cfg):
    assert len(arguments) == 2
    makeNode = IndexedTensorDescription.fromNode
    argnodes = [makeNode(arguments[0], node.leftTerm()), makeNode(arguments[1], node.rightTerm())]
    return self.handleLinear(makeNode(result, node), argnodes, add, scalar, node.transA(), node.transB())
  
  def create_IndexSum(self, node, result, arguments, condition, add, scalar, prefetchName, routineCache, gemm_cfg):
    return create_Reduction(node, result, arguments, condition, add, scalar, prefetchName, routineCache, gemm_cfg)
  
  def create_Product(self, node, result, arguments, condition, add, scalar, prefetchName, routineCache, gemm_cfg):
    return create_Elementwise(node, result, arguments, condition, add, scalar, prefetchName, routineCache, gemm_cfg)

  def create_Permute(self, node, result, arguments, condition, add, scalar, prefetchName, routineCache, gemm_cfg):
    term = arguments[0]
    return self.handleLinear(IndexedTensorDescription.fromVar(result, node.indices), [IndexedTensorDescription.fromVar(term, node.term().indices)], add, scalar, False, False)
  
  def simple(self, result, term, condition, add, scalar, routineCache, gemm_cfg):
    return self.handleLinear(IndexedTensorDescription.fromVar(result, self._indices(result)), [IndexedTensorDescription.fromVar(term, self._indices(term))], add, scalar, False, False)

  def getIndices(self, dest, ops):
    if dest is None:
      target_indices = []
    else:
      target_indices = dest.indices

    indexindex = {index:i for i, index in enumerate(target_indices)}
    contract_counter = -1

    for op in ops:
      for index in op.indices:
        if index not in indexindex:
          indexindex[index] = contract_counter
          contract_counter -= 1

    target = [[indexindex[index] for index in op.indices] for op in ops]
    permute = [[i for i,_ in enumerate(op.indices)] for op in ops]

    return target, permute

  def handleLinear(self, dest, ops, add, scalar, transposeA, transposeB):
    # convert indices to loop numbers

    target, permute = self.getIndices(dest, ops)
    
    if not (scalar == 1 or scalar == 1.0):
      ops += [scalar]
      target += [[]]
      permute += [[]]
    
    return self.generator.add_linear_operation(dest, ops, target, permute, add)
