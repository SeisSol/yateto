import string
from ..ast.indices import Indices, Range
from ..ast.node import IndexedTensor
from ..memory import DenseMemoryLayout
from .. import aspp
from .common import forLoops, INDEX_PREFIX, TensorDescription, IndexedTensorDescription, BatchedOperationsAux
from . import copyscaleadd, log, product, fused_gemms, elementwise, reduction
from ..type import Datatype, AddressingMode, Scalar, Tensor
from ..controlflow.graph import Guard
from ..ops import Add, Mul

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
    """`size` is an element count of `datatype` (bytes when datatype is None)."""
    assert(iniZero == False or len(memory) == 0)

    if datatype is None:
      datatype = Datatype.I8

    if self._target == 'cpu':
      # NOTE: onHeap() works on bytes, whereas size is an element count
      if self._arch.onHeap(size * datatype.size()):
        if len(self._freeList) == 0:
          self._cpp(f'int {self.ERROR_NAME};')
        self._cpp(f'{datatype.ctype()}* {bufname};')
        self._cpp(f'{self.ERROR_NAME} = posix_memalign(reinterpret_cast<void**>(&{bufname}), {self._arch.cacheline}, {size}*sizeof({datatype.ctype()}));')
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
        self._cpp(f'alignas({self._arch.cacheline}) {datatype.ctype()} {bufname}[{size}] {ini};')
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
    guard = Guard.coerce(condition)
    if guard.isAlways():
      return generate()
    if guard.isNever():
      return 0
    with self._cpp.If(f'({guard.ccode()})'):
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
      leftTerm = IndexedTensorDescription.fromNode(arguments[0], node[0]),
      rightTerm = IndexedTensorDescription.fromNode(arguments[1], node[1]),
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

  def create_Elementwise(self, node, result, arguments, condition, add, scalar, prefetchName, routineCache, gemm_cfg):
    # a binary product has its own backend, which can also unroll a CSC operand
    if node.optype == Mul() and len(node) == 2:
      return self._product(node, result, arguments, condition, add, scalar, routineCache, gemm_cfg)
    return self._elementwise(node, result, arguments, condition, add, scalar, routineCache, gemm_cfg)

  def _product(self, node, result, arguments, condition, add, scalar, routineCache, gemm_cfg):
    assert len(arguments) == 2
    description = product.Description(
      alpha = scalar,
      add = add,
      result = IndexedTensorDescription.fromNode(result, node),
      leftTerm = IndexedTensorDescription.fromNode(arguments[0], node[0]),
      rightTerm = IndexedTensorDescription.fromNode(arguments[1], node[1])
    )
    generator = product.generator(self._arch, description, self._target)
    return self._conditional(condition, lambda: generator.generate(self._cpp, routineCache))

  def _elementwise(self, node, result, arguments, condition, add, scalar, routineCache, gemm_cfg):
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

  def create_Permute(self, node, result, arguments, condition, add, scalar, prefetchName, routineCache, gemm_cfg):
    result = IndexedTensorDescription.fromNode(result, node)
    term = IndexedTensorDescription.fromNode(arguments[0], node.term())
    return self._csa(result, term, condition, add, scalar, routineCache, gemm_cfg)

  def create_Broadcast(self, node, result, arguments, condition, add, scalar, prefetchName, routineCache, gemm_cfg):
    result = IndexedTensorDescription.fromNode(result, node)
    term = IndexedTensorDescription.fromNode(arguments[0], node.term())
    return self._csa(result, term, condition, add, scalar, routineCache, gemm_cfg)

  def simple(self, result, term, condition, add, scalar, routineCache, gemm_cfg):
    result = IndexedTensorDescription.fromVar(result, self._indices(result))
    term = IndexedTensorDescription.fromVar(term, self._indices(term))
    return self._csa(result, term, condition, add, scalar, routineCache, gemm_cfg)

  def _csa(self, result, term, condition, add, scalar, routineCache, gemm_cfg):
    description = copyscaleadd.Description(
      alpha = scalar,
      beta = 1.0 if add else 0.0,
      result = result,
      term = term
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


  def create_Permute(self, node, result, arguments, condition, add, scalar, prefetchName, routineCache, gemm_cfg):
    assert node.indices <= node.term().indices and node.term().indices <= node.indices
    resultTerm = self._formatTerm(result, node.indices)
    termTerm = self._formatTerm(arguments[0], node.term().indices)
    return self._conditional(condition, lambda: self._simpleBody(resultTerm, termTerm, add, scalar, node.indices))

  def create_Broadcast(self, node, result, arguments, condition, add, scalar, prefetchName, routineCache, gemm_cfg):
    assert node.term().indices <= node.indices
    resultTerm = self._formatTerm(result, node.indices)
    termTerm = self._formatTerm(arguments[0], node.term().indices)
    return self._conditional(condition, lambda: self._simpleBody(resultTerm, termTerm, add, scalar, node.indices))

  def _product(self, node, result, arguments, condition, add, scalar, prefetchName, routineCache, gemm_cfg):
    # the loops below run over node.indices, so the address strings have to be
    # built from those very indices
    resultTerm = self._formatTerm(result, node.indices)

    argTerms = [self._formatTerm(argument, term.indices) for argument, term in zip(arguments, node)]
    termTerm = f'({argTerms[0]}) * ({argTerms[1]})'

    return self._conditional(condition, lambda: self._simpleBody(resultTerm, termTerm, add, scalar, node.indices))

  def create_Elementwise(self, node, result, arguments, condition, add, scalar, prefetchName, routineCache, gemm_cfg):
    if node.optype == Mul() and len(node) == 2:
      return self._product(node, result, arguments, condition, add, scalar, prefetchName, routineCache, gemm_cfg)

    resultTerm = self._formatTerm(result, node.indices)

    argTerms = [self._formatTerm(argument, term.indices) for argument, term in zip(arguments, node)]
    termTerm = node.optype.callstr(*node.fillTerms(argTerms))

    return self._conditional(condition, lambda: self._simpleBody(resultTerm, termTerm, add, scalar, node.indices))

  def create_Accumulate(self, node, result, arguments, condition, add, scalar, prefetchName, routineCache, gemm_cfg):
    resultTerm = self._formatTerm(result, node.indices)

    argTerms = [self._formatTerm(argument, term.indices) for argument, term in zip(arguments, node)]
    termTerm = argTerms[0]
    for argTerm in argTerms[1:]:
      termTerm = node.optype.callstr(termTerm, argTerm)

    return self._conditional(condition, lambda: self._simpleBody(resultTerm, termTerm, add, scalar, node.indices))

  def create_Reduction(self, node, result, arguments, condition, add, scalar, prefetchName, routineCache, gemm_cfg):
    resultTerm = self._formatTerm(result, node.indices)
    termTerm = self._formatTerm(arguments[0], node.term().indices)
    datatype = node.datatype

    # the reference implementation folds the reduced index explicitly
    def body():
      sumIndex = node.sumIndexName()
      size = node.term().indices.indexSize(sumIndex)
      accumulator = '_acc'
      init = f'{datatype.ctype()} {accumulator} = {node.optype.neutralLiteral(datatype)};'
      inner = f'{accumulator} = {node.optype.callstr(accumulator, termTerm)};'
      return self._simpleBody(resultTerm, accumulator, add, scalar, node.indices,
                              reduceIdx=(sumIndex, size, init, inner))

    return self._conditional(condition, body)

  def create_IfThenElse(self, node, result, arguments, condition, add, scalar, prefetchName, routineCache, gemm_cfg):
    resultTerm = self._formatTerm(result, node.indices)
    yesTerm = self._formatTerm(arguments[0], node.yesTerm().indices)
    noTerm = self._formatTerm(arguments[1], node.noTerm().indices)
    conditionTerm = self._formatTerm(arguments[2], node.condition().indices)

    termTerm = f'(({conditionTerm}) ? ({yesTerm}) : ({noTerm}))'

    return self._conditional(condition, lambda: self._simpleBody(resultTerm, termTerm, add, scalar, node.indices))

  def _simpleBody(self, resultTerm, termTerm, add, scalar, indices, reduceIdx = None):
    ranges = {idx: Range(0, indices.indexSize(idx)) for idx in indices}

    if scalar and scalar != 1.0:
      termTerm = f'{scalar} * {termTerm}'

    assign = '+=' if add else '='

    class AssignBody(object):
      def __call__(s):
        if reduceIdx is not None:
          # own scope for the accumulator, as several rank-0 reductions may
          # share one enclosing scope
          sumIndex, size, init, inner = reduceIdx
          with self._cpp.AnonymousScope():
            self._cpp(init)
            with self._cpp.For(f'int {INDEX_PREFIX}{sumIndex} = 0; {INDEX_PREFIX}{sumIndex} < {size}; ++{INDEX_PREFIX}{sumIndex}'):
              self._cpp(inner)
            self._cpp(f'{resultTerm} {assign} {termTerm};')
        else:
          self._cpp(f'{resultTerm} {assign} {termTerm};')
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
      # an all-zero reference (bool results, comparison kernels) would make the
      # relative error 0/0 == NaN, and NaN compares false against any epsilon
      self._cpp('if (refNorm == 0.0) { refNorm = 1.0; }')
      self._cpp(self._testFramework.assertLessThan('sqrt(error/refNorm)', epsMult*self._arch.epsilon))

  def tensor(self, node, resultName, maxValue = 512, scale = 1 / 512):
    ml = node.memoryLayout()
    size = ml.requiredReals()

    datatype = node.getDatatype(self._arch)

    spp = node.spp()
    isDense = spp.count_nonzero() == size
    if isDense:
      self.temporary(resultName, size, node.getDatatype(self._arch))
      with self._cpp.For(f'int i = 0; i < {size}; ++i'):
        if datatype.isBool():
          # alternate, so that both branches of a guard get exercised;
          # `(i + r) % maxValue + 1` is never zero and would be all-true
          self._cpp(f'{resultName}[i] = ((i + {self._rand}) % 2) == 0;')
        else:
          self._cpp(f'{resultName}[i] = static_cast<{datatype.ctype()}>((i + {self._rand}) % {maxValue} + 1);')
    else:
      memory = [datatype.literal(0)]*size
      nz = spp.nonzero()
      for entry in zip(*nz):
        addr = ml.address(entry)
        memory[addr] = datatype.literal(((addr + self._rand) % maxValue)+1.0)
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

  def add_tensor(self, description):
    pass

class ExportFactory(KernelFactory):
  @classmethod
  def makeFactory(cls, generator):
    return lambda cpp, arch, target: cls(generator(arch), cpp, arch, target)

  def __init__(self, generator, cpp, arch, target):
    super().__init__(cpp, arch, target)
    self.generator = generator
    self.tensors = {}
    self.scalarcounter = 0

  def post_generate(self, routine_cache):
    self.generator.generate(self._cpp, routine_cache)

  def allocateTemporary(self):
    return False

  def _nodeTensor(self, tensor, node):
    return self._handleTensorDesc(IndexedTensorDescription.fromNode(tensor, node))

  def _varTensor(self, var, indices):
    return self._handleTensorDesc(IndexedTensorDescription.fromVar(var, indices))

  def _handleAddressing(self, desc):
    if desc.addressing is None:
      addressing = BatchedOperationsAux.deduce_addresing(desc)
    else:
      addressing = desc.addressing

    # & == deref
    # n == current element
    # N == element size
    # o == extraOffset
    # *,+ == default add and mul
    # read left to right
    if addressing == AddressingMode.DIRECT:
      return '&'
    elif addressing == AddressingMode.STRIDED:
      return 'n*N+o&'
    elif addressing == AddressingMode.INDIRECT:
      return 'n&+o&'
    elif addressing == AddressingMode.SCALAR:
      return ''

    raise NotImplementedError(addressing)

  def _handleTensorDesc(self, tensorIndexed: IndexedTensorDescription):
    if isinstance(tensorIndexed.memoryLayout, DenseMemoryLayout):
      shape = list(tensorIndexed.memoryLayout.shape())
      shapeXt = [max(rng.stop - rng.start, shp) for rng, shp in zip(tensorIndexed.memoryLayout.bbox(), shape)]
      storage = {
        'shape': shapeXt,
        'type': 'bbox',
        'start': [rng.start for rng in tensorIndexed.memoryLayout.bbox()],
        'sizes': [rng.stop - rng.start for rng in tensorIndexed.memoryLayout.bbox()]
      }
    else:
      assert False

    # 0-d safe: rank-0 tensors (condition variables, scalar reduction results)
    # have a sparsity pattern too, but numpy refuses nonzero() on 0-d arrays
    spp = aspp.nonzeroIndices(tensorIndexed.eqspp)

    values = None if tensorIndexed.values is None else list(tensorIndexed.values)

    tensor = {
      'name': tensorIndexed.name,
      'addressing': self._handleAddressing(tensorIndexed),
      #'eqspp': spp,
      'datatype': str(tensorIndexed.datatype),
      'storage': storage,
      'values': values,
      'flags': {
        'temporary': tensorIndexed.is_temporary,
        'constant': tensorIndexed.is_compute_constant
      }
    }

    return self._handleTensor(tensor, spp, tensorIndexed.indices)

  def _scalarTensor(self, scalar):
    if isinstance(scalar, (int, float)): # TODO numpy types
      name = f'_scalar{self.scalarcounter}'
      self.scalarcounter += 1

      tensor = {
        'name': name,
        'addressing': '',
        'eqspp': (),
        'datatype': str(self._arch.datatype),
        'storage': {
          'shape': (),
          'type': 'full'
        },
        'values': {
          (): scalar
        },
        'flags': {
          'temporary': False,
          'constant': True
        }
      }
    elif isinstance(scalar, Tensor):
      tensor = {
        'name': scalar.name(),
        'addressing': '',
        'eqspp': (),
        'datatype': str(scalar.getDatatype(self._arch)),
        'storage': {
          'shape': (),
          'type': 'full'
        },
        'values': None,
        'flags': {
          'temporary': False,
          'constant': True
        }
      }
    else:
      assert False

    return self._handleTensor(tensor, (), ())

  def _handleTensor(self, tensor, eqspp, indices):
    if tensor['name'] not in self.tensors:
      self.tensors[tensor['name']] = tensor
      self.generator.add_tensor(tensor)
    else:
      assert tensor == self.tensors[tensor['name']]

    return {
      'name': tensor['name'],
      'spp': eqspp,
      'indices': indices
    }

  def _handleCondition(self, condition):
    """A guard exports as a flat conjunction of literals.

    `version` distinguishes successive values of the same condition tensor; two
    literals over the same tensor at different versions are different values.
    """
    guard = Guard.coerce(condition)
    if guard.isNever():
      return None
    return [{
      'tensor': self._varTensor(var, ()),
      'version': version,
      'negated': not polarity,
    } for var, version, polarity in guard.literals()]

  def create_Elementwise(self, node, result, arguments, condition, add, scalar, prefetchName, routineCache, gemm_cfg):
    result = self._nodeTensor(result, node)
    preArgs = [self._nodeTensor(argument, term) for argument, term in zip(arguments, node)]
    # immediate (non-Node) operands have to be exported as scalars, not raw values
    args = [arg if isinstance(arg, dict) else self._scalarTensor(arg)
            for arg in node.fillTerms(preArgs)]

    description = {
      'type': 'elementwise',
      'result': result,
      'args': args,
      'condition': self._handleCondition(condition),
      'linear': {
        'alpha': self._scalarTensor(scalar),
        'add': add,
      },
      'optype': str(node.optype)
    }
    return self.generator.add_operation(description)

  def create_Reduction(self, node, result, arguments, condition, add, scalar, prefetchName, routineCache, gemm_cfg):
    assert len(arguments) == 1
    result = self._nodeTensor(result, node)
    argnodes = [self._nodeTensor(arguments[0], node.term())]

    description = {
      'type': 'reduction',
      'result': result,
      'args': argnodes,
      'condition': self._handleCondition(condition),
      'linear': {
        'alpha': self._scalarTensor(scalar),
        'add': add,
      },
      'optype': str(node.optype)
    }
    return self.generator.add_operation(description)

  def create_LoopOverGEMM(self, node, result, arguments, condition, add, scalar, prefetchName, routineCache, gemm_cfg):
    assert len(arguments) == 2
    argnodes = [self._nodeTensor(arguments[0], node[0]), self._nodeTensor(arguments[1], node[1])]
    return self.handleLinear(self._nodeTensor(result, node), argnodes, condition, add, scalar, node.transA(), node.transB())


  def create_Permute(self, node, result, arguments, condition, add, scalar, prefetchName, routineCache, gemm_cfg):
    term = arguments[0]
    return self.handleLinear(self._varTensor(result, node.indices), [self._varTensor(term, node.term().indices)], condition, add, scalar, False, False)

  def create_Broadcast(self, node, result, arguments, condition, add, scalar, prefetchName, routineCache, gemm_cfg):
    term = arguments[0]
    return self.handleLinear(self._varTensor(result, node.indices), [self._varTensor(term, node.term().indices)], condition, add, scalar, False, False)

  def simple(self, result, term, condition, add, scalar, routineCache, gemm_cfg):
    return self.handleLinear(self._varTensor(result, self._indices(result)), [self._varTensor(term, self._indices(term))], condition, add, scalar, False, False)

  def getIndices(self, dest, ops):
    if dest is None:
      target_indices = []
    else:
      target_indices = dest['indices']

    indexindex = {index:i for i, index in enumerate(target_indices)}
    contract_counter = -1

    for op in ops:
      for index in op['indices']:
        if index not in indexindex:
          indexindex[index] = contract_counter
          contract_counter -= 1

    target = [[indexindex[index] for index in op['indices']] for op in ops]
    permute = [[i for i,_ in enumerate(op['indices'])] for op in ops]

    return target, permute

  def handleLinear(self, dest, ops, condition, add, scalar, transposeA, transposeB):
    # convert indices to loop numbers

    target, permute = self.getIndices(dest, ops)

    if not (scalar == 1 or scalar == 1.0):
      ops += [self._scalarTensor(scalar)]
      target += [[]]
      permute += [[]]

    description = {
      'type': 'multilinear',
      'result': dest,
      'args': ops,
      'condition': self._handleCondition(condition),
      'permute': permute,
      'target': target,
      'linear': {
        'alpha': self._scalarTensor(scalar),
        'add': add,
      },
      # 'optype': node.optype
    }
    return self.generator.add_operation(description)
