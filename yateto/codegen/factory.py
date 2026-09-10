import inspect
import string
from ..ast.indices import BoundingBox, Indices, Range
from ..ast.node import IndexedTensor
from ..memory import DenseMemoryLayout, CSCMemoryLayout, PatternMemoryLayout, MemoryLayoutView
from .. import aspp
from .common import forLoops, loopRanges, INDEX_PREFIX, TensorDescription, IndexedTensorDescription, BatchedOperationsAux, KernelAttributes
from . import copyscaleadd, log, fused_gemms, elementwise, reduction
from ..type import Datatype, AddressingMode, Scalar, Tensor
from ..guard import Guard
from .. import ir
from ..ops import Add, Mul

class KernelFactory(object):
  ERROR_NAME = '_error'

  def __init__(self, cpp, arch, target, attrs=None):
    self._cpp = cpp
    self._arch = arch
    self._freeList = list()
    self._target = target
    #: The attributes of the kernel being generated. Every generator that
    #: emits a call into an external kernel needs them, because the flags
    #: member such a call would name only exists when the kernel declares it.
    self._attrs = attrs if attrs is not None else KernelAttributes()

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

  def optimizes(self):
    """Whether the kernel this builds is meant to be fast.

    The reference implementation of a unit test is not: it is there to be
    obviously right, and a pass that rearranged it would be one more thing
    for the comparison to be wrong about.
    """
    return False

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
      # Nothing to reset where the kernel has no flags member.
      if self._attrs.flags:
        self._cpp(f'{BatchedOperationsAux.FLAGS_NAME} = nullptr;')
    else:
      raise RuntimeError('unknown compute target')

  def _indices(self, var):
    shape = var.memoryLayout().shape()
    return Indices(string.ascii_lowercase[:len(shape)], shape)

  def _conditional(self, condition, statement, touches=(), writes=()):
    """One statement of the kernel, as a region, guarded where it is guarded.

    `statement` is the region the statement lowers to, or a callable for a
    generator that still writes itself -- which becomes a call, since what
    such a generator does is its own to write and its own to report. What it
    touches is stated here either way, because a call that has not been asked
    may touch anything, and then nothing can be said about the storage the
    kernel needs.
    """
    guard = Guard.coerce(condition)
    if guard.isNever():
      return ir.Region()
    region = statement if isinstance(statement, ir.Region) \
             else ir.Region([ir.Call(lambda cpp, cache: statement(),
                                     reads=[self._buffer(term) for term in touches],
                                     writes=[self._buffer(term) for term in writes])])
    if guard.isAlways():
      return region
    self._checkGuardIsReadable(guard)
    return ir.Region([ir.If(f'({guard.ccode()})', region)])

  @staticmethod
  def _buffer(term):
    """The buffer behind a tensor description or a control-flow variable.

    The two spell the same things differently -- one states its layout, the
    other answers when asked -- and a statement is described with whichever
    of them its generator was handed.
    """
    layout = term.memoryLayout
    eqspp = term.eqspp
    return ir.Buffer(term.name, term.datatype,
                     layout() if callable(layout) else layout,
                     eqspp() if callable(eqspp) else eqspp,
                     getattr(term, 'is_temporary', False))

  @staticmethod
  def _statement(generator, write, *lowering):
    """The region a generator makes, or the call that still writes it."""
    lower = getattr(generator, 'lower', None)
    return lower(*lowering) if lower is not None else write

  def _checkGuardIsReadable(self, guard):
    """A guard emitted here is read on the host, so it has to live there.

    On a device target a tensor argument is a pointer per batch element, which
    the host cannot dereference -- and reading it as a plain pointer would
    silently be true. A scalar is passed by value and is fine; a per-element
    decision belongs in the kernel and goes through the external generator,
    which receives the guard as data.
    """
    if self._target != 'gpu':
      return
    for var in guard.variables():
      if not var.isPassedByValue():
        raise NotImplementedError(
          f'"{var}" guards a statement on a device target, but it is passed by '
          f'pointer and cannot be read on the host. Use a Scalar for a decision '
          f'that is uniform over the batch, or an external generator for one '
          f'that is not.')

class OptimizedKernelFactory(KernelFactory):
  def __init__(self, cpp, arch, target, attrs=None):
    super().__init__(cpp, arch, target, attrs)

  def optimizes(self):
    # on a device a statement is a kernel launch, and putting two of them
    # together is the external generator's business
    return self._target == 'cpu'

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
    generator = log.generator(self._arch, description, self._target, self._attrs)
    return self._conditional(condition, self._statement(
      generator, lambda: generator.generate(self._cpp, routineCache, gemm_cfg), gemm_cfg),
      touches=[description.result, description.leftTerm, description.rightTerm],
      writes=[description.result])

  def create_FusedGEMMs(self, node, result, arguments, condition, add, scalar, prefetchName, routineCache, gemm_cfg):
    description = fused_gemms.Description(node, result, arguments, condition, add, scalar)
    generator = fused_gemms.generator(self._arch, description, gemm_cfg, self._target,
                                      self._attrs)
    return self._conditional(condition,
                             lambda: generator.generate(self._cpp, routineCache, gemm_cfg),
                             touches=[result] + list(arguments), writes=[result])

  def create_Elementwise(self, node, result, arguments, condition, add, scalar, prefetchName, routineCache, gemm_cfg):
    return self._elementwise(node, result, arguments, condition, add, scalar, routineCache, gemm_cfg)

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
    return self._conditional(condition, self._statement(
      generator, lambda: generator.generate(self._cpp, routineCache)),
      touches=[description.result] + list(description.terms),
      writes=[description.result])

  def create_Reduction(self, node, result, arguments, condition, add, scalar, prefetchName, routineCache, gemm_cfg):
    description = reduction.Description(
      alpha = scalar,
      add = add,
      result = IndexedTensorDescription.fromNode(result, node),
      term = IndexedTensorDescription.fromNode(arguments[0], node.term()),
      optype = node.optype,
    )
    generator = reduction.generator(self._arch, description, self._target)
    return self._conditional(condition, self._statement(
      generator, lambda: generator.generate(self._cpp, routineCache)),
      touches=[description.result, description.term],
      writes=[description.result])

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
    generator = copyscaleadd.generator(self._arch, description, gemm_cfg, self._target,
                                       self._attrs)
    return self._conditional(condition, self._statement(
      generator, lambda: generator.generate(self._cpp, routineCache)),
      touches=[description.result, description.term],
      writes=[description.result])

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

    class EinsumBody(object):
      def __call__(s):
        self._cpp(f"{resultTerm} += {' * '.join(terms)};")
        return len(terms)

    def statement():
      # the zeroing belongs to this statement, so it is written where the
      # statement is and not where the statement was built
      if not add:
        self._cpp.memset(self._name(result), result.memoryLayout().requiredReals(), result.datatype.ctype())
      return forLoops(self._cpp, g, ranges, EinsumBody(), pragmaSimd=False)

    return self._conditional(condition, statement,
                             touches=[result] + list(arguments), writes=[result])


  def create_Permute(self, node, result, arguments, condition, add, scalar, prefetchName, routineCache, gemm_cfg):
    assert node.indices <= node.term().indices and node.term().indices <= node.indices
    resultTerm = self._formatTerm(result, node.indices)
    termTerm = self._formatTerm(arguments[0], node.term().indices)
    return self._conditional(
      condition, lambda: self._simpleBody(resultTerm, termTerm, add, scalar, node.indices),
      touches=[result] + list(arguments), writes=[result])

  def create_Broadcast(self, node, result, arguments, condition, add, scalar, prefetchName, routineCache, gemm_cfg):
    assert node.term().indices <= node.indices
    resultTerm = self._formatTerm(result, node.indices)
    termTerm = self._formatTerm(arguments[0], node.term().indices)
    return self._conditional(
      condition, lambda: self._simpleBody(resultTerm, termTerm, add, scalar, node.indices),
      touches=[result] + list(arguments), writes=[result])

  def create_Elementwise(self, node, result, arguments, condition, add, scalar, prefetchName, routineCache, gemm_cfg):
    # the loops below run over node.indices, so the address strings have to be
    # built from those very indices
    resultTerm = self._formatTerm(result, node.indices)

    argTerms = [self._formatTerm(argument, term.indices) for argument, term in zip(arguments, node)]
    termTerm = node.optype.callstr(*node.fillTerms(argTerms))

    return self._conditional(
      condition, lambda: self._simpleBody(resultTerm, termTerm, add, scalar, node.indices),
      touches=[result] + list(arguments), writes=[result])

  def create_Accumulate(self, node, result, arguments, condition, add, scalar, prefetchName, routineCache, gemm_cfg):
    resultTerm = self._formatTerm(result, node.indices)

    argTerms = [self._formatTerm(argument, term.indices) for argument, term in zip(arguments, node)]
    termTerm = argTerms[0]
    for argTerm in argTerms[1:]:
      termTerm = node.optype.callstr(termTerm, argTerm)

    return self._conditional(
      condition, lambda: self._simpleBody(resultTerm, termTerm, add, scalar, node.indices),
      touches=[result] + list(arguments), writes=[result])

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

    return self._conditional(condition, body,
                             touches=[result] + list(arguments), writes=[result])

  def create_IfThenElse(self, node, result, arguments, condition, add, scalar, prefetchName, routineCache, gemm_cfg):
    resultTerm = self._formatTerm(result, node.indices)
    yesTerm = self._formatTerm(arguments[0], node.yesTerm().indices)
    noTerm = self._formatTerm(arguments[1], node.noTerm().indices)
    conditionTerm = self._formatTerm(arguments[2], node.condition().indices)

    termTerm = f'(({conditionTerm}) ? ({yesTerm}) : ({noTerm}))'

    return self._conditional(
      condition, lambda: self._simpleBody(resultTerm, termTerm, add, scalar, node.indices),
      touches=[result] + list(arguments), writes=[result])

  def _simpleBody(self, resultTerm, termTerm, add, scalar, indices, reduceIdx = None):
    ranges = {idx: Range(0, indices.indexSize(idx)) for idx in indices}

    if scalar and scalar != 1.0:
      # parenthesised: `*` binds tighter than the operators an operation may
      # spell itself with, so the factor would otherwise land on one operand
      termTerm = f'{scalar} * ({termTerm})'

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

    return self._conditional(
      condition, lambda: self._simpleBody(resultTerm, termTerm, add, scalar, g),
      touches=[result, term], writes=[result])

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

  def tensor(self, node, resultName, maxValue = 512, scale = 1 / 512,
             caseVar = None, caseBit = None):
    ml = node.memoryLayout()
    size = ml.requiredReals()

    datatype = node.getDatatype(self._arch)
    span = self._valueSpan(datatype, maxValue)

    if caseVar is not None:
      # A condition the kernel is guarded by. Its value is what distinguishes
      # one run of the test from the next, so it comes from the case index
      # rather than from the filling pattern -- which would pick one
      # assignment and never leave it.
      self.temporary(resultName, size, datatype)
      with self._cpp.For(f'int i = 0; i < {size}; ++i'):
        self._cpp(f'{resultName}[i] = (({caseVar} >> {caseBit}) & 1) != 0;')
      self._rand += 1
      return

    spp = node.spp()
    isDense = spp.count_nonzero() == size
    if isDense:
      self.temporary(resultName, size, datatype)
      with self._cpp.For(f'int i = 0; i < {size}; ++i'):
        self._cpp(f'{resultName}[i] = {self._valueExpression(datatype, "i", span, scale)};')
    else:
      memory = [datatype.literal(0)]*size
      nz = spp.nonzero()
      for entry in zip(*nz):
        addr = ml.address(entry)
        memory[addr] = datatype.literal(self._value(datatype, addr, span, scale))
      self.temporary(resultName, size, datatype, memory=memory)
    self._rand += 1

  def _valueSpan(self, datatype, maxValue):
    """How many distinct values the filling pattern may cycle through.

    A narrow integer cannot hold `maxValue`, and the pattern would wrap into
    the negatives (or trap, depending on the conversion).
    """
    if datatype.isBool():
      return 2
    _, hi = datatype.limits()
    return maxValue if hi is None else min(maxValue, int(hi))

  def _value(self, datatype, offset, span, scale):
    """The value the entry at `offset` is filled with, as a Python value."""
    if datatype.isBool():
      # alternate, so that both branches of a guard get exercised; a pattern
      # that is never zero would be all-true
      return ((offset + self._rand) % 2) == 0
    value = float((offset + self._rand) % span) + 1.0
    if datatype.isFloat():
      # Keep the magnitude at or below one. A chain of contractions multiplies
      # the operand magnitude once per factor, while the reference is compared
      # against a fixed relative epsilon -- with values of order `span` the
      # single-precision runs of the longer examples sit close to that bound.
      return value * scale
    return value

  def _valueExpression(self, datatype, offsetVar, span, scale):
    """The same value as `_value`, as C++ over a loop variable."""
    ctype = datatype.ctype()
    if datatype.isBool():
      return f'(({offsetVar} + {self._rand}) % 2) == 0'
    value = f'static_cast<{ctype}>(({offsetVar} + {self._rand}) % {span} + 1)'
    if datatype.isFloat():
      return f'{value} * static_cast<{ctype}>({scale})'
    return value

class ExportGenerator:
  #: What this yateto sends, raised whenever a field is added that an
  #: exporter ignoring it would get *wrong* rather than merely miss.
  #:
  #: 6: a scale factor is stated once, as `linear.alpha`, for every kind of
  #:    operation. A multilinear one also listed it among its operands, so an
  #:    exporter honouring both -- which is the only way to be right about an
  #:    element-wise operation, where the factor is never an operand -- applied
  #:    it twice. An operation whose guard can never hold is no longer sent at
  #:    all, rather than sent with a null condition that reads like no guard.
  #:
  #: 5: an occurrence may state `offset_from`, a shift along an axis that is
  #:    only known once the kernel runs. An exporter that ignores it reads
  #:    the same slice every time.
  #:
  #: 4: `add` is a mask over the destination's axes rather than a bool. An
  #:    exporter reading it as one gets the common case right by accident and
  #:    a rank-0 destination wrong: the empty mask accumulates, and `bool([])`
  #:    says it does not.
  #:
  #: 3: a kernel arrives as one description rather than as a call per tensor
  #:    and per operation, and the description is data -- it survives
  #:    `json.dumps`, so it can be recorded, replayed and compared without
  #:    running yateto again.
  #:
  #: 2: an occurrence states the bounding box it touches, the shift a slicing
  #:    operand imposes and whether it is a slice at all, and a tensor states
  #:    the alignment its layout promises. An exporter that ignores the box
  #:    runs every operation over the whole storage; for an assignment that
  #:    writes over entries the operation was never meant to touch. Sparse
  #:    layouts are also described now, by their entries, rather than refused.
  INTERFACE_VERSION = 6

  def __init__(self, arch, attrs=None):
    self.arch = arch
    self.attrs = attrs or {}

  def generate(self, cpp, cache):
    pass

  def add_kernel(self, description):
    """The whole kernel, as data.

    ``{'version': int, 'tensors': [...], 'operations': [...]}``. Everything
    in it is a str, a number, a bool, None, a list or a dict, so it can be
    written out and read back.
    """
    pass

class ExportFactory(KernelFactory):
  @classmethod
  def makeFactory(cls, generator):
    return lambda cpp, arch, target, attrs=None: cls(
      cls._makeExporter(generator, arch, attrs), cpp, arch, target, attrs)

  @staticmethod
  def _makeExporter(generator, arch, attrs):
    """The exporter, told which kernel it is about to generate.

    Its own interface decides what it gets. An exporter that takes ``attrs``
    generates the kernel those attributes describe. One that does not
    predates the channel and can only generate the interface that existed
    before it, which has a flags parameter on every kernel -- and since this
    side then emits no flags member for that call to name, saying so here is
    better than a compile error two repositories away.
    """
    params = inspect.signature(generator).parameters
    takesAttrs = ('attrs' in params
                  or any(p.kind is p.VAR_KEYWORD for p in params.values()))
    if not takesAttrs:
      raise RuntimeError(
        f'routine exporter {getattr(generator, "__name__", generator)} does not '
        f'accept kernel attributes: it cannot be told whether a kernel takes '
        f'batch flags, and this yateto no longer generates them unconditionally. '
        f'Update the exporter.')

    exporter = generator(arch, attrs=(attrs.as_dict() if attrs is not None else {}))

    # Asked of the exporter, not of whatever produced it: a factory function
    # is a perfectly good way to register one, and it carries no version.
    # An exporter that predates the field speaks the first one. A newer
    # exporter is fine -- the fields it knows and this yateto does not send
    # simply do not appear -- but an older one drops the ones it needs.
    spoken = getattr(exporter, 'INTERFACE_VERSION', 1)
    if spoken < ExportGenerator.INTERFACE_VERSION:
      raise RuntimeError(
        f'routine exporter {exporter.__class__.__name__} speaks interface '
        f'version {spoken}, this yateto sends '
        f'{ExportGenerator.INTERFACE_VERSION}. See the changelog on '
        f'ExportGenerator for what each version added and what an exporter '
        f'ignoring it gets wrong. Update the exporter rather than this check.')

    return exporter

  def __init__(self, generator, cpp, arch, target, attrs=None):
    super().__init__(cpp, arch, target, attrs)
    self.generator = generator
    self.tensors = {}
    self.operations = []
    self.scalarcounter = 0

  def post_generate(self, routine_cache):
    self.generator.add_kernel({
      'version': ExportGenerator.INTERFACE_VERSION,
      # dict order is insertion order, and a tensor is inserted the first
      # time an operation names it, so this is the order they are met in
      'tensors': list(self.tensors.values()),
      'operations': self.operations,
    })
    self.generator.generate(self._cpp, routine_cache)

  def _emit(self, description):
    if description['condition'] is None:
      # a guard that can never hold: the C++ factory emits no action for one
      # either, and an operation the receiving side cannot tell from an
      # unguarded one -- both `None` and `[]` are falsy -- would run always
      return 0
    self.operations.append(description)
    # Nothing is built here, so the statement is an empty region: there is no
    # code to write and no arithmetic to count.
    return ir.Region()

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
    """Describe one occurrence of a tensor.

    Two coordinate systems meet here and they are not interchangeable. The
    *storage* is what the tensor actually occupies; an address is formed in
    it. The *logical* space is what the operand names, and for an operand
    that names a slice the two differ by a shift. The equivalent sparsity
    pattern lives in the logical space, so the bounding box derived from it
    does too, and the shift is stated separately rather than folded in --
    boxes are intersected across operands further down, which only means
    anything if every operand contributes its box in the same space.
    """
    ml = tensorIndexed.memoryLayout

    # A view names a slice. Peel the views off to reach the storage, adding
    # up the shift they impose on the way.
    sliced = isinstance(ml, MemoryLayoutView)
    offset = [0] * len(ml.shape())
    while isinstance(ml, MemoryLayoutView):
      offset = list(ml.relidx(offset))
      ml = ml.base
    ml = ml.storage()

    if isinstance(ml, DenseMemoryLayout):
      shape = list(ml.shape())
      shapeXt = [max(rng.stop - rng.start, shp) for rng, shp in zip(ml.bbox(), shape)]
      storage = {
        'shape': self._ints(shapeXt),
        'type': 'bbox',
        'start': self._ints(rng.start for rng in ml.bbox()),
        'sizes': self._ints(rng.stop - rng.start for rng in ml.bbox())
      }
    elif isinstance(ml, (CSCMemoryLayout, PatternMemoryLayout)):
      shape = list(ml.shape())
      entries = ml.entries(*[Range(0, extent) for extent in shape])
      # sorted by the address the layout gives them: the receiving side
      # numbers the entries in the order they arrive, and that numbering has
      # to be the storage order or every address disagrees
      entries.sort(key=ml.address)
      storage = {
        'shape': self._ints(shape),
        'type': 'spp',
        'entries': [self._ints(entry) for entry in entries]
      }
    else:
      raise NotImplementedError(
        f'{tensorIndexed.name} has a {ml.__class__.__name__}, which the '
        f'description has no storage kind for.')

    values = (None if tensorIndexed.values is None
              else {'kind': 'flat', 'data': [float(v) for v in tensorIndexed.values]})

    tensor = {
      'name': tensorIndexed.name,
      'addressing': self._handleAddressing(tensorIndexed),
      #'eqspp': spp,
      'datatype': str(tensorIndexed.datatype),
      'storage': storage,
      'values': values,
      # What the layout guarantees about the address of a column, in bytes.
      # Zero is not "unaligned", it is "no promise" -- the receiving side
      # decides what to do with a promise, and can make none out of nothing.
      'alignment': self._alignment(tensorIndexed.memoryLayout),
      'flags': {
        'temporary': tensorIndexed.is_temporary,
        'constant': tensorIndexed.is_compute_constant
      }
    }

    return self._handleTensor(tensor, tensorIndexed.indices,
                              self._logicalBox(tensorIndexed), offset, sliced)

  @staticmethod
  def _ints(values):
    """Plain integers.

    Indices and bounds arrive as numpy scalars, which are numbers everywhere
    except where the description is written out -- `json` refuses an
    `int64` -- so they are made ordinary here rather than at every use.
    """
    return [int(value) for value in values]

  def _alignment(self, memoryLayout):
    """The alignment the layout promises for a column, in bytes.

    A tensor without axes has no column and promises nothing; asking the
    layout would read a bounding box that has no first dimension.
    """
    if len(memoryLayout.shape()) == 0:
      return 0
    return self._arch.alignment if memoryLayout.alignedStride() else 0

  @staticmethod
  def _logicalBox(tensorIndexed):
    """The box this occurrence touches, in the space the operand names.

    Derived from the equivalent sparsity pattern, which is what the whole
    optimisation upstream of here computed: it is the range the operation
    actually runs over, and it is regularly a good deal smaller than the
    storage. A pattern with no non-zeros at all has no box; the pair is None
    then, and whoever receives it falls back on the storage.
    """
    eqspp = tensorIndexed.eqspp
    if eqspp is None or eqspp.ndim == 0 or eqspp.count_nonzero() == 0:
      return None
    box = BoundingBox.fromSpp(eqspp)
    return [ExportFactory._ints(rng.start for rng in box),
            ExportFactory._ints(rng.stop for rng in box)]

  def _scalarTensor(self, scalar):
    if isinstance(scalar, (int, float)): # TODO numpy types
      name = f'_scalar{self.scalarcounter}'
      self.scalarcounter += 1

      tensor = {
        'name': name,
        'addressing': '',
        'datatype': str(self._arch.datatype),
        'storage': {
          'shape': [],
          'type': 'full'
        },
        # a scalar is passed by value; there is no address to promise anything about
        'alignment': 0,
        'values': {'kind': 'entries', 'data': [[[], scalar]]},
        'flags': {
          'temporary': False,
          'constant': True
        }
      }
    elif isinstance(scalar, Tensor):
      tensor = {
        'name': scalar.name(),
        'addressing': '',
        'datatype': str(scalar.getDatatype(self._arch)),
        'storage': {
          'shape': [],
          'type': 'full'
        },
        # a scalar is passed by value; there is no address to promise anything about
        'alignment': 0,
        'values': None,
        'flags': {
          'temporary': False,
          'constant': True
        }
      }
    else:
      assert False

    return self._handleTensor(tensor, [])

  def _handleTensor(self, tensor, indices, bbox=None, offset=None,
                    sliced=False, offsetFrom=None):
    if tensor['name'] not in self.tensors:
      self.tensors[tensor['name']] = tensor
    else:
      assert tensor == self.tensors[tensor['name']]

    # `bbox`, `offset` and `sliced` belong to this occurrence, not to the
    # tensor: two operands may name two different slices of the same thing.
    # NOTE: the full list of non-zero indices used to travel with every
    #       reference and nothing ever read it -- the bounding box is what
    #       narrows a loop, and a sparse layout states its entries once, on
    #       the tensor. For the `indices` example that list was 25 MiB of the
    #       25.7 MiB description.
    #
    # `offset_from` is the same shift, for an axis whose slice is only known
    # once the kernel runs: an entry names a rank-0 integer tensor whose value
    # is added to `offset` on that axis. That is how one of several matrices
    # gets selected -- a family is a tensor with one axis more, and that axis
    # is addressed rather than iterated, which is what an offset already is.
    return {
      'name': tensor['name'],
      'indices': [str(index) for index in indices],
      'bbox': bbox,
      'offset': self._ints(offset) if offset is not None else None,
      'offset_from': offsetFrom,
      'sliced': sliced
    }

  @staticmethod
  def _addMask(dest, add):
    """Which of the destination's axes the accumulated value spans.

    `False` for an operation that overwrites. Otherwise the axes, stated the
    way `target` states an operand's, because that is the same question: a
    value spanning fewer axes than the destination is broadcast over the rest,
    and indexing it by an axis it does not have reads somewhere else entirely.
    A bare `True` could not say which, so it had to mean "all of them".
    """
    if not add or dest is None:
      return False
    return list(range(len(dest['indices'])))

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
        'add': self._addMask(result, add),
      },
      'optype': str(node.optype)
    }
    return self._emit(description)

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
        'add': self._addMask(result, add),
      },
      'optype': str(node.optype)
    }
    return self._emit(description)

  def create_LoopOverGEMM(self, node, result, arguments, condition, add, scalar, prefetchName, routineCache, gemm_cfg):
    assert len(arguments) == 2
    # NOTE: no transposition flags. Which axis of an operand goes where is
    #       already in `target`, and a flag saying it again could disagree.
    argnodes = [self._nodeTensor(arguments[0], node[0]), self._nodeTensor(arguments[1], node[1])]
    return self.handleLinear(self._nodeTensor(result, node), argnodes, condition, add, scalar)


  def create_Permute(self, node, result, arguments, condition, add, scalar, prefetchName, routineCache, gemm_cfg):
    term = arguments[0]
    return self.handleLinear(self._varTensor(result, node.indices), [self._varTensor(term, node.term().indices)], condition, add, scalar)

  def create_Broadcast(self, node, result, arguments, condition, add, scalar, prefetchName, routineCache, gemm_cfg):
    term = arguments[0]
    return self.handleLinear(self._varTensor(result, node.indices), [self._varTensor(term, node.term().indices)], condition, add, scalar)

  def simple(self, result, term, condition, add, scalar, routineCache, gemm_cfg):
    return self.handleLinear(self._varTensor(result, self._indices(result)), [self._varTensor(term, self._indices(term))], condition, add, scalar)

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

  def handleLinear(self, dest, ops, condition, add, scalar):
    # convert indices to loop numbers

    target, permute = self.getIndices(dest, ops)

    # NOTE: the factor belongs in `linear.alpha` and nowhere else. Listing it
    #       among the operands as well made every exporter that also reads
    #       alpha -- as it must, since an element-wise operation states its
    #       factor there and cannot state it as an operand -- scale twice.
    description = {
      'type': 'multilinear',
      'result': dest,
      'args': ops,
      'condition': self._handleCondition(condition),
      'permute': permute,
      'target': target,
      'linear': {
        'alpha': self._scalarTensor(scalar),
        'add': self._addMask(dest, add),
      },
      # 'optype': node.optype
    }
    return self._emit(description)
