from __future__ import annotations

from .. import aspp
from ..type import AddressingMode, Datatype
from ..ast.indices import BoundingBox
from ..ast.log import splitByDistance
from ..description import TensorDescription, IndexedTensorDescription
from ..ir import INDEX_PREFIX
from .tiny_tensor_language import Dump, Function, ScalarType, IntegerType, FloatingType, MemrefType, GroupType, IntImmValue, FloatImmValue, DYNAMIC, SubviewInst, LoadInst
import hashlib

def forLoops(cpp, indexNames, ranges, body, pragmaSimd=True, prefix=INDEX_PREFIX, fixed={}, indexNo=None):
  flops = 0
  firstLoop = False
  if indexNo == None:
    indexNo = len(indexNames)-1
    firstLoop = True
    # bail out before emitting anything if any pinned index misses its range,
    # otherwise we leave empty scopes with unused constexpr variables behind
    for index in indexNames:
      if index in fixed and not (ranges[index].start <= fixed[index] < ranges[index].stop):
        return 0
    # A nest with nothing between its loops is one iteration space, and saying
    # so beats marking the innermost loop alone: a short innermost loop is
    # unrolled away before the vectoriser sees it, and what is left is scalar.
    # A pinned index breaks the nesting, so it rules the clause out.
    if pragmaSimd and len(indexNames) > 1 and not any(i in fixed for i in indexNames):
      cpp(f'#pragma omp simd collapse({len(indexNames)})')
      pragmaSimd = False
  if indexNo < 0:
    if firstLoop:
      with cpp.AnonymousScope():
        flops = body()
    else:
      flops = body()
  else:
    index = indexNames[indexNo]
    rng = ranges[index]
    if index in fixed:
      with cpp.AnonymousScope():
        cpp(f'[[maybe_unused]] constexpr int {prefix}{index} = {fixed[index]};')
        flops = forLoops(cpp, indexNames, ranges, body, pragmaSimd, prefix, fixed, indexNo-1)
    else:
      # the pragma belongs on the innermost *emitted* loop, i.e. the one over the
      # fastest-running index that has not been pinned by unrolling
      if pragmaSimd and all(indexNames[i] in fixed for i in range(indexNo)):
        cpp('#pragma omp simd')
      with cpp.For('int {3}{0} = {1}; {3}{0} < {2}; ++{3}{0}'.format(index, rng.start, rng.stop, prefix)):
        flops = forLoops(cpp, indexNames, ranges, body, pragmaSimd, prefix, fixed, indexNo-1)
      flops = flops * rng.size()
  return flops

def loopRanges(term: IndexedTensorDescription, loopIndices):
  # NOTE: the term's own index order decides the insertion order here, and that
  #       order reaches the generated code as the nesting of the unrolled loops
  #       (Generic._generateUnroll). Iterating an intersection of two sets would
  #       hand back an order that PYTHONHASHSEED varies between runs.
  wanted = set(loopIndices)
  bbox = BoundingBox.fromSpp(term.eqspp)
  return {index: bbox[position] for position, index in enumerate(term.indices)
          if index in wanted}

def testLoopRangesEqual(A, B):
  overlap = A.keys() & B.keys()
  return all([A[index] == B[index] for index in overlap])

def testLoopRangesAContainedInB(A, B):
  overlap = A.keys() & B.keys()
  return all([A[index] in B[index] for index in overlap])

def boundingBoxFromLoopRanges(indices, loopRanges):
  return BoundingBox([loopRanges[index] for index in indices])

def reduceSpp(spp, sourceIndices, targetIndices, fixedIndices):
  return spp.indexSum(sourceIndices, targetIndices, fixedIndices)

class KernelAttributes:
  """Switches the caller sets on one kernel at ``Generator.add`` time.

  They are not code generation options: a kernel's attributes are part of
  what the kernel *is*, because they change its generated interface. The
  batch flags are the first one -- a kernel that does not declare them has
  no ``flags`` member to assign to, so a caller that means to mask elements
  off and forgot the attribute finds out from the compiler rather than from
  a result that silently ignored the mask.

  Unknown keys are rejected here rather than ignored, since an attribute
  that does nothing looks exactly like a typo in one that would have.
  """

  FLAGS = 'flags'
  KNOWN = frozenset({FLAGS})

  def __init__(self, attrs=None):
    attrs = dict(attrs) if attrs else {}
    unknown = sorted(set(attrs) - self.KNOWN)
    if unknown:
      raise ValueError(
        'unknown kernel attribute(s) {}; known are {}'.format(
          ', '.join(repr(key) for key in unknown),
          ', '.join(repr(key) for key in sorted(self.KNOWN))))
    self._attrs = attrs

  @property
  def flags(self):
    """Whether the kernel takes a per-element mask of elements to skip."""
    return bool(self._attrs.get(self.FLAGS, False))

  def as_dict(self):
    """The attributes as the external code generators want them."""
    return dict(self._attrs)

  def __eq__(self, other):
    if isinstance(other, KernelAttributes):
      return self._attrs == other._attrs
    return NotImplemented

  def __repr__(self):
    return f'KernelAttributes({self._attrs!r})'


class BatchedOperationsAux:
  NUM_ELEMENTS_NAME = 'numElements'
  EXTRA_OFFSET_NAME = 'extraOffset'
  STREAM_PTR_NAME = 'streamPtr'
  FLAGS_NAME = 'flags'
  FORBIDDEN_STREAM_PTR = 'reinterpret_cast<void*>(std::numeric_limits<uintptr_t>::max())'

  @classmethod
  def _get_ptr_type(cls, addressing: AddressingMode):
    return addressing.pointer_type()

  @classmethod
  def flags_arg(cls, attrs):
    """The batch-flags argument for a kernel that always takes one.

    The external generators (GemmForge, ChainForge) put a flags parameter in
    every kernel they emit, so the choice at the call site is between the
    member and a literal null -- and the member only exists when the kernel
    declares the attribute.
    """
    return cls.FLAGS_NAME if attrs.flags else 'nullptr'

  @classmethod
  def deduce_addresing(cls, term):
    if term.addressing is not None:
      return term.addressing

    # default deduction
    if term.is_compute_constant:
      return AddressingMode.DIRECT
    if term.is_temporary:
      return AddressingMode.STRIDED
    else:
      return AddressingMode.INDIRECT

  @classmethod
  def deduce_ptr_arg(cls, term, as_const=False):
    if as_const:
      addressing = cls.deduce_addresing(term)
      ptr = cls._get_ptr_type(addressing)
      assert term.datatype is not None
      datatype = term.datatype.ctype()
      const_ptr_type = f'const {datatype} {ptr}'
      return f'const_cast<{const_ptr_type}>({term.name})'
    else:
      return f'{term.name}'

  @classmethod
  def deduce_offset_arg(cls, term):
    if term.is_compute_constant or term.is_temporary:
      return '0'
    else:
      return f'{cls.EXTRA_OFFSET_NAME}_{term.name}'

class TinytcKernelArgument:

  def __init__(self, name: str, datatype: str, call_expr: str, constant: bool, temporary: bool, modified: bool, offset: int = 0):
    """Kernel argument for TinytcWrapper.

    Arguments:
    name -- Argument name
    datatype -- Argument datatype in C/C++
    call_expr -- Expression used in calling wrapper
    constant -- Whether a tensor is invariant to group id
    temporary -- Whether a tensor is stored in a temporary buffer
    modified -- Whether tensor is modified during kernel
    """
    self.name = name
    self.call_expr = call_expr
    self.constant = constant
    self.temporary = temporary
    self.modified = modified
    self.offset = offset

class TinytcScalarKernelArgument:

  def __init__(self, name: str, call_expr: str):
    self.name = name
    self.call_expr = call_expr

class TinytcWrapper:

  def __init__(self, kernel: Function, arguments: list[TinytcKernelArgument | TinytcScalarKernelArgument], name: str = ''):
    self.kernel_name = kernel.name
    self.source = Dump().visit(kernel)
    if name:
      self.name = name
    else:
      hasher = hashlib.sha512()
      hasher.update(self.source.encode('utf-8'))
      self.name = f'tinytc_wrapper_{hasher.hexdigest()}'

    self.wrapper_args = [f'long {BatchedOperationsAux.NUM_ELEMENTS_NAME}', f'void* {BatchedOperationsAux.STREAM_PTR_NAME}']
    self.wrapper_call_args = []
    self.call_args = []
    for arg in arguments:
        if isinstance(arg, TinytcScalarKernelArgument):
            self.wrapper_args.append(f'{arg.datatype} {arg.name}')
            self.wrapper_call_args.append(arg.name)
            self.call_args.append(arg.call_expr)
        else:
          ptr2ptr = '*' if not (arg.constant or arg.temporary) else ''
          const = ' const' if not (arg.modified or arg.temporary) else ''
          wrapper_type = f'{arg.datatype}{const}*{ptr2ptr}'
          self.wrapper_args.append(f'{wrapper_type} {arg.name}')
          self.wrapper_call_args.append(arg.name)
          self.call_args.append(f'const_cast<{wrapper_type}>({arg.call_expr})')
          if not arg.constant:
            self.wrapper_call_args.append(BatchedOperationsAux.NUM_ELEMENTS_NAME)
          if not arg.temporary and not arg.constant:
            offset_name = f'{BatchedOperationsAux.EXTRA_OFFSET_NAME}_{arg.name}'
            self.wrapper_args.append(f'long {offset_name}')
            self.wrapper_call_args.append(offset_name)
            self.call_args.append(f'{BatchedOperationsAux.EXTRA_OFFSET_NAME}_{arg.call_expr}')
          if arg.offset:
            self.call_args[-1] += f' + {arg.offset}'

  def definition(self):
    make_kernel = """    struct custom_kernel { ::sycl::kernel kernel; ::sycl::range<3u> group_size; };
    static auto k = [&] (::sycl::queue const& queue) -> custom_kernel {
        static const std::string source = R\"tinytc(
"""
    make_kernel += self.source
    make_kernel += """)tinytc\";
        auto err_log = std::string{};
        try {
            auto ctx = tinytc::create_compiler_context();
            tinytc::set_error_reporter(ctx.get(), [](char const *what, const tinytc_location_t *, void *log) {
                *static_cast<std::string*>(log) += what;
            }, &err_log);
            auto program = tinytc::parse_string(source, ctx.get());
            auto bundle = tinytc::create_kernel_bundle(queue.get_context(), queue.get_device(), program.get(), 0);"""
    make_kernel += f'            auto kernel = tinytc::create_kernel(bundle, "{self.kernel_name}");\n'
    make_kernel += """            auto group_size = tinytc::get_group_size(kernel);
            return {std::move(kernel), std::move(group_size)};
        } catch (tinytc::status const& st) {
            if (!err_log.empty()) {
                throw std::runtime_error(err_log);
            } else {
                throw std::runtime_error(tinytc::to_string(st));
            }
        }
    }""";
    make_kernel += f'(*static_cast<::sycl::queue*>({BatchedOperationsAux.STREAM_PTR_NAME}));\n'

    wrapper = f'{self.prototype()[:-1]} {{\n'
    wrapper += make_kernel
    wrapper += f'    static_cast<::sycl::queue*>({BatchedOperationsAux.STREAM_PTR_NAME})->submit([&](::sycl::handler &h) {{\n';
    wrapper += f'        h.set_args({", ".join(self.wrapper_call_args)});\n'
    wrapper += f'        h.parallel_for(::sycl::nd_range{{tinytc::get_global_size({{1,1,static_cast<std::size_t>({BatchedOperationsAux.NUM_ELEMENTS_NAME})}}, k.group_size), k.group_size}}, k.kernel);\n'
    wrapper +=  '    });\n'
    wrapper += '}\n\n'

    return wrapper

  def call(self):
    return f'{self.name}({BatchedOperationsAux.NUM_ELEMENTS_NAME}, {BatchedOperationsAux.STREAM_PTR_NAME}, {", ".join(self.call_args)});'

  def prototype(self):
    return f'void {self.name}({", ".join(self.wrapper_args)});'

def makeMemrefType(scalarTy, memoryLayout, needsBatchMode: bool, local: bool=False):
  shape = tuple(r.size() for r in memoryLayout.bbox())
  stride = memoryLayout.stride()
  if needsBatchMode:
    shape = shape + (DYNAMIC, )
    stride = stride + (memoryLayout.requiredReals(), )
  return MemrefType(scalarTy, shape, stride, local)

def makeBatchType(scalarTy, memoryLayout, isComputeConstant: bool, isTemporary: bool):
  if isComputeConstant:
    return makeMemrefType(scalarTy, memoryLayout, False)
  elif isTemporary:
    return makeMemrefType(scalarTy, memoryLayout, True)
  else:
    return GroupType(makeMemrefType(scalarTy, memoryLayout, False), DYNAMIC)

def makeLoad(bb, operand, gid, isComputeConstant: bool, isTemporary: bool):
  if isComputeConstant:
    return operand
  elif isTemporary:
    offsetList = [IntImmValue(IntegerType.index, 0)] * (operand.type().order() - 1)
    sizeList = [IntImmValue(IntegerType.index, DYNAMIC)] * (operand.type().order() - 1)
    offsetList.append(gid)
    sizeList.append(None)
    return bb.add(SubviewInst(operand, offsetList, sizeList))
  else:
    return bb.add(LoadInst(operand, [gid]))

def toTinyTCType(datatype: Datatype):
  return {
    Datatype.BOOL: ScalarType(IntegerType.i1), # presumably, maybe i8
    Datatype.I8: ScalarType(IntegerType.i8),
    Datatype.I16: ScalarType(IntegerType.i16),
    Datatype.I32: ScalarType(IntegerType.i32),
    Datatype.I64: ScalarType(IntegerType.i64),
    Datatype.F32: ScalarType(FloatingType.f32),
    Datatype.F64: ScalarType(FloatingType.f64)
  }[datatype]

def toTinyTCImmediate(datatype: Datatype, value):
  return {
    Datatype.BOOL: lambda value: IntImmValue(IntegerType.i1, value),
    Datatype.I8: lambda value: IntImmValue(IntegerType.i8, value),
    Datatype.I16: lambda value: IntImmValue(IntegerType.i16, value),
    Datatype.I32: lambda value: IntImmValue(IntegerType.i32, value),
    Datatype.I64: lambda value: IntImmValue(IntegerType.i64, value),
    Datatype.F32: lambda value: FloatImmValue(FloatingType.f32, value),
    Datatype.F64: lambda value: FloatImmValue(FloatingType.f64, value),
  }[datatype](value)
