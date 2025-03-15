from __future__ import annotations
from .. import aspp
from ..ast.indices import BoundingBox
from ..ast.log import splitByDistance
from .tiny_tensor_language import Dump, Function, IntegerType, MemrefType, GroupType, IntImmValue, DYNAMIC, SubviewInst, LoadInst
import hashlib


class TensorDescription(object):
  def __init__(self, name, memoryLayout, eqspp, is_compute_constant=False, is_temporary=False):
    """

    Args:
      name (str): tensor's symbol name
      memoryLayout:
      eqspp:
      is_compute_constant (bool): if true then sparsity patterns and numerical values of tensor
          elements are known at compile time
      is_temporary (bool): if true then the description is for a temporary tensor which
          usually results from a result of an intermediate computation
    """
    self.name = name
    self.memoryLayout = memoryLayout
    self.eqspp = eqspp
    self.is_compute_constant = is_compute_constant
    self.is_temporary = is_temporary
    BoundingBox(eqspp)
  
  @classmethod
  def fromNode(cls, name, node):
    return cls(name, node.memoryLayout(), node.eqspp())

class IndexedTensorDescription(TensorDescription):
  def __init__(self, name, indices, memoryLayout, eqspp, is_compute_constant=False, is_temporary=False):
    super().__init__(name, memoryLayout, eqspp, is_compute_constant, is_temporary)
    self.indices = indices

  @classmethod
  def fromNode(cls, var, node):
    is_const = False
    if hasattr(node, 'tensor'):
      is_const = node.tensor.is_compute_constant()
    return cls(str(var), node.indices, var.memoryLayout(), node.eqspp(), is_const, var.is_temporary)

  @classmethod
  def fromVar(cls, var, indices):
    is_const = False
    if hasattr(var, 'tensor'):
      if var.tensor is not None:
        is_const = var.tensor.is_compute_constant()
    return cls(str(var), indices, var.memoryLayout(), var.eqspp(), is_const, var.is_temporary)

def forLoops(cpp, indexNames, ranges, body, pragmaSimd=True, prefix='_', indexNo=None, indexer=None):
  flops = 0
  if indexNo == None:
    indexNo = len(indexNames)-1
  if indexNo < 0:
    flops = body()
  else:
    index = indexNames[indexNo]
    rng = ranges[index]
    iterstart = rng.start
    increment = 1
    if pragmaSimd and indexNo == 0:
      if indexer is None:
        cpp('#pragma omp simd')
      else:
        iterstart = f'{rng.start} + {indexer[0]}'
        increment = f'{indexer[1]}'
    with cpp.For('int {3}{0} = {1}; {3}{0} < {2}; {3}{0} += {4}'.format(index, iterstart, rng.stop, prefix, increment)):
      flops = forLoops(cpp, indexNames, ranges, body, pragmaSimd, prefix, indexNo-1, indexer)
    flops = flops * rng.size()
  return flops
  
def loopRanges(term: IndexedTensorDescription, loopIndices):
  overlap = set(loopIndices) & set(term.indices)
  bbox = BoundingBox.fromSpp(term.eqspp)
  return {index: bbox[term.indices.find(index)] for index in overlap}

def testLoopRangesEqual(A, B):
  overlap = A.keys() & B.keys()
  return all([A[index] == B[index] for index in overlap])
  
def testLoopRangesAContainedInB(A, B):
  overlap = A.keys() & B.keys()
  return all([A[index] in B[index] for index in overlap])

def boundingBoxFromLoopRanges(indices, loopRanges):
  return BoundingBox([loopRanges[index] for index in indices])

def reduceSpp(spp, sourceIndices, targetIndices):
  return spp.indexSum(sourceIndices, targetIndices)

def initializeWithZero(cpp, arch, result: TensorDescription, writeBB = None):
  if writeBB:
    addresses = sorted(result.memoryLayout.notWrittenAddresses(writeBB))
    if len(addresses) > 0:
      regions = splitByDistance(addresses)
      for region in regions:
        m, M = min(region), max(region)
        initialAddress = '{} + {}'.format(result.name, m)
        cpp.memset(initialAddress, M-m+1, arch.typename)
  else:
    cpp.memset(result.name, result.memoryLayout.requiredReals(), arch.typename)


class BatchedOperationsAux:
  NUM_ELEMENTS_NAME = 'numElements'
  EXTRA_OFFSET_NAME = 'extraOffset'
  STREAM_PTR_NAME = 'streamPtr'
  FLAGS_NAME = 'flags'
  FORBIDDEN_STREAM_PTR = 'reinterpret_cast<void*>(std::numeric_limits<uintptr_t>::max())'

  def __init__(self, underlying_data_type):
    self.underlying_data_type = underlying_data_type

  def _get_ptr_type(self, addressing):
    return '**' if addressing == 'pointer_based' else '*'

  def deduce_addresing(self, term):
    if term.is_compute_constant:
      return 'none'
    if term.is_temporary:
      return 'strided'
    else:
      return 'pointer_based'

  def deduce_ptr_arg(self, term, as_const=False):
    if as_const:
      addressing = self.deduce_addresing(term)
      ptr = self._get_ptr_type(addressing)
      const_ptr_type = f'const {self.underlying_data_type} {ptr}'
      return f'const_cast<{const_ptr_type}>({term.name})'
    else:
      return f'{term.name}'

  def deduce_offset_arg(self, term):
    if term.is_compute_constant or term.is_temporary:
      return '0'
    else:
      return f'{self.EXTRA_OFFSET_NAME}_{term.name}'

class TinytcKernelArgument:

  def __init__(self, name: str, call_expr: str, constant: bool, temporary: bool, modified: bool, offset: int = 0):
    """Kernel argument for TinytcWrapper.

    Arguments:
    name -- Argument name
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

  def __init__(self, kernel: Function, arguments: list[TinytcKernelArgument | TinytcScalarKernelArgument], real_type: str, name: str = ''):
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
            self.wrapper_args.append(f'{real_type} {arg.name}')
            self.wrapper_call_args.append(arg.name)
            self.call_args.append(arg.call_expr)
        else:
          ptr2ptr = '*' if not (arg.constant or arg.temporary) else ''
          const = ' const' if not (arg.modified or arg.temporary) else ''
          wrapper_type = f'{real_type}{const}*{ptr2ptr}'
          self.wrapper_args.append(f'{wrapper_type} {arg.name}')
          self.wrapper_call_args.append(arg.name)
          self.call_args.append(f'const_cast<{wrapper_type}>({arg.call_expr})')
          if arg.temporary:
            self.wrapper_call_args.append(BatchedOperationsAux.NUM_ELEMENTS_NAME)
          elif not arg.constant:
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
        auto source_ctx = tinytc::source_context{};
        try {
            source_ctx = tinytc::make_source_context();
	        auto program = tinytc::parse_string(source, source_ctx);
            auto bundle = tinytc::make_kernel_bundle(queue.get_context(), queue.get_device(), std::move(program), 0, source_ctx);"""
    make_kernel += f'            auto kernel = tinytc::make_kernel(bundle, "{self.kernel_name}");\n'
    make_kernel += """            auto group_size = tinytc::get_group_size(kernel);
            return {std::move(kernel), std::move(group_size)};
        } catch (tinytc::status const& st) {
            throw std::runtime_error(source_ctx.get_error_log());
        }
    }""";
    make_kernel += f'(*static_cast<::sycl::queue*>({BatchedOperationsAux.STREAM_PTR_NAME}));\n'

    wrapper = f'{self.prototype()[:-1]} {{\n'
    wrapper += make_kernel
    wrapper += f'    static_cast<::sycl::queue*>({BatchedOperationsAux.STREAM_PTR_NAME})->submit([&](::sycl::handler &h) {{\n';
    wrapper += f'        h.set_args({", ".join(self.wrapper_call_args)});\n'
    wrapper += f'        h.parallel_for(::sycl::nd_range{{tinytc::get_global_size({BatchedOperationsAux.NUM_ELEMENTS_NAME}, k.group_size), k.group_size}}, k.kernel);\n'
    wrapper +=  '    });\n'
    wrapper += '}\n\n'

    return wrapper

  def call(self):
    return f'{self.name}({BatchedOperationsAux.NUM_ELEMENTS_NAME}, {BatchedOperationsAux.STREAM_PTR_NAME}, {", ".join(self.call_args)});'

  def prototype(self):
    return f'void {self.name}({", ".join(self.wrapper_args)});'

def makeMemrefType(scalarTy, memoryLayout, needsBatchMode: bool):
  shape = tuple(r.size() for r in memoryLayout.bbox())
  stride = memoryLayout.stride()
  if needsBatchMode:
    shape = shape + (DYNAMIC, )
    stride = stride + (memoryLayout.requiredReals(), )
  return MemrefType(scalarTy, shape, stride)

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
