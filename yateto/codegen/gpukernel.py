from .factory import KernelFactory
from kernelforge.generators.descriptions import MultilinearDescr
from common import *

from .common import TensorDescription, IndexedTensorDescription, BatchedOperationsAux
from ..ast.indices import BoundingBox, Range
from ..type import Scalar
from .cache import RoutineGenerator, GpuRoutineGenerator
from kernelforge.interface import YatetoInterface as yi
from kernelforge.generators.descriptions import GemmDescr
from kernelforge.common.basic_types import Addressing, FloatingPointType, DataFlowDirection
from kernelforge.common.context import Context
from kernelforge.common.aux import generate_tmp_tensor
from kernelforge.common.matrix.tensor import Tensor, TensorWrapper, SubTensor
from kernelforge.generators.generator import Generator as KernelForgeGenerator

class GpuKernelGenerator:
  def __init__(self, arch):
    self._arch = arch
    self._batch_aux = BatchedOperationsAux(self._arch.typename)
    self._cache = {}
    self._tmp_matrices = {}
    self._descr_list = []

  def add_operation(self, dest, ops, target, permute, add):
      self._cache_matrices(dest, ops, target, permute)
      can_be_aligned = self._can_be_aligned(dest, ops, target, permute)
      self._descr_list.append(MultilinearDescr(self._cache[dest.name],
                                [self._cache[op.name() if isinstance(op, Scalar) else op.name] for op in ops],
                                target, permute, add=add,
                                 strict_match=False,
                                 prefer_align=can_be_aligned))
      return 0# self._descr_list[-1].get_flops()

  def generate(self, cpp, routineCache):
    context = Context(arch=self._arch.name,
                      backend=self._arch.backend,
                      fp_type=FloatingPointType.str2enum(self._arch.typename))

    kernelforge_generator = KernelForgeGenerator(self._descr_list, context)
    kernelforge_generator.register()

    cpp(f'{self._gen_call_site(kernelforge_generator)}')
    routine_name = kernelforge_generator.get_base_name()
    kernelforge_generator.generate()

    routineCache.addRoutine(routine_name, KernelForgeWriter(kernelforge_generator, context.get_vm().get_headers()))

  def _can_be_aligned(self, dest, ops, target, permute):
    # TODO: useful?
    aligned = dest.memoryLayout.alignedStride()
    for i, op in enumerate(ops):
      if 0 in target[i]:
        aligned &= dest.memoryLayout.alignedStride() and permute[i][0] == 0
    
    return aligned

  def _cache_matrices(self, dest, ops, target, permute):
    can_be_aligned = self._can_be_aligned(dest, ops, target, permute)

    def make_tensor(op, dims):
      if isinstance(op, Scalar):
        entry = self._add_scalar(op)
        entry_name = op.name()
      else:
        # TODO: refine
        currentPreShape = list(BoundingBox.fromSpp(op.eqspp))
        if can_be_aligned:
          for i, dim in enumerate(dims):
            if i == 0 and op.memoryLayout.alignedStride(): # previously: dim == 0
              currentPreShape[i] = currentPreShape[i].aligned(self._arch)
        currentShape = [b.stop for b in currentPreShape]
        currentRange = list(BoundingBox(Range(0, b) for b in currentShape))
              
        entry = self._get_kernelforge_matrix(tensor=op,
                                            tensor_variable=op,
                                            shape=currentShape,
                                            bboxrange=currentRange)
        entry_name = op.name

      if not (entry_name in self._cache and entry.is_same(self._cache[entry_name])):
        self._cache[entry_name] = entry
    
    # no add onto a matrix that doesn't exist (TODO: check if that's always the case)
    assert not(dest.is_temporary and dest in ops)

    for op, optarget in zip(ops, target):
      make_tensor(op, optarget)

    if dest.is_temporary: # (dest is never a scalar---for the time being)
      self._cache[dest.name] = self._gen_tmp_matix(ops, target, permute, dest.name)
    else:
      make_tensor(dest, [i for i in range(len(dest.indices))])

  def _add_scalar(self, scalar):
    tensor = Tensor([], Addressing.SCALAR, alias=scalar.name())
    self._tmp_matrices[scalar.name()] = tensor # SubTensor(tensor, tensor.bbox)
    return self._tmp_matrices[scalar.name()]

  def _get_kernelforge_matrix(self, tensor, tensor_variable, shape, bboxrange):
    addr_mode = self._batch_aux.deduce_addresing(tensor)
    if tensor_variable.is_temporary:
      if not tensor_variable.name in self._tmp_matrices:
        raise RuntimeError(f'expected tmp. tensor {tensor_variable.name} to be cached '
                           f'while code generation for fused-gemms')
      else:
        return self._tmp_matrices[tensor_variable.name]

    return yi.gen_matrix(shape,
                               bboxrange,
                               addressing=addr_mode,
                               name=tensor_variable.name,
                               is_tmp=tensor_variable.is_temporary,
                               permute=None,
                               pattern = tensor_variable.eqspp,
                               values = tensor_variable.eqspp)

  def _gen_tmp_matix(self, ops, target, permute, res_name):
    # TODO: ignore scalars here?
    tmp_matrix = generate_tmp_tensor(ops=[self._cache[op.name() if isinstance(op, Scalar) else op.name] for op in ops],
                                     target=target) # permute?
    self._tmp_matrices[res_name] = tmp_matrix
    return tmp_matrix

  def _gen_call_site(self, generator):
    mat_name_map = {}
    offset_name_map = {}
    for name, matrix in self._cache.items():
      if matrix.direction == DataFlowDirection.SOURCE and matrix.addressing != Addressing.SCALAR:
        ptr_type = f'const {self._arch.typename}{Addressing.addr2ptr_type(matrix.addressing)}'
        mat_name_map[name] = f'const_cast<{ptr_type}>({name})'
      else:
        mat_name_map[name] = name

      if matrix.is_tmp or matrix.addressing == Addressing.NONE:
        offset_name_map[name] = '0'
      else:
        offset_name_map[name] = f'{BatchedOperationsAux.EXTRA_OFFSET_NAME}_{name}'
    
    return generator.generate_call_site(mat_name_map,
                                        offset_name_map,
                                        BatchedOperationsAux.NUM_ELEMENTS_NAME,
                                        BatchedOperationsAux.FLAGS_NAME,
                                        BatchedOperationsAux.STREAM_PTR_NAME)

class KernelForgeWriter(GpuRoutineGenerator):
  def __init__(self, kernelforge_generator, headers):
    self._headers = list(headers) + list(kernelforge_generator.get_helper_headers())
    self._generator = kernelforge_generator
    self._basename = self._generator.get_base_name()

  def __eq__(self, other):
    if isinstance(other, KernelForgeWriter):
      return self._basename == other._basename
    else:
      return False

  def header(self, cpp):
    cpp.includes(self._headers)

  def __call__(self, routineName, fileName):
    launcher = self._generator.get_launcher()
    kernel = self._generator.get_kernel()

    with open(fileName, 'a', encoding='utf-8') as file:
      file.write(kernel)
      file.write(launcher)

    return self._generator.get_header()


class GpuKernelFactory(KernelFactory):
  def __init__(self, cpp, arch, target):
    super().__init__(cpp, arch, target)
    self.generator = GpuKernelGenerator(arch)

  def temporary(self, bufname, size, iniZero=False, memory=list()):
    # disabled for GPU kernels
    self._cpp('{}* {};'.format(self._arch.typename, bufname))

  def freeTmp(self, routineCache):
    # generate the kernel and the kernel call here... A tiny bit hacky, but it works
    self.generator.generate(self._cpp, routineCache)

  def create(self, node, *args):
    method = 'create_' + node.__class__.__name__
    factory = getattr(self, method, self.generic_create)
    return factory(node, *args)
  
  def create_LoopOverGEMM(self, node, result, arguments, add, scalar, prefetchName, routineCache, gemm_cfg):
    assert len(arguments) == 2
    return self.handle(IndexedTensorDescription.fromNode(result, node), [IndexedTensorDescription.fromNode(arguments[0], node.leftTerm()), IndexedTensorDescription.fromNode(arguments[1], node.rightTerm())], add, scalar, node.transA(), node.transB())
  
  def create_IndexSum(self, node, result, arguments, add, scalar, prefetchName, routineCache, gemm_cfg):
    assert len(arguments) == 1
    return self.handle(IndexedTensorDescription.fromNode(result, node), [IndexedTensorDescription.fromNode(arguments[0], node.term())], add, scalar, False, False)
  
  def create_Product(self, node, result, arguments, add, scalar, prefetchName, routineCache, gemm_cfg):
    assert len(arguments) == 2
    return self.handle(IndexedTensorDescription.fromNode(result, node), [IndexedTensorDescription.fromNode(arguments[0], node.leftTerm()), IndexedTensorDescription.fromNode(arguments[1], node.rightTerm())], add, scalar, False, False)

  def create_Permute(self, node, result, arguments, add, scalar, prefetchName, routineCache, gemm_cfg):
    term = arguments[0]
    return self.handle(IndexedTensorDescription(str(result), node.indices, result.memoryLayout(), result.eqspp()), [IndexedTensorDescription(str(term), node.term().indices, term.memoryLayout(), term.eqspp())], add, scalar, False, False)
  
  def simple(self, result, term, add, scalar, routineCache):
    return self.handle(IndexedTensorDescription(str(result), self._indices(result), result.memoryLayout(), result.eqspp()), [IndexedTensorDescription(str(term), self._indices(term), term.memoryLayout(), term.eqspp())], add, scalar, False, False)
  
  def handle(self, dest, ops, add, scalar, transposeA, transposeB):
    # convert indices to loop numbers

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
    
    if not (scalar == 1 or scalar == 1.0):
      ops += [scalar]
      target += [[]]
      permute += [[]]
    
    return self.generator.add_operation(dest, ops, target, permute, add)
