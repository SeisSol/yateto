from ..common import TensorDescription, IndexedTensorDescription, BatchedOperationsAux
from ...ast.indices import BoundingBox
from ..cache import RoutineGenerator, GpuRoutineGenerator
from gemmboost.interfaces import YatetoInterface as yi
from gemmboost.common import GemmDescr, Addressing, FloatingPointType, DataFlowDirection
from gemmboost.common import vm_factory, generate_tmp_matrix
from gemmboost.backend.generator import Generator as GemmBoostGenerator


class FusedGemms:
  def __init__(self, arch, descr):
    self._arch = arch
    self._descr = descr
    self._batch_aux = BatchedOperationsAux(self._arch.typename)
    self._cache = {}
    self._tmp_matrices = {}

  def generate(self, cpp, routineCache, cfg):
    self._tmp_matrices = {}
    self._cache = {}
    gemm_list = []
    flops = 0
    for item in self._descr:
      node, args, add, scalar = item
      res, op1, op2 = args

      self._get_matrices(node, res, op1, op2)
      gemm_list.append(GemmDescr(trans_a=node.transA(),
                                 trans_b=node.transB(),
                                 a=self._cache[op1.name],
                                 b=self._cache[op2.name],
                                 c=self._cache[res.name],
                                 alpha=scalar,
                                 beta=1.0 if add else 0.0))
      flops += gemm_list[-1].compute_flops()

    vm = vm_factory(name=self._arch.name,
                    sub_name=self._arch.sub_name,
                    fp_type=FloatingPointType.str2enum(self._arch.typename))

    gemmboost_generator = GemmBoostGenerator(gemm_list, vm)
    gemmboost_generator.register()

    cpp(f'{self._gen_call_size(gemmboost_generator)}')
    routine_name = gemmboost_generator.get_base_name()
    routineCache.addRoutine(routine_name, GemmBoostWriter(gemmboost_generator))
    return flops

  def _get_matrices(self, node, res, op1, op2):
    res_tensor = IndexedTensorDescription.fromNode(res, node)
    op1_tensor = IndexedTensorDescription.fromNode(op1, node.leftTerm())
    op2_tensor = IndexedTensorDescription.fromNode(op2, node.rightTerm())
    m, n, k = FusedGemms._get_gemm_mnk(op1=op1_tensor,
                                       trans_op1=node.transA(),
                                       op2=op2_tensor,
                                       trans_op2=node.transB())

    matrix = self._get_gemmboost_matrix(tensor=op1_tensor,
                                        tensor_variable=op1,
                                        range=(m, k))

    if not (op1.name in self._cache and matrix.is_same(self._cache[op1.name])):
      self._cache[op1.name] = matrix

    matrix = self._get_gemmboost_matrix(tensor=op2_tensor,
                                        tensor_variable=op2,
                                        range=(k, n))

    if not (op2.name in self._cache and matrix.is_same(self._cache[op2.name])):
      self._cache[op2.name] = matrix

    if res.is_temporary:
      self._cache[res.name] = self._gen_tmp_matix(op1, op2, node, res.name)
    else:
      self._cache[res.name] = self._get_gemmboost_matrix(tensor=res_tensor,
                                                         tensor_variable=res,
                                                         range=(m, n))

  def _get_gemmboost_matrix(self, tensor, tensor_variable, range):
    addr_mode = self._batch_aux.deduce_addresing(tensor)
    if tensor_variable.is_temporary:
      if not tensor_variable.name in self._tmp_matrices:
        raise RuntimeError(f'expected tmp. tensor {tensor_variable.name} to be cached '
                           f'while code generation for fused-gemms')
      else:
        return self._tmp_matrices[tensor_variable.name]

    return yi.gen_dense_matrix(range,
                               tensor.memoryLayout.bbox(),
                               addressing=Addressing.str2addr(addr_mode),
                               name=tensor_variable.name,
                               is_tmp=tensor_variable.is_temporary)

  def _gen_tmp_matix(self, op1, op2, res_node, res_name):
    tmp_matrix = generate_tmp_matrix(op1=self._cache[op1.name],
                                     op2=self._cache[op2.name],
                                     trans_op1=res_node.transA(),
                                     trans_op2=res_node.transB())
    self._tmp_matrices[res_name] = tmp_matrix
    return tmp_matrix

  def _gen_call_size(self, generator):
    mat_name_map = {}
    offset_name_map = {}
    for name, matrix in self._cache.items():
      if matrix.direction == DataFlowDirection.SOURCE:
        ptr_type = f'{self._arch.typename} {Addressing.addr2ptr_type(matrix.addressing)}'
        mat_name_map[name] = f'const_cast<{ptr_type}>({name})'
      else:
        mat_name_map[name] = name

      if matrix.is_tmp or matrix.addressing == Addressing.NONE:
        offset_name_map[name] = '0'
      else:
        offset_name_map[name] = f'{BatchedOperationsAux.EXTRA_OFFSET_NAME}_{name}'

    beta = 1.0 if self._descr.add[-1] else 0.0
    alpha = self._descr.scalar[-1]
    return generator.generate_call_site(mat_name_map,
                                        offset_name_map,
                                        alpha,
                                        beta,
                                        BatchedOperationsAux.NUM_ELEMENTS_NAME,
                                        BatchedOperationsAux.STREAM_PTR_NAME)

  @classmethod
  def _get_gemm_mnk(cls, op1, trans_op1, op2, trans_op2):
    bbox_op1 = BoundingBox.fromSpp(op1.eqspp)
    bbox_op2 = BoundingBox.fromSpp(op2.eqspp)
    k_op1 = 1 if not trans_op1 else 0
    k_op2 = 0 if not trans_op2 else 1

    k = bbox_op1[k_op1] & bbox_op2[k_op2]
    m = bbox_op1[1 - k_op1]
    n = bbox_op2[1 - k_op2]
    return m, n, k


class GemmBoostWriter(GpuRoutineGenerator):
  def __init__(self, gemmboost_generator):
    self._generator = gemmboost_generator
    self._basename = self._generator.get_base_name()

  def __eq__(self, other):
    if isinstance(other, GemmBoostWriter):
      return self._basename == other._basename
    else:
      return False

  def header(self, cpp):
    #cpp.include('gemmboost_aux.h')
    pass

  def __call__(self, routineName, fileName):
    self._generator.generate()
    launcher = self._generator.get_launcher()
    kernel = self._generator.get_kernel()

    with open(fileName, 'a') as file:
      file.write(kernel)
      file.write(launcher)

    return self._generator.get_header()
