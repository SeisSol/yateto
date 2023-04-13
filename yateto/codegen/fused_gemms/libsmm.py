from ..common import TensorDescription, IndexedTensorDescription, BatchedOperationsAux
from ...ast.indices import BoundingBox
from ..cache import RoutineGenerator, GpuRoutineGenerator
from ...ast.node import IndexedTensor
from ...type import Tensor

import hashlib


class FusedGemmsLibsmm:
  W_PREFIX = '_w_'
  ARG_PREFIX = '_arg_'
  OFF_PREFIX = '_offset_'

  def __init__(self, arch, descr):
    self._arch = arch
    self._descr = descr
    self._batch_aux = BatchedOperationsAux(self._arch.typename)
    self._cache = {}
    self._tmp_matrices = {}

  def generate(self, cpp, routineCache, cfg):
    input_matrices = dict()
    is_constant = dict()
    is_modified = dict()
    var_name = dict()

    def store_matrix(var, node, is_result):
        if not (var.is_temporary and is_result) and var not in var_name:
            input_matrices[var] = node.memoryLayout()
            is_constant[var] = node.tensor.is_compute_constant() if isinstance(node, IndexedTensor) else False
            base_name = str(var)
            if not base_name.startswith('_'):
                base_name = Tensor.getBaseName(base_name)
            name = base_name
            counter = 1
            while name in var_name.values():
                name = f'{base_name}{counter}'
                counter = counter + 1
            var_name[var] = name
        if is_result:
            is_modified[var] = True

    w_name = lambda x: f'{self.W_PREFIX}{var_name[x]}'
    arg_name = lambda x: f'{self.ARG_PREFIX}{var_name[x]}'
    off_name = lambda x: f'{self.OFF_PREFIX}{var_name[x]}'
    memref_type = lambda ml: f'smm::ir::memref_type(real_t, {ml.bboxi(0).size()}, {ml.bboxi(1).size()}, {ml.stridei(1)})'
    has_offset = lambda var: not (var.is_temporary or is_constant[var])
    
    body = ''
    flops = 0
    for item in self._descr:
      node, args, add, scalar = item
      res, op1, op2 = args
      store_matrix(res, node, True)
      store_matrix(op1, node.leftTerm(), False)
      store_matrix(op2, node.rightTerm(), False)

      if res.is_temporary and res not in var_name:
        var_name[res] = str(res)
        body += f'auto {res} = bb.create_alloca({memref_type(node.memoryLayout())}, "{res}");\n'

      bbA = BoundingBox.fromSpp(node.leftTerm().eqspp())
      bbB = BoundingBox.fromSpp(node.rightTerm().eqspp())
      bbC = BoundingBox.fromSpp(node.eqspp())

      if node.transA() or node.transB():
          #raise NotImplementedError('Transposition not supported')
          print(f'WARNING: Transposition not supported yet in {res} = {op1} * {op2}')
      
      k = bbA[1] & bbB[0]
      m = bbA[0]
      n = bbB[1]

      if node.leftTerm().memoryLayout().alignedStride() and node.memoryLayout().alignedStride():
          m = m.aligned(self._arch)

      slic = lambda r, i: f'smm::ir::slice{{{i.start-r.start}, {i.stop-r.start}}}'
      name = lambda x: f'{w_name(x)}' if x in input_matrices else var_name[x]
      sub = lambda x, ml, i, j: f'    bb.create_submatrix({name(x)}, {slic(ml.bboxi(0), i)}, {slic(ml.bboxi(1), j)}),\n'

      body += f'bb.create_matmul(\n';
      body += sub(op1, node.leftTerm().memoryLayout(), m, k)
      body += sub(op2, node.rightTerm().memoryLayout(), k, n)
      body += sub(res, node.memoryLayout(), m, n)
      body += f'{scalar}, {1.0 if add else 0.0});\n'

      flops += 2 * m.size() * n.size() * k.size() 

    def batch_type(var):
        ml = input_matrices[var]
        if is_constant[var]:
            return f'{memref_type(ml)}'
        stride = f'smm::strided{{{ml.requiredReals()}}}' if var.is_temporary else 'smm::pointers{}'
        return f'smm::ir::batch_type({memref_type(ml)}, {stride})'
      
    pre_body = 'fb.body([&](smm::ir::block_builder& bb) {\n'
    for key in input_matrices.keys():
        if is_constant[key]:
            pre_body += f'auto {w_name(key)} = {arg_name(key)};\n'
        elif has_offset(key):
            pre_body += f'auto {w_name(key)} = bb.create_get_work_item({arg_name(key)}, {off_name(key)});\n'
        else:
            pre_body += f'auto {w_name(key)} = bb.create_get_work_item({arg_name(key)});\n'
    post_body = '});\n'

    args = f'constexpr auto real_t = smm::ir::to_scalar_type_v<{self._arch.typename}>;\n'
    for key in input_matrices.keys():
        args += f'auto {arg_name(key)} = fb.argument({batch_type(key)}, "{var_name[key]}");\n'
        if has_offset(key):
            args += f'auto {off_name(key)} = fb.argument(smm::ir::data_type(smm::ir::scalar_type::i32), "offset");\n'

    pre_header = 'static auto kernel = smm::custom_kernel([](smm::ir::function_builder &fb) {\n'
    post_header = f'}}, *static_cast<::sycl::queue*>({BatchedOperationsAux.STREAM_PTR_NAME}));\n'
    make_kernel = f'{pre_header}{args}{pre_body}{body}{post_body}{post_header}'

    def wrapper_type(key):
        ptr2ptr = '*' if not is_constant[key] and not key.is_temporary else ''
        const = ' const' if key not in is_modified and not key.is_temporary else ''
        return f'{self._arch.typename}{const}*{ptr2ptr}'

    hasher = hashlib.sha512()
    hasher.update(make_kernel.encode('utf-8'))
    wrapper_name = f'libsmm_wrapper_{hasher.hexdigest()}'
    wrapper_args = [f'unsigned {BatchedOperationsAux.NUM_ELEMENTS_NAME}', f'void* {BatchedOperationsAux.STREAM_PTR_NAME}']
    wrapper_call_args = []
    call_args = []
    for key in input_matrices.keys():
        ptr2ptr = '*' if not is_constant[key] and not key.is_temporary else ''
        const = ' const' if key not in is_modified and not key.is_temporary else ''
        wrapper_args += [f'{wrapper_type(key)} {var_name[key]}']
        wrapper_call_args += [var_name[key]]
        call_args += [f'const_cast<{wrapper_type(key)}>({str(key)})']
        if has_offset(key):
            offset_name = f'{BatchedOperationsAux.EXTRA_OFFSET_NAME}_{var_name[key]}' 
            wrapper_args += [f'int {offset_name}']
            wrapper_call_args += [offset_name]
            call_args += [f'{BatchedOperationsAux.EXTRA_OFFSET_NAME}_{key}']
    wrapper_call_args = ', '.join(wrapper_call_args)
    call_args = ', '.join(call_args)
    wrapper_signature = f'void {wrapper_name}({", ".join(wrapper_args)});'
    wrapper = f'{wrapper_signature[:-1]} {{\n'
    wrapper += make_kernel
    wrapper += f'kernel({BatchedOperationsAux.NUM_ELEMENTS_NAME}, {wrapper_call_args}).wait();\n'
    wrapper += '}\n\n'

    cpp(f'{wrapper_name}({BatchedOperationsAux.NUM_ELEMENTS_NAME}, {BatchedOperationsAux.STREAM_PTR_NAME}, {call_args});')

    routineCache.addRoutine(wrapper_signature, LibsmmWriter(wrapper_signature, wrapper))

    return flops

class LibsmmWriter(GpuRoutineGenerator):
  def __init__(self, signature, source):
    self._source = source
    self._signature = signature

  def __eq__(self, other):
    return self._signature == other._signature

  def header(self, cpp):
    cpp.include('smm/custom_kernel.hpp')
    cpp.include('smm/ir/builder.hpp')
    cpp.include('smm/ir/data_type.hpp')
    cpp.include('smm/ir/scalar_type.hpp')
    cpp.include('smm/ir/slice.hpp')
    cpp.includeSys('CL/sycl.hpp')

  def __call__(self, routineName, fileName):
    with open(fileName, 'a') as f:
      f.write(self._source)

    return self._signature
