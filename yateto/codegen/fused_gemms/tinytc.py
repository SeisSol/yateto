from ..common import TensorDescription, IndexedTensorDescription, BatchedOperationsAux
from ...ast.indices import BoundingBox
from ..cache import TinytcWriter
from ...ast.node import IndexedTensor
from ...type import Tensor

import hashlib


class FusedGemmsTinytc:
  def __init__(self, arch, descr):
    self._arch = arch
    self._descr = descr
    self._batch_aux = BatchedOperationsAux(self._arch.typename)
    self._cache = {}
    self._tmp_matrices = {}
    self._scalar_type = 'f64' if self._arch.bytesPerReal == 8 else 'f32'
    self._var_counter = 0

  def next_var(self):
    count = self._var_counter
    self._var_counter += 1
    return count

  def generate(self, cpp, routineCache, cfg):
    input_matrices = dict()
    is_constant = dict()
    is_modified = dict()
    var_name = dict()
    work_item_name = dict()
    self._var_counter = 0

    def store_matrix(var, node, is_result):
        if var not in var_name:
            if var.is_temporary and is_result:
                var_name[res] = 'tmp'
            else:
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

    def batch_type(var):
        ml = input_matrices[var]
        if is_constant[var]:
            return f'{memref_type(ml)}'
        elif var.is_temporary:
            return f'{batch_memref_type(ml)}'
        else:
            return f'group<{memref_type(ml)}, offset: ?>'
 

    memref_type = lambda ml: f'memref<{self._scalar_type}x{ml.bboxi(0).size()}x{ml.bboxi(1).size()},strided<1,{ml.stridei(1)}>>'
    batch_memref_type = lambda ml: f'memref<{self._scalar_type}x{ml.bboxi(0).size()}x{ml.bboxi(1).size()}x?,strided<1,{ml.stridei(1)},{ml.requiredReals()}>>'

    for item in self._descr:
      node, args, _, _ = item
      res, op1, op2 = args
      store_matrix(res, node, True)
      store_matrix(op1, node.leftTerm(), False)
      store_matrix(op2, node.rightTerm(), False)

    args = [f'%{var_name[key]}: {batch_type(key)}' for key in input_matrices.keys()]
    args_str = ',\n    '.join(args)
    source = f'func @fused_gemm({args_str}) {{\n'

    source += f'    %{self.next_var()} = group_id\n'
    gid = self._var_counter-1
    for key in input_matrices.keys():
        if not is_constant[key]:
            new_var = self.next_var()
            if key.is_temporary:
                source += f'    %{new_var} = load %{var_name[key]}[:,:,%{gid}] : {batch_type(key)}\n'
            else:
                source += f'    %{new_var} = load %{var_name[key]}[%{gid}] : {batch_type(key)}\n'
            work_item_name[key] = str(new_var)
    
    flops = 0
    for item in self._descr:
      node, args, add, scalar = item
      res, op1, op2 = args

      if res.is_temporary:
        var_name[res] = f'tmp{self.next_var()}'
        source += f'    %{var_name[res]} = alloca -> {memref_type(node.memoryLayout())}\n'

      bbA = BoundingBox.fromSpp(node.leftTerm().eqspp())
      bbB = BoundingBox.fromSpp(node.rightTerm().eqspp())
      bbC = BoundingBox.fromSpp(node.eqspp())
      
      k_op1 = 0 if node.transA() else 1
      k_op2 = 1 if node.transB() else 0
      k = bbA[k_op1] & bbB[k_op2]
      m = bbA[1 - k_op1]
      n = bbB[1 - k_op2]

      if not node.transA() and node.leftTerm().memoryLayout().alignedStride() and node.memoryLayout().alignedStride():
          m = m.aligned(self._arch)

      slic = lambda r, i: f'{i.start-r.start}:{i.stop-i.start}'
      name = lambda var: work_item_name[var] if var in work_item_name else var_name[var]
      subview = lambda var, ml, range1, range2: (f'    %{self.next_var()} = subview %{name(var)}[{slic(ml.bboxi(0), range1)},{slic(ml.bboxi(1), range2)}] : {memref_type(ml)}\n', f'memref<{self._scalar_type}x{range1.stop-range1.start}x{range2.stop-range2.start},strided<1,{ml.stridei(1)}>>')
      trans = lambda t: 't' if t else 'n'

      op1_sub, op1_sub_ty = subview(op1, node.leftTerm().memoryLayout(), m, k)
      op2_sub, op2_sub_ty = subview(op2, node.rightTerm().memoryLayout(), k, n)
      res_sub, res_sub_ty = subview(res, node.memoryLayout(), m, n)
      source += op1_sub + op2_sub + res_sub
      source += f'    gemm.{trans(node.transA())}.{trans(node.transB())} {scalar}, %{self._var_counter-3}, %{self._var_counter-2}, {1.0 if add else 0.0}, %{self._var_counter-1} : {self._scalar_type}, {op1_sub_ty}, {op2_sub_ty}, {self._scalar_type}, {res_sub_ty}\n';

      flops += 2 * m.size() * n.size() * k.size() 

    source += '}\n'

    make_kernel = """    struct custom_kernel { ::sycl::kernel kernel; ::sycl::range<3u> group_size; };
    static auto k = [&] (::sycl::queue const& queue) -> custom_kernel {
        static const std::string source = R\"tinytc(
"""
    make_kernel += source
    make_kernel += """)tinytc\";
    auto source_ctx = tinytc::make_source_context();
        try {
	        auto program = tinytc::parse_string(source, source_ctx);
            auto bundle = tinytc::make_kernel_bundle(queue.get_context(), queue.get_device(), std::move(program), 0, source_ctx);
            auto kernel = tinytc::make_kernel(bundle, "fused_gemm");
            auto group_size = tinytc::get_group_size(kernel);
            return {std::move(kernel), std::move(group_size)};
        } catch (tinytc::status const& st) {
            throw std::runtime_error(source_ctx.get_error_log());
        }
    }""";
    make_kernel += f'(*static_cast<::sycl::queue*>({BatchedOperationsAux.STREAM_PTR_NAME}));\n'

    def wrapper_type(key):
        ptr2ptr = '*' if not is_constant[key] and not key.is_temporary else ''
        const = ' const' if key not in is_modified and not key.is_temporary else ''
        return f'{self._arch.typename}{const}*{ptr2ptr}'

    hasher = hashlib.sha512()
    hasher.update(make_kernel.encode('utf-8'))
    wrapper_name = f'tinytc_wrapper_{hasher.hexdigest()}'
    wrapper_args = [f'unsigned {BatchedOperationsAux.NUM_ELEMENTS_NAME}', f'void* {BatchedOperationsAux.STREAM_PTR_NAME}']
    wrapper_call_args = []
    call_args = []
    for key in input_matrices.keys():
        ptr2ptr = '*' if not is_constant[key] and not key.is_temporary else ''
        const = ' const' if key not in is_modified and not key.is_temporary else ''
        wrapper_args += [f'{wrapper_type(key)} {var_name[key]}']
        wrapper_call_args += [var_name[key]]
        call_args += [f'const_cast<{wrapper_type(key)}>({str(key)})']
        if key.is_temporary:
            wrapper_call_args.append(BatchedOperationsAux.NUM_ELEMENTS_NAME)
        elif not is_constant[key]:
            offset_name = f'{BatchedOperationsAux.EXTRA_OFFSET_NAME}_{var_name[key]}' 
            wrapper_args.append(f'int {offset_name}')
            wrapper_call_args.append(offset_name)
            call_args.append(f'{BatchedOperationsAux.EXTRA_OFFSET_NAME}_{key}')
    wrapper_call_args = ', '.join(wrapper_call_args)
    call_args = ', '.join(call_args)
    wrapper_signature = f'void {wrapper_name}({", ".join(wrapper_args)});'
    wrapper = f'{wrapper_signature[:-1]} {{\n'
    wrapper += make_kernel
    wrapper += f'    static_cast<::sycl::queue*>({BatchedOperationsAux.STREAM_PTR_NAME})->submit([&](::sycl::handler &h) {{\n';
    wrapper += f'        h.set_args({wrapper_call_args});\n'
    wrapper += f'        h.parallel_for(::sycl::nd_range{{tinytc::get_global_size({BatchedOperationsAux.NUM_ELEMENTS_NAME}, k.group_size), k.group_size}}, k.kernel);\n'
    wrapper +=  '    });\n'
    wrapper += '}\n\n'

    cpp(f'{wrapper_name}({BatchedOperationsAux.NUM_ELEMENTS_NAME}, {BatchedOperationsAux.STREAM_PTR_NAME}, {call_args});')

    routineCache.addRoutine(wrapper_signature, TinytcWriter(wrapper_signature, wrapper))

    return flops
