from ..common import TinytcKernelArgument, TinytcScalarKernelArgument, TinytcWrapper, makeMemrefType, makeBatchType, makeLoad, toTinyTCType, toTinyTCImmediate
from ...ast.indices import BoundingBox
from ..cache import TinytcWriter
from ...type import Tensor
from ..tiny_tensor_language import *

import hashlib


class FusedGemmsTinytc:

    def __init__(self, arch, descr):
        self._arch = arch
        self._descr = descr

    def generate(self, cpp, routineCache):
        args = dict()
        vals = dict()
        tensors = dict()
        is_constant = dict()
        modified = set()
        bb = RegionBuilder()
        gid = bb.add(GroupIdInst())

        def addVal(tensor):
            if tensor.name not in vals:
                name = tensor.name
                if not name.startswith('_'):
                    groups = Tensor.getGroup(name)
                    name = Tensor.getBaseName(name)
                    if groups:
                        name += '_' + '_'.join(str(g) for g in groups)
                else:
                    # Names starting with underscore are illegal in tinytc
                    name = ''
                is_constant[tensor.name] = tensor.is_compute_constant
                arg = LocalValue(
                    makeBatchType(toTinyTCType(tensor.datatype), tensor.memoryLayout,
                                  tensor.is_compute_constant, tensor.is_temporary), name)
                args[tensor.name] = arg
                tensors[tensor.name] = tensor
                vals[tensor.name] = makeLoad(bb, arg, gid, tensor.is_compute_constant,
                                             tensor.is_temporary)

        flops = 0
        for statement in self._descr:
            res, op1, op2 = statement.result, *statement.terms

            addVal(op1)
            op1_val = vals[op1.name]
            addVal(op2)
            op2_val = vals[op2.name]

            res_val = None
            if res.is_temporary:
                res_val = bb.add(
                    AllocaInst(
                        makeMemrefType(toTinyTCType(res.datatype), res.memoryLayout, False, True)))
                vals[res.name] = res_val
            else:
                modified.add(res.name)
                addVal(res)
                res_val = vals[res.name]

            bbA = BoundingBox.fromSpp(op1.eqspp)
            bbB = BoundingBox.fromSpp(op2.eqspp)

            k_op1 = 0 if statement.transA else 1
            k_op2 = 1 if statement.transB else 0
            k = bbA[k_op1] & bbB[k_op2]
            m = bbA[1 - k_op1]
            n = bbB[1 - k_op2]

            if not statement.transA and op1.memoryLayout.alignedStride() \
               and res.memoryLayout.alignedStride():
                m = m.aligned(self._arch)

            def offsetSizeLists(ml, range0, range1):
                offsets = (range0.start - ml.bboxi(0).start,
                           range1.start - ml.bboxi(1).start)
                sizes = (range0.size(), range1.size())
                return ([IntImmValue(IntegerType.index, o) for o in offsets],
                        [IntImmValue(IntegerType.index, s) for s in sizes])

            alpha = bb.add(
                ConstantInst(toTinyTCImmediate(toTinyTCType(res.datatype), statement.alpha)))
            op1_sub = bb.add(
                SubviewInst(op1_val, *offsetSizeLists(op1.memoryLayout, m, k)))
            op2_sub = bb.add(
                SubviewInst(op2_val, *offsetSizeLists(op2.memoryLayout, k, n)))
            beta = bb.add(
                ConstantInst(toTinyTCImmediate(toTinyTCType(res.datatype),
                                               1.0 if statement.add else 0.0)))
            res_sub = bb.add(
                SubviewInst(res_val, *offsetSizeLists(res.memoryLayout, m, n)))

            trans = lambda t: Transpose.t if t else Transpose.n
            bb.add(
                GemmInst(trans(statement.transA), trans(statement.transB), alpha,
                         op1_sub, op2_sub, beta, res_sub))

            flops += 2 * m.size() * n.size() * k.size()

        ast = bb.get_product()
        hash_ = hashlib.sha256(Dump().visit(ast).encode()).hexdigest()
        kernel = Function(f'fused_gemm_{hash_}', args.values(), ast)
        AssignIdentifiers().visit(kernel)

        wrapper_args = []
        for key, val in args.items():
            name = f'_tmp{val.name}' if val.name.isnumeric() else val.name
            wrapper_args.append(
                TinytcKernelArgument(name, key, is_constant[key],
                                     tensors[key].is_temporary, key in modified))
        wrapper = TinytcWrapper(kernel, wrapper_args)
        cpp(wrapper.call())
        prototype = wrapper.prototype()
        routineCache.addRoutine(prototype,
                                TinytcWriter(prototype, wrapper.definition()))

        return flops
