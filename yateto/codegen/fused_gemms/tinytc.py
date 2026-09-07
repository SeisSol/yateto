from ..common import TinytcKernelArgument, TinytcScalarKernelArgument, TinytcWrapper, makeMemrefType, makeBatchType, makeLoad
from ...ast.indices import BoundingBox
from ..cache import TinytcWriter
from ...ast.node import IndexedTensor
from ...type import Tensor
from ..tiny_tensor_language import *

import hashlib


class FusedGemmsTinytc:

    def __init__(self, arch, descr):
        self._arch = arch
        self._descr = descr

    def generate(self, cpp, routineCache, cfg):
        args = dict()
        vals = dict()
        is_constant = dict()
        modified = set()
        bb = RegionBuilder()
        gid = bb.add(GroupIdInst())

        def addVal(var, node):
            if var not in vals:
                name = str(var)
                if not name.startswith('_'):
                    groups = Tensor.getGroup(name)
                    name = Tensor.getBaseName(name)
                    if groups:
                        name += '_' + '_'.join(str(g) for g in groups)
                else:
                    # Names starting with underscore are illegal in tinytc
                    name = ''
                is_constant[var] = node.tensor.is_compute_constant(
                ) if isinstance(node, IndexedTensor) else False
                arg = LocalValue(
                    makeBatchType(toTinyTCType(var.datatype), node.memoryLayout(),
                                  is_constant[var], var.is_temporary), name)
                args[var] = arg
                vals[var] = makeLoad(bb, arg, gid, is_constant[var], var.is_temporary)

        flops = 0
        for item in self._descr:
            node, variables, add, scalar = item
            res, op1, op2 = variables

            addVal(op1, node.leftTerm())
            op1_val = vals[op1]
            addVal(op2, node.rightTerm())
            op2_val = vals[op2]

            res_batch = None
            res_val = None
            if res.is_temporary:
                res_val = bb.add(
                    AllocaInst(
                        makeMemrefType(toTinyTCType(res.datatype), res.memoryLayout(), False, True)))
                vals[res] = res_val
            else:
                modified.add(res)
                addVal(res, node)
                res_val = vals[res]

            bbA = BoundingBox.fromSpp(node.leftTerm().eqspp())
            bbB = BoundingBox.fromSpp(node.rightTerm().eqspp())
            bbC = BoundingBox.fromSpp(node.eqspp())

            k_op1 = 0 if node.transA() else 1
            k_op2 = 1 if node.transB() else 0
            k = bbA[k_op1] & bbB[k_op2]
            m = bbA[1 - k_op1]
            n = bbB[1 - k_op2]

            if not node.transA() and node.leftTerm().memoryLayout(
            ).alignedStride() and node.memoryLayout().alignedStride():
                m = m.aligned(self._arch)

            def offsetSizeLists(ml, range0, range1):
                offsets = (range0.start - ml.bboxi(0).start,
                           range1.start - ml.bboxi(1).start)
                sizes = (range0.size(), range1.size())
                return ([IntImmValue(IntegerType.index, o) for o in offsets],
                        [IntImmValue(IntegerType.index, s) for s in sizes])

            alpha = bb.add(ConstantInst(toTinyTCImmediate(toTinyTCType(res.datatype), scalar)))
            op1_sub = bb.add(
                SubviewInst(
                    op1_val,
                    *offsetSizeLists(node.leftTerm().memoryLayout(), m, k)))
            op2_sub = bb.add(
                SubviewInst(
                    op2_val,
                    *offsetSizeLists(node.rightTerm().memoryLayout(), k, n)))
            beta = bb.add(ConstantInst(toTinyTCImmediate(toTinyTCType(res.datatype), 1.0 if add else 0.0)))
            res_sub = bb.add(
                SubviewInst(res_val,
                            *offsetSizeLists(node.memoryLayout(), m, n)))

            trans = lambda t: Transpose.t if t else Transpose.n
            bb.add(
                GemmInst(trans(node.transA()), trans(node.transB()), alpha,
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
                TinytcKernelArgument(name, str(key), is_constant[key],
                                     key.is_temporary, key in modified))
        wrapper = TinytcWrapper(kernel, wrapper_args)
        cpp(wrapper.call())
        prototype = wrapper.prototype()
        routineCache.addRoutine(prototype,
                                TinytcWriter(prototype, wrapper.definition()))

        return flops
