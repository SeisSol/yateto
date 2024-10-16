# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

from ..common import TensorDescription, IndexedTensorDescription, BatchedOperationsAux, TinytcKernelArgument, TinytcScalarKernelArgument, TinytcWrapper, makeMemrefType, makeBatchType, makeLoad
from ..cache import TinytcWriter
from ..tiny_tensor_language import *


class CopyScaleAddTinytc:

    def __init__(self, arch, descr):
        self._arch = arch
        self._descr = descr
        self._scalar_type = 'f64' if self._arch.bytesPerReal == 8 else 'f32'
        self._ty = ScalarType(
            FloatingType.f64) if self._arch.bytesPerReal == 8 else ScalarType(
                FloatingType.f32)

    def generate(self, cpp, routineCache):
        d = self._descr

        # Order can be 1 or 2
        def MakeLoopOverAxpby(d, order, transpose, A, B):
            beta = FloatImmValue(self._ty, d.beta)
            A_offset_list = [None] * len(d.term.indices)
            A_size_list = [None] * len(d.term.indices)
            B_offset_list = [None] * len(d.result.indices)
            B_size_list = [None] * len(d.result.indices)

            for j in range(0, order):
                idx = d.result.indices[j]
                j_perm = d.term.indices.find(d.result.indices[j])
                offset = IntImmValue(IntegerType.index,
                                     d.loopRanges[idx].start)
                size = IntImmValue(
                    IntegerType.index,
                    d.loopRanges[idx].stop - d.loopRanges[idx].start)
                A_offset_list[j_perm] = offset
                A_size_list[j_perm] = size
                B_offset_list[j] = offset
                B_size_list[j] = size

            for j in range(order, len(d.result.indices)):
                loop_var = LocalValue(ScalarType(IntegerType.index))
                j_perm = d.term.indices.find(d.result.indices[j])
                A_offset_list[j_perm] = loop_var
                B_offset_list[j] = loop_var

            csa_bb = RegionBuilder()
            a = csa_bb.add(SubviewInst(A, A_offset_list, A_size_list))
            b = csa_bb.add(SubviewInst(B, B_offset_list, B_size_list))
            csa_bb.add(AxpbyInst(trans, alpha, a, beta, b))
            csa_region = csa_bb.get_product()

            for j in range(2, len(d.result.indices)):
                idx = d.result.indices[j]
                loop_var = LocalValue(ScalarType(IntegerType.index))
                start = IntImmValue(IntegerType.index, d.loopRanges[idx].start)
                stop = IntImmValue(IntegerType.index, d.loopRanges[idx].stop)
                csa_region = Region(
                    [ForInst(B_offset_list[j], start, stop, csa_region)])
            return csa_region

        alpha = LocalValue(self._ty, 'alpha')
        Abatch = LocalValue(
            makeBatchType(self._ty, d.term.memoryLayout,
                          d.term.is_compute_constant, d.term.is_temporary),
            'A')
        Bbatch = LocalValue(
            makeBatchType(self._ty, d.result.memoryLayout,
                          d.result.is_compute_constant, d.result.is_temporary),
            'B')
        kernel = Function('copyscaleadd', [alpha, Abatch, Bbatch], None)

        bb = RegionBuilder()
        gid = bb.add(GroupIdInst())
        A = makeLoad(bb, Abatch, gid, d.term.is_compute_constant,
                     d.term.is_temporary)
        B = makeLoad(bb, Bbatch, gid, d.result.is_compute_constant,
                     d.result.is_temporary)

        trans = Transpose.n
        if len(d.result.indices) == 0:
            raise NotImplementedError
        if len(d.result.indices) == 1:
            bb.extend(MakeLoopOverAxpby(d, 1, Transpose.n, A, B))
        if len(d.result.indices) >= 2:
            i0 = d.result.indices[0]
            i1 = d.result.indices[1]
            trans = Transpose.t if d.term.indices.find(
                i0) > d.term.indices.find(i1) else Transpose.n
            bb.extend(MakeLoopOverAxpby(d, 2, trans, A, B))

        kernel.body = bb.get_product()
        AssignIdentifiers().visit(kernel)

        args = [
            TinytcScalarKernelArgument('alpha', str(d.alpha)),
            TinytcKernelArgument('A', d.term.name, d.term.is_compute_constant,
                                 d.term.is_temporary, False),
            TinytcKernelArgument('B', d.result.name,
                                 d.result.is_compute_constant,
                                 d.result.is_temporary, True)
        ]
        wrapper = TinytcWrapper(kernel, args, self._arch.typename)

        prototype = wrapper.prototype()
        routineCache.addRoutine(prototype,
                                TinytcWriter(prototype, wrapper.definition()))

        cpp(wrapper.call())

        flops = 1
        if d.beta != 0.0:
            flops += 1
        for rng in d.loopRanges.values():
            flops *= rng.size()
        return flops
