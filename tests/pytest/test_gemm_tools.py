"""Which tool a GEMM goes to when its sparse operand stores more than it multiplies.

A GEMM takes the rows it multiplies from where its operands overlap. A CSC
layout keeps whole columns, so in the columns the GEMM reads it can store
entries outside those rows, whose values lie among the ones the GEMM reads.
libxsmm takes the pattern it is given as all of the operand and rejects
entries outside the GEMM; such a GEMM goes to another tool, or to the code
yateto writes itself, which steps over them.
"""
from __future__ import annotations

import numpy as np

from yateto import Generator, GlobalRoutineCache, Tensor, useArchitectureIdentifiedBy
from yateto.ast.indices import Range
from yateto.gemm_configuration import DENSE, LIBXSMM, GeneratorCollection, Sparsity
from yateto.memory import CSCMemoryLayout
from yateto.type import Datatype


def operands(outside=True):
    """C = A B, where A multiplies through its column 1 only: the GEMM takes
    row 1 of B. B keeps row 3 of its first column as well, unless `outside`
    is false."""
    sppA = np.zeros((8, 4))
    sppA[:, 1] = 1
    sppB = np.zeros((4, 2))
    sppB[1, 0] = sppB[1, 1] = 1
    if outside:
        sppB[3, 0] = 1
    A = Tensor('A', (8, 4), spp=sppA)
    B = Tensor('B', (4, 2), spp=sppB, memoryLayoutClass=CSCMemoryLayout)
    C = Tensor('C', (8, 2))
    return A, B, C


def generate(tmp_path, outside=True):
    """The routines the kernel C = A B calls, with libxsmm as the only tool, and its code."""
    arch = useArchitectureIdentifiedBy('dsnb')
    A, B, C = operands(outside)
    g = Generator(arch)
    g.add('k', C['ij'] <= A['ik'] * B['kj'])
    cache = GlobalRoutineCache()
    g.generate(outputDir=str(tmp_path), gemm_cfg=GeneratorCollection([LIBXSMM(arch)]),
               routine_cache=cache)
    return [name for name, _ in cache.cache.routines()], (tmp_path / 'kernel.cpp').read_text()


def libxsmmTakes(sparseB):
    libxsmm = LIBXSMM(useArchitectureIdentifiedBy('dsnb'))
    return libxsmm.supported(8, 2, 1, DENSE, sparseB, False, False, 1.0, 0.0, True, True,
                             Datatype.F64, Datatype.F64, Datatype.F64, 'cpu')


class TestEntriesOutsideTheGemm:
    def test_a_layout_tells_the_entries_outside_the_rows(self):
        _, B, _ = operands()
        layout = B.memoryLayout()

        assert layout.storesOutside(Range(1, 2), Range(0, 2))
        assert not layout.storesOutside(Range(1, 4), Range(0, 2))
        assert not layout.storesOutside(Range(1, 2), Range(1, 2))

    def test_libxsmm_leaves_such_an_operand_to_another_tool(self):
        _, B, _ = operands()

        assert not libxsmmTakes(Sparsity.of(B.memoryLayout(), Range(1, 2), Range(0, 2)))
        assert libxsmmTakes(Sparsity.of(B.memoryLayout(), Range(1, 4), Range(0, 2)))

    def test_the_kernel_reads_the_values_by_the_stored_pattern(self, tmp_path):
        routines, kernel = generate(tmp_path)

        assert not [name for name in routines if name.startswith('libxsmm')]
        # row 1 of each column: the first value of column 0, and the value of
        # column 1 behind the two values of column 0
        assert '* B[0 + 0];' in kernel
        assert '* B[0 + 2];' in kernel

    def test_an_operand_without_entries_outside_goes_to_libxsmm(self, tmp_path):
        routines, kernel = generate(tmp_path, outside=False)

        assert [name for name in routines if name.startswith('libxsmm_bsparse_')]
