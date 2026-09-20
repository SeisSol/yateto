"""The identity of a GEMM routine, and that it is carried by data alone."""

from __future__ import annotations

import json

import numpy as np
import pytest

from yateto import useArchitectureIdentifiedBy
from yateto.codegen.gemm.descriptor import GemmDescriptor
from yateto.codegen.gemm.gemmgen import GemmGen
from yateto.gemm_configuration import LIBXSMM, PSpaMM
from yateto.type import Datatype


def gemmParameters(**overrides):
    gemm = dict(M=8, N=4, K=8, LDA=8, LDB=8, LDC=8,
                alpha='1', beta='0',
                alignedA=1, alignedC=1, prefetch='nopf',
                transA=False, transB=False,
                datatypeA=Datatype.F64, datatypeB=Datatype.F64, datatypeC=Datatype.F64)
    gemm.update(overrides)
    return gemm


def descriptor(sppA=None, sppB=None, operation='libxsmm', arch='hsw', **overrides):
    return GemmDescriptor.create(operation, arch, gemmParameters(**overrides), sppA, sppB)


class TestRoutineName:
    def test_a_dense_gemm(self):
        assert descriptor().routineName() == (
            'libxsmm_dense_f64_f64_f64_hsw_m8_n4_k8_ldA8_ldB8_ldC8'
            '_alpha1_beta0_alignedA1_alignedC1_transAFalse_transBFalse_nopf')

    def test_a_sparse_left_operand(self):
        assert descriptor(sppA=[(0, 0), (3, 1)]).routineName() == (
            'libxsmm_asparse_cab8b866bab1cba6f5e8e58671ad23d4'
            '_f64_f64_f64_hsw_m8_n4_k8_ldA8_ldB8_ldC8'
            '_alpha1_beta0_alignedA1_alignedC1_transAFalse_transBFalse_nopf')

    @pytest.mark.parametrize("sppA,sppB,expected", [
        (None, None, '_dense_'),
        ([(0, 0)], None, '_asparse_'),
        (None, [(0, 0)], '_bsparse_'),
        ([(0, 0)], [(0, 0)], '_absparse_'),
    ])
    def test_the_sparsity_is_in_the_name(self, sppA, sppB, expected):
        assert expected in descriptor(sppA=sppA, sppB=sppB).routineName()

    def test_differing_patterns_are_named_apart(self):
        first = descriptor(sppA=[(0, 0), (3, 1)]).routineName()
        second = descriptor(sppA=[(0, 0), (4, 1)]).routineName()

        assert first != second

    def test_the_generator_is_in_the_name(self):
        assert descriptor(operation='pspamm').routineName().startswith('pspamm_')

    def test_a_hyphenated_architecture_stays_an_identifier(self):
        name = descriptor(arch='thunderx2-t99').routineName()

        assert 'thunderx2_t99' in name
        assert '-' not in name


class TestTheNameDoesNotDependOnHowIntegersArrive:
    """Index positions come out of the layouts as the array library's integers.

    Were that spelling to reach the name, the same kernel would be called
    different things depending on which versions are installed, and a routine
    cache shared across environments would miss every time.
    """

    def test_array_scalars_and_plain_integers_agree(self):
        fromArray = descriptor(sppA=[(np.int64(0), 0), (np.int64(3), 1)])
        fromPython = descriptor(sppA=[(0, 0), (3, 1)])

        assert fromArray == fromPython
        assert fromArray.routineName() == fromPython.routineName()

    def test_the_pattern_is_held_as_plain_integers(self):
        held = descriptor(sppA=[(np.int64(0), np.int64(3))]).sppA

        assert held == ((0, 3),)
        assert all(type(i) is int for entry in held for i in entry)

    def test_a_datatype_is_held_by_its_name(self):
        assert descriptor().datatypeA == 'f64'


class TestItSurvivesSerialisation:
    """A routine named where the kernel is written, emitted somewhere else."""

    def test_a_round_trip_changes_nothing(self):
        original = descriptor(sppA=[(0, 0), (3, 1)], sppB=[(1, 2)])

        restored = GemmDescriptor.fromJson(json.loads(json.dumps(original.asJson())))

        assert restored == original
        assert restored.routineName() == original.routineName()

    def test_a_dense_round_trip_changes_nothing(self):
        original = descriptor()

        restored = GemmDescriptor.fromJson(json.loads(json.dumps(original.asJson())))

        assert restored == original

    def test_nothing_but_plain_types_comes_out(self):
        allowed = (int, float, str, bool, type(None), list)

        for name, value in descriptor(sppA=[(0, 0)]).asJson().items():
            assert isinstance(value, allowed), name
            if isinstance(value, list):
                for entry in value:
                    assert all(isinstance(i, int) for i in entry), name


class TestTheGeneratorAgrees:
    def test_the_gemm_generator_describes_what_it_calls(self):
        arch = useArchitectureIdentifiedBy('dhsw')
        for cfg in (LIBXSMM(arch), PSpaMM(arch)):
            gen = GemmGen(arch, None, cfg)

            described = gen.describe(gemmParameters(), None, None)

            assert described.operation == cfg.operation_name
            assert described.arch == 'hsw'


def description(arch, layoutA, layoutC=None, sppA=None):
    """A GEMM of a 6x8 by 8x4, with A's arrangement left to the caller."""
    import numpy as np
    from yateto import Tensor
    from yateto.codegen.common import TensorDescription
    from yateto.codegen.gemm.factory import Description
    from yateto.memory import DenseMemoryLayout

    values = np.ones((6, 8)) if sppA is None else sppA
    a = Tensor('a', (6, 8), values, alignStride=True)
    a.setMemoryLayout(layoutA, alignStride=True)
    b = Tensor('b', (8, 4))
    c = Tensor('c', (6, 4))
    if layoutC is not None:
        c.setMemoryLayout(layoutC, alignStride=True)

    def td(t):
        return TensorDescription(t.name(), t.memoryLayout(), t.spp(),
                                 datatype=t.getDatatype(arch))

    return Description(result=td(c), leftTerm=td(a), rightTerm=td(b),
                       transA=False, transB=False, alpha=1.0, beta=0.0,
                       arch=arch, alignedStartA=True, alignedStartC=True)


class TestTheQuestionHoldsNoAnswer:
    """A query says what the operation is, not how its operands are arranged."""

    def _query(self, archName):
        from yateto.codegen.gemm.descriptor import GemmQuery
        from yateto.memory import DenseMemoryLayout

        arch = useArchitectureIdentifiedBy(archName)
        return GemmQuery.create(arch, description(arch, DenseMemoryLayout))

    def test_it_carries_the_unwidened_extents(self):
        query = self._query('dhsw')

        assert query.m == [0, 6]
        assert query.n == [0, 4]
        assert query.k == [0, 8]

    def test_widening_the_extent_does_not_reach_it(self):
        """Padding m out to a vector boundary is arrangement, not operation."""
        narrow = self._query('dhsw')
        wide = self._query('dskx')

        assert narrow.m == wide.m
        assert narrow.alignedReals != wide.alignedReals

    def test_it_names_no_leading_dimension_and_no_alignment(self):
        query = self._query('dhsw')

        spelled = ' '.join(query._fields)
        for answer in ('ld', 'aligned', 'sparsity', 'encoding'):
            assert answer not in spelled.replace('alignedReals', '')

    def test_a_dense_operand_has_no_pattern(self):
        query = self._query('dhsw')

        assert query.patternA is None
        assert query.patternB is None

    def test_holes_in_an_operand_are_reported(self):
        import numpy as np
        from yateto.codegen.gemm.descriptor import GemmQuery
        from yateto.memory import DenseMemoryLayout

        values = np.zeros((6, 8))
        values[0, 0] = values[3, 1] = 1.0
        arch = useArchitectureIdentifiedBy('dhsw')

        query = GemmQuery.create(arch, description(arch, DenseMemoryLayout, sppA=values))

        assert query.patternA == ((0, 0), (3, 1))

    def test_how_the_operand_is_stored_does_not_reach_it(self):
        """The same holes, once stored densely and once compressed."""
        import numpy as np
        from yateto.codegen.gemm.descriptor import GemmQuery
        from yateto.memory import CSCMemoryLayout, DenseMemoryLayout

        values = np.zeros((6, 8))
        values[0, 0] = values[3, 1] = 1.0
        arch = useArchitectureIdentifiedBy('dhsw')

        asDense = GemmQuery.create(arch, description(arch, DenseMemoryLayout, sppA=values))
        asCsc = GemmQuery.create(arch, description(arch, CSCMemoryLayout, sppA=values))

        assert asDense == asCsc

    def test_a_round_trip_changes_nothing(self):
        import numpy as np
        from yateto.codegen.gemm.descriptor import GemmQuery
        from yateto.memory import DenseMemoryLayout

        values = np.zeros((6, 8))
        values[0, 0] = values[3, 1] = 1.0
        arch = useArchitectureIdentifiedBy('dhsw')
        original = GemmQuery.create(arch, description(arch, DenseMemoryLayout, sppA=values))

        restored = GemmQuery.fromJson(json.loads(json.dumps(original.asJson())))

        assert restored == original
