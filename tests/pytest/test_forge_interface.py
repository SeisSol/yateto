"""What yateto hands to gemmforge and chainforge.

Neither is a dependency of the test suite, so these tests check yateto's side
of the interface: the spelling of an addressing mode the forges read, and the
description the fused-GEMM factory builds for them.
"""

import io

import pytest

from yateto.arch import useArchitectureIdentifiedBy
from yateto.codegen import fused_gemms
from yateto.codegen.code import Cpp
from yateto.codegen.common import BatchedOperationsAux
from yateto.codegen.factory import OptimizedKernelFactory
from yateto.type import AddressingMode


class Term:
    def __init__(self, addressing=None, is_compute_constant=False, is_temporary=False):
        self.addressing = addressing
        self.is_compute_constant = is_compute_constant
        self.is_temporary = is_temporary


class TestForgeAddressing:
    """gemmforge and chainforge name the modes as strings of their own."""

    @pytest.mark.parametrize('term, spelled', [
        (Term(is_compute_constant=True), 'none'),
        (Term(is_temporary=True), 'strided'),
        (Term(), 'pointer_based'),
        (Term(addressing=AddressingMode.DIRECT), 'none'),
        (Term(addressing=AddressingMode.STRIDED), 'strided'),
        (Term(addressing=AddressingMode.INDIRECT), 'pointer_based'),
    ])
    def test_the_modes_they_read_are_spelled_their_way(self, term, spelled):
        assert BatchedOperationsAux.forge_addressing(term) == spelled

    @pytest.mark.parametrize('mode', [AddressingMode.SCALAR, AddressingMode.IMMEDIATE])
    def test_a_mode_they_cannot_read_is_refused(self, mode):
        with pytest.raises(ValueError, match='cannot read an operand addressed as'):
            BatchedOperationsAux.forge_addressing(Term(addressing=mode))


class TestFusedGemmsDescription:
    """The factory describes a chain of GEMMs the way the generators iterate it."""

    def test_the_factory_builds_a_description_the_generator_can_read(self, monkeypatch):
        seen = {}

        class Generator:
            def __init__(self, descr):
                seen['descr'] = descr

            def generate(self, cpp, routineCache, gemm_cfg):
                return 0

        monkeypatch.setattr(fused_gemms, 'generator',
                            lambda arch, descr, gemm_cfg, target, attrs=None: Generator(descr))

        arch = useArchitectureIdentifiedBy('dhsw', 'dsm_86', 'cuda')
        factory = OptimizedKernelFactory(Cpp(io.StringIO()), arch, 'gpu')
        node, result, arguments = object(), object(), ['C', 'A', 'B']
        # fused actions carry one guard per GEMM
        assert factory.create_FusedGEMMs(node, result, arguments, [True], [False], [1.0],
                                         None, None, None) == 0

        descr = seen['descr']
        assert descr.node is node and descr.result is result
        assert descr.args == arguments
        assert descr.add == [False] and descr.scalar == [1.0]
