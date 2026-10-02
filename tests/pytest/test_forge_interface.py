"""What yateto hands to gemmforge and chainforge.

Neither is a dependency of the test suite, so these tests check yateto's side
of the interface: the spelling of an addressing mode the forges read, the
description the fused-GEMM factory builds for them, and how a GEMM's scale
factor reaches gemmforge.
"""

import io

import pytest

from yateto.arch import useArchitectureIdentifiedBy
from yateto.codegen import fused_gemms
from yateto.codegen.code import Cpp
from yateto.codegen.common import BatchedOperationsAux
from yateto.codegen.factory import OptimizedKernelFactory
from yateto.memory import MemoryLayout
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


class TestGemmForgeFactors:
    """How a GEMM's scale factor reaches gemmforge.

    gemmforge spells the name of a routine with int(alpha), and takes a factor
    it cannot spell so by the name of an argument. A stand-in records what it
    is handed and names the routine the same way.
    """

    @pytest.fixture
    def forge(self, monkeypatch):
        import sys
        import types
        from yateto.codegen.gemm import gemmgen

        seen = {}

        class Generator:
            def __init__(self, vm):
                pass

            def set(self, transA, transB, a, b, c, alpha, beta):
                seen['alpha'] = alpha

            def get_base_name(self):
                return f'gemm_alpha_{int(seen["alpha"])}'

        class VM:
            def get_headers(self):
                return []

        fake = types.ModuleType('gemmforge')
        fake.YatetoInterface = types.SimpleNamespace(produce_dense_matrix=lambda *args, **kwargs: None)
        fake.vm_factory = lambda *args, **kwargs: VM()
        fake.GemmGenerator = Generator
        fake.GenerationError = RuntimeError
        monkeypatch.setitem(sys.modules, 'gemmforge', fake)
        monkeypatch.setattr(gemmgen, 'gf_spec', True)
        yield seen
        MemoryLayout.DEFAULT_ALIGNMENT_ARCH = None

    def call(self, statement):
        from yateto import Tensor
        from yateto.ast.cost import BoundingBoxCostEstimator
        from yateto.codegen.cache import RoutineCache
        from yateto.codegen.datacache import DataCache
        from yateto.codegen.visitor import OptimizedKernelGenerator
        from yateto.gemm_configuration import GemmForge, GeneratorCollection
        from yateto.generator import Kernel

        arch = useArchitectureIdentifiedBy('dhsw', 'dsm_86', 'cuda')
        A, B, C = (Tensor(name, (8, 8)) for name in 'ABC')
        kernel = Kernel('k', statement(A, B, C), target='gpu')
        kernel.prepareUntilUnitTest(arch)
        kernel.prepareUntilCodeGen(BoundingBoxCostEstimator, False, True)
        outline = OptimizedKernelGenerator(arch, RoutineCache(), DataCache(), {}).generateKernelOutline(
            kernel.nonZeroFlops, kernel.cfg, GeneratorCollection([GemmForge(arch)]), 'gpu')
        return next(line for line in outline.function.splitlines() if 'gemm_alpha' in line).strip()

    def test_a_factor_known_now_is_part_of_the_routine(self, forge):
        call = self.call(lambda A, B, C: C['ij'] <= 2.0 * A['ik'] * B['kj'])
        assert forge['alpha'] == 2.0 and call.startswith('gemm_alpha_2(const_cast')

    def test_a_factor_known_at_run_time_is_an_argument(self, forge):
        from yateto import Scalar
        call = self.call(lambda A, B, C: C['ij'] <= Scalar('s') * A['ik'] * B['kj'])
        assert int(forge['alpha']) == 0 and forge['alpha'] == 'alpha'
        assert call.startswith('gemm_alpha_0(s, ')

    def test_so_is_a_negative_one(self, forge):
        # its value would spell a minus into the routine's name
        call = self.call(lambda A, B, C: C['ij'] <= -2.0 * A['ik'] * B['kj'])
        assert forge['alpha'] == 'alpha'
        assert call.startswith('gemm_alpha_0(-2.0, ')
