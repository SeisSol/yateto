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
