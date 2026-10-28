"""A constant sparse A, whose values PSpaMM builds into its routine.

Where PSpaMM says it can (--capabilities), yateto hands it the values of a
sparse A whose values are known along with the pattern, and asks it to build
them in (--abake). The routine reads nothing of A then: it needs neither the
pattern in whole vector tiles nor A stored as the tensor lays it out. Two
routines of one pattern differ by their values, so the values are part of the
name.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess

import numpy as np
import pytest

from yateto import Generator, Tensor, useArchitectureIdentifiedBy
from yateto.codegen.gemm import gemmgen
from yateto.codegen.gemm.descriptor import GemmDescriptor
from yateto.gemm_configuration import DENSE, GeneratorCollection, PSpaMM, Sparsity
from yateto.memory import CSCMemoryLayout
from yateto.type import Datatype


def gemmParameters():
    return dict(M=8, N=4, K=8, LDA=0, LDB=8, LDC=8, alpha='1', beta='0',
                alignedA=0, alignedC=1, prefetch='nopf', transA=False, transB=False,
                datatypeA=Datatype.F64, datatypeB=Datatype.F64, datatypeC=Datatype.F64)


def descriptor(valuesA=None, sppA=((0, 0), (3, 1))):
    return GemmDescriptor.create('pspamm', 'hsw', gemmParameters(), sppA, None, valuesA)


class TestTheName:
    def test_two_routines_of_one_pattern_differ_by_their_values(self):
        assert descriptor([1.0, 2.0]).routineName() != descriptor([1.0, 3.0]).routineName()

    def test_a_routine_that_reads_A_is_named_as_before(self):
        assert descriptor().routineName() == (
            'pspamm_asparse_cab8b866bab1cba6f5e8e58671ad23d4'
            '_f64_f64_f64_hsw_m8_n4_k8_ldA0_ldB8_ldC8'
            '_alpha1_beta0_alignedA0_alignedC1_transAFalse_transBFalse_nopf')

    def test_the_values_follow_the_pattern(self):
        assert descriptor([1.0, 2.0]).routineName().startswith(
            'pspamm_asparse_cab8b866bab1cba6f5e8e58671ad23d4_v')

    def test_how_the_values_arrive_does_not_reach_it(self):
        """A tensor holds its values as strings; the array library hands out its own floats."""
        names = {descriptor(values).routineName()
                 for values in ([0.5, -2.0], ['0.5', '-2.0'], [np.float64(0.5), np.float64(-2.0)])}

        assert len(names) == 1

    def test_a_round_trip_changes_nothing(self):
        original = descriptor([0.1, -1e-300])

        restored = GemmDescriptor.fromJson(json.loads(json.dumps(original.asJson())))

        assert restored == original
        assert restored.routineName() == original.routineName()

    def test_a_descriptor_written_before_the_values_reads_back(self):
        written = descriptor().asJson()
        del written['valuesA']

        assert GemmDescriptor.fromJson(written) == descriptor()


def csc(values):
    return Tensor('A', values.shape, values, memoryLayoutClass=CSCMemoryLayout)


def elementwise():
    """A 6 by 8 with two entries a row: no vector tile of it is whole."""
    values = np.zeros((6, 8))
    for row in range(6):
        values[row, row] = 1.0 + row
        values[row, (row + 3) % 8] = -0.5
    return values


def pspammTakes(sparseA):
    pspamm = PSpaMM(useArchitectureIdentifiedBy('dskx'))
    return pspamm.supported(6, 4, 8, sparseA, DENSE, False, False, 1.0, 0.0, False, True,
                            Datatype.F64, Datatype.F64, Datatype.F64, 'cpu')


@pytest.fixture
def capabilities(monkeypatch):
    """What PSpaMM says it supports, as the test sets it."""
    said = {}
    monkeypatch.setattr(PSpaMM, 'capabilities', lambda self: said)
    return said


class TestWhatPSpaMMTakes:
    def test_an_element_wise_pattern_of_known_values_where_it_builds_them_in(self, capabilities):
        capabilities['abake'] = True
        layout = csc(elementwise()).memoryLayout()

        assert pspammTakes(Sparsity.of(layout, constant=True))

    def test_not_where_it_does_not_say_so(self, capabilities):
        layout = csc(elementwise()).memoryLayout()

        assert not pspammTakes(Sparsity.of(layout, constant=True))

    def test_not_without_the_values(self, capabilities):
        capabilities['abake'] = True
        layout = csc(elementwise()).memoryLayout()

        assert not pspammTakes(Sparsity.of(layout))

    def test_a_dense_operand_is_dense_whatever_its_values(self):
        layout = Tensor('D', (6, 8), np.ones((6, 8))).memoryLayout()

        assert Sparsity.of(layout, constant=True) == DENSE


def bakingPspamm():
    """The program of PSpaMM for this interpreter, where it builds values in."""
    pytest.importorskip('pypspamm.cli')
    gemmgen._pspammInProcess.cache_clear()
    gemmgen.pspammCapabilities.cache_clear()
    if shutil.which('pspamm-generator') is None or gemmgen._pspammInProcess('pspamm-generator') is None:
        pytest.skip('needs the program of PSpaMM, installed for this interpreter')
    if not gemmgen.pspammCapabilities('pspamm-generator').get('abake'):
        pytest.skip('needs a PSpaMM that builds the values of A in')


def generate(directory, tensors):
    """C = A B for each A given, with PSpaMM as the only tool, for AVX2."""
    arch = useArchitectureIdentifiedBy('dhsw')
    g = Generator(arch)
    B = Tensor('B', (8, 4))
    C = Tensor('C', (6, 4), alignStride=True)
    for index, A in enumerate(tensors):
        g.add(f'k{index}', C['ij'] <= A['ik'] * B['kj'])
    directory.mkdir()
    g.generate(str(directory), gemm_cfg=GeneratorCollection([PSpaMM(arch)]))
    return (directory / 'subroutine.cpp').read_text(), C.memoryLayout().stridei(1)


def runsAVX2():
    try:
        with open('/proc/cpuinfo') as cpuinfo:
            flags = cpuinfo.read().split()
    except OSError:
        return False
    return 'avx2' in flags and 'fma' in flags


class TestTheRoutine:
    def test_carries_the_values(self, tmp_path):
        bakingPspamm()
        code, _ = generate(tmp_path / 'out', [csc(elementwise())])

        assert 'pspamm_A_table' in code

    def test_is_one_per_set_of_values(self, tmp_path):
        bakingPspamm()
        other = elementwise()
        other[0, 0] = 7.0
        code, _ = generate(tmp_path / 'out', [csc(elementwise()), Tensor('E', (6, 8), other, memoryLayoutClass=CSCMemoryLayout)])

        assert code.count('pspamm_A_table[') == 2

    def test_without_values_there_is_none_for_such_a_pattern(self, tmp_path):
        bakingPspamm()
        pattern = Tensor('A', (6, 8), elementwise() != 0, memoryLayoutClass=CSCMemoryLayout)
        code, _ = generate(tmp_path / 'out', [pattern])

        assert 'pspamm_' not in code

    @pytest.mark.skipif(shutil.which('c++') is None or not runsAVX2(), reason='needs a C++ compiler and AVX2')
    def test_reads_nothing_of_A(self, tmp_path):
        """A passed as NaN: what comes out is what the values say."""
        bakingPspamm()
        values = elementwise()
        out = tmp_path / 'out'
        _, ldc = generate(out, [csc(values)])
        (tmp_path / 'main.cpp').write_text(
            '#include <cmath>\n#include <cstdio>\n#include "kernel.h"\n'
            'long long libxsmm_num_total_flops = 0;\nlong long pspamm_num_total_flops = 0;\n'
            'int main() {\n'
            '  alignas(64) static double A[64], B[32], C[64];\n'
            '  for (int i = 0; i < 64; ++i) { A[i] = NAN; C[i] = 0; }\n'
            '  for (int i = 0; i < 32; ++i) B[i] = 0.25 * (i % 5) - 0.5;\n'
            '  yateto::kernel::k0 krnl;\n'
            '  krnl.A = A; krnl.B = B; krnl.C = C;\n'
            '  krnl.execute();\n'
            '  for (int i = 0; i < 64; ++i) printf("%.17g\\n", C[i]);\n'
            '  return 0;\n'
            '}\n')
        include = os.path.join(os.path.dirname(__file__), '..', '..', 'include')
        sources = [str(tmp_path / 'main.cpp')] + [str(out / name) for name in
                                                  ('kernel.cpp', 'subroutine.cpp', 'init.cpp', 'pool.cpp', 'tensor.cpp')
                                                  if (out / name).exists()]
        built = subprocess.run(['c++', '-std=c++17', '-mavx2', '-mfma', f'-I{include}', f'-I{out}', *sources,
                                '-o', str(tmp_path / 'baked')], capture_output=True, text=True)
        assert built.returncode == 0, built.stderr
        ran = subprocess.run([str(tmp_path / 'baked')], capture_output=True, text=True, check=True)
        C = np.array([float(x) for x in ran.stdout.split()])[:4 * ldc].reshape(4, ldc).T[:6]
        B = (0.25 * (np.arange(32) % 5) - 0.5).reshape(4, 8).T

        assert np.allclose(C, values @ B, rtol=0, atol=1e-14)
