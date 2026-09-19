"""What the emitted C++ headers have to guarantee.

These tests read the generated text rather than compiling it: the C++
side is covered by the CxxTest suite that ``Generator.generate`` writes
out and by the ``yateto-cpu.yml`` workflow. What we pin here is the
shape of the emission, mainly the checks that turn an out-of-range
family index from a silent out-of-bounds read into a failing assert.
"""
from __future__ import annotations

import re

import pytest

from yateto import Generator, Tensor, useArchitectureIdentifiedBy
from yateto.gemm_configuration import GeneratorCollection


ARCH = 'dhsw'


@pytest.fixture
def arch():
    return useArchitectureIdentifiedBy(ARCH)


def generate(tmp_path, build, arch, namespace='yateto'):
    """Runs ``build(generator)`` and returns the emitted files by name."""
    g = Generator(arch)
    build(g)
    g.generate(str(tmp_path), namespace=namespace, gemm_cfg=GeneratorCollection([]))
    return {p.name: p.read_text() for p in tmp_path.iterdir() if p.is_file()}


def matmul(g):
    """One plain kernel, no families anywhere."""
    a = Tensor('a', (5, 5))
    b = Tensor('b', (5, 5))
    c = Tensor('c', (5, 5))
    g.add('k', c['ij'] <= a['ik'] * b['kj'])


class TestTensorHeader:
    def test_family_index_is_range_checked(self, tmp_path, arch):
        def build(g):
            dq = [Tensor('dQ({})'.format(i), (5, 5)) for i in range(3)]
            a = Tensor('a', (5, 5))
            g.add('k', dq[0]['ij'] <= dq[1]['ik'] * a['kj'] + dq[2]['ij'])

        files = generate(tmp_path, build, arch)
        tensor_h = files['tensor.h']
        assert '#include <cassert>' in tensor_h
        # one check in index() covers size() and the container accessors
        index_body = re.search(r'index\(unsigned i0\) \{(.*?)\}', tensor_h, re.S)
        assert index_body is not None
        assert 'assert(1*i0 < 3);' in index_body.group(1)

    def test_single_tensor_has_no_index_function(self, tmp_path, arch):
        tensor_h = generate(tmp_path, matmul, arch)['tensor.h']
        assert 'index(' not in tensor_h


class TestKernelHeader:
    @pytest.fixture
    def files(self, tmp_path, arch):
        def build(g):
            a = Tensor('a', (5, 5))
            b = Tensor('b', (5, 5))
            c = Tensor('c', (5, 5))
            # index 0 is never added, so the dispatch table starts with a hole
            for i in range(1, 3):
                g.add('fam({})'.format(i), c['ij'] <= a['ik'] * b['kj'])

        return generate(tmp_path, build, arch)

    @staticmethod
    def body(text, signature):
        match = re.search(re.escape(signature) + r' \{(.*?)\n(\s*)\}', text, re.S)
        assert match is not None, signature
        return match.group(1)

    def test_dispatch_table_has_a_hole(self, files):
        # the premise of the next two tests
        assert 'ExecutePtrs[] = {nullptr,' in files['kernel.h']

    def test_execute_rejects_a_hole(self, files):
        assert 'assert(findExecute(i0) != nullptr);' in files['kernel.h']

    def test_family_accessors_are_range_checked(self, files):
        kernel_h = files['kernel.h']
        assert '#include <cassert>' in kernel_h
        for accessor in ('findExecute', 'nonZeroFlops', 'hardwareFlops',
                         'inboundConstBytes', 'inboundBytes', 'outboundBytes',
                         'tmpMemRequiredInBytes'):
            assert 'assert(1*i0 < 3);' in self.body(
                kernel_h, '{}(unsigned i0)'.format(accessor)), accessor

    def test_single_kernel_has_no_bounds_check(self, tmp_path, arch):
        kernel_h = generate(tmp_path, matmul, arch)['kernel.h']
        assert 'assert(' not in kernel_h


class TestHeaderGuards:
    def test_guards_follow_the_namespace(self, tmp_path, arch):
        files = generate(tmp_path, matmul, arch, namespace='someproject::variant')
        for name, guard in (('tensor.h', 'SOMEPROJECT_VARIANT_TENSOR_H_'),
                            ('init.h', 'SOMEPROJECT_VARIANT_INIT_H_'),
                            ('kernel.h', 'SOMEPROJECT_VARIANT_KERNEL_H_')):
            assert '#ifndef {}'.format(guard) in files[name], name
