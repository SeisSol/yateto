"""What the emitted C++ headers have to guarantee.

These tests read the generated text rather than compiling it: the C++
side is covered by the CxxTest suite that ``Generator.generate`` writes
out and by the ``yateto-cpu.yml`` workflow. What we pin here is the
shape of the emission, mainly the checks that turn an out-of-range
family index from a silent out-of-bounds read into a failing assert.
"""
from __future__ import annotations

import pathlib
import re
import shutil
import subprocess

import numpy as np
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


class TestEmission:
    def test_no_redundant_out_of_line_definitions(self, tmp_path, arch):
        """A constexpr static member is implicitly inline since C++17, and a
        definition outside the class is deprecated -- both GCC and clang say
        so, once per member."""
        def build(g):
            dq = [Tensor('dQ({})'.format(i), (5, 5)) for i in range(3)]
            a = Tensor('a', (5, 5))
            g.add('k', dq[0]['ij'] <= dq[1]['ik'] * a['kj'] + dq[2]['ij'])
            for i in range(1, 3):
                g.add('fam({})'.format(i), dq[0]['ij'] <= dq[1]['ik'] * a['kj'])

        files = generate(tmp_path, build, arch)
        for name in ('Shape', 'Size', 'ExecutePtrs'):
            assert name not in files['tensor.cpp'], name
            assert name not in files['kernel.cpp'], name

    def test_no_empty_namespace_blocks(self, tmp_path, arch):
        """A group with no members used to open and close a namespace anyway.
        The outermost namespace of a source file is not in scope here: it is
        the file's own, and it stays even when there is nothing to put in it."""
        files = generate(tmp_path, matmul, arch)
        for name, text in files.items():
            if not name.endswith('.h'):
                continue
            assert re.search(r'namespace \w+ \{\s*\n\s*\} // namespace', text) is None, name


class TestIncludes:
    def test_headers_bring_their_own_integer_types(self, tmp_path, arch):
        """int8_t and friends turn up in the emission whenever a tensor or a
        GPU scratch buffer asks for them, so the headers cannot rely on the
        support library having pulled <cstdint> in first."""
        files = generate(tmp_path, matmul, arch)
        for name in ('kernel.h', 'init.h'):
            assert '#include <cstdint>' in files[name], name


class TestDeviceMarkers:
    """CUDA and HIP compile a translation unit once per side and reject a
    call into a function the other side owns, so anything a kernel may reach
    from device code has to carry the marker."""

    def test_headers_include_the_marker(self, tmp_path, arch):
        files = generate(tmp_path, matmul, arch)
        for name in ('tensor.h', 'kernel.h'):
            assert '#include "yateto/Marker.h"' in files[name], name

    def test_tensor_accessors_are_marked(self, tmp_path, arch):
        def build(g):
            dq = [Tensor('dQ({})'.format(i), (5, 5)) for i in range(3)]
            a = Tensor('a', (5, 5))
            g.add('k', dq[0]['ij'] <= dq[1]['ik'] * a['kj'] + dq[2]['ij'])

        tensor_h = generate(tmp_path, build, arch)['tensor.h']
        for accessor in ('unsigned index(unsigned i0)', 'unsigned size(unsigned i0)',
                         'T& operator()(unsigned i0)'):
            line = next(l for l in tensor_h.splitlines() if accessor in l)
            assert 'YATETO_HOSTDEVICE' in line, accessor

    def test_view_factories_are_marked(self, tmp_path, arch):
        init_h = generate(tmp_path, matmul, arch)['init.h']
        for line in init_h.splitlines():
            if ' create(' in line:
                assert 'YATETO_HOSTDEVICE' in line, line

    def test_host_side_dispatch_stays_on_the_host(self, tmp_path, arch):
        """execute() goes through a member function pointer into code an
        external generator emitted for the host; marking it would be a
        promise the kernel bodies cannot keep."""
        def build(g):
            a = Tensor('a', (5, 5))
            b = Tensor('b', (5, 5))
            c = Tensor('c', (5, 5))
            for i in range(1, 3):
                g.add('fam({})'.format(i), c['ij'] <= a['ik'] * b['kj'])

        kernel_h = generate(tmp_path, build, arch)['kernel.h']
        line = next(l for l in kernel_h.splitlines() if 'void execute(unsigned i0)' in l)
        assert 'YATETO_HOSTDEVICE' not in line


class TestDeviceMarkersOnData:
    """The marker is an execution space, and only code has one.

    nvcc refuses it on a variable -- "memory qualifier on data member is not
    allowed" -- so a header that marks its constexpr data compiles with
    clang and g++ and stops every CUDA translation unit that includes it.
    A constexpr datum needs no marker to be read on either side.
    """

    @staticmethod
    def build(g):
        from yateto.memory import CSCMemoryLayout
        dq = [Tensor('dQ({})'.format(i), (5, 5)) for i in range(3)]
        a = Tensor('a', (5, 5), spp=np.eye(5))
        s = Tensor('s', (5, 5), spp=np.eye(5), memoryLayoutClass=CSCMemoryLayout)
        for i in range(1, 3):
            g.add('fam({})'.format(i), dq[0]['ij'] <= dq[i]['ik'] * a['kj'])
        g.add('k', dq[0]['ij'] <= dq[1]['ij'] + dq[2]['ij'])
        g.add('sparse', dq[0]['ij'] <= dq[1]['ik'] * s['kj'])

    @staticmethod
    def factories(init_h, tensor):
        """The `create` lines of one tensor's view struct."""
        block = init_h.split(f'struct {tensor} : ')[1].split('    };\n')[0]
        return [line.strip() for line in block.splitlines() if ' create(' in line]

    def test_a_sparse_view_s_factory_stays_on_the_host(self, tmp_path, arch):
        """Its index arrays are host data, which device code may not name."""
        init_h = generate(tmp_path, self.build, arch)['init.h']
        sparse = self.factories(init_h, 's')
        dense = self.factories(init_h, 'a')
        assert len(sparse) == 2 and len(dense) == 2
        assert all(line.startswith('static inline') for line in sparse)
        assert all(line.startswith('YATETO_HOSTDEVICE static inline') for line in dense)

    def test_no_datum_is_marked(self, tmp_path, arch):
        files = generate(tmp_path, self.build, arch)
        for name in ('tensor.h', 'init.h', 'kernel.h'):
            for line in files[name].splitlines():
                declaration = line.strip()
                if 'YATETO_HOSTDEVICE' in declaration and declaration.endswith(';') \
                   and ' = ' in declaration:
                    pytest.fail(f'{name} marks a datum: {declaration}')

    @pytest.mark.skipif(shutil.which('nvcc') is None, reason='needs nvcc')
    def test_the_headers_compile_as_cuda(self, tmp_path, arch):
        gen = tmp_path / 'gen'
        gen.mkdir()
        generate(gen, self.build, arch)
        source = tmp_path / 'device.cu'
        source.write_text(
            '#include "tensor.h"\n#include "init.h"\n#include "kernel.h"\n'
            '__global__ void touch(unsigned* out) {\n'
            '  out[0] = yateto::tensor::dQ::size(1) + yateto::tensor::a::size();\n'
            '}\n')
        include = pathlib.Path(__file__).resolve().parents[2] / 'include'
        subprocess.run(['nvcc', '-std=c++17', '-c', str(source), '-o', str(tmp_path / 'device.o'),
                        f'-I{include}', f'-I{gen}'], check=True, capture_output=True, text=True)


class TestHeaderGuards:
    def test_guards_follow_the_namespace(self, tmp_path, arch):
        files = generate(tmp_path, matmul, arch, namespace='someproject::variant')
        for name, guard in (('tensor.h', 'SOMEPROJECT_VARIANT_TENSOR_H_'),
                            ('init.h', 'SOMEPROJECT_VARIANT_INIT_H_'),
                            ('kernel.h', 'SOMEPROJECT_VARIANT_KERNEL_H_')):
            assert '#ifndef {}'.format(guard) in files[name], name
