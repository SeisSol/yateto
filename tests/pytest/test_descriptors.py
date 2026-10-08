"""Descriptors: where the values of a tensor are, as data.

Every tensor gets a `TensorDescriptor` in init.h, built from the arrays its
view is built from, and every generation a table of all of them in
`init::tensorTable()`. What is pinned here is what the descriptors say, and --
compiled -- that the offset a descriptor gives for an entry is the one the
memory layout gives: the layout is what the kernels were generated against, so
it is the truth a descriptor has to tell.
"""
from __future__ import annotations

import itertools
import pathlib
import re
import shutil
import subprocess

import numpy as np
import pytest

from yateto import Generator, Tensor, useArchitectureIdentifiedBy
from yateto.gemm_configuration import GeneratorCollection
from yateto.memory import CSCMemoryLayout, PatternMemoryLayout
from yateto.type import Datatype


INCLUDE = pathlib.Path(__file__).resolve().parents[2] / 'include'


def generate(tmp_path, build, arch='dhsw', namespace='yateto'):
    """Runs ``build(generator)`` and returns the generation result and the emitted files."""
    arch = useArchitectureIdentifiedBy(arch)
    g = Generator(arch)
    tensors = build(g)
    result = g.generate(str(tmp_path), namespace=namespace, gemm_cfg=GeneratorCollection([]),
                        include_tensors=set(tensors or []))
    return result, {p.name: p.read_text() for p in tmp_path.iterdir() if p.is_file()}


def struct(init_h, name):
    """The body of `struct name : tensor::name`."""
    match = re.search(r'struct {0} : tensor::{0} \{{(.*?)\n    \}};'.format(name), init_h, re.S)
    assert match is not None, name
    return match.group(1)


def descriptor(init_h, name, member=''):
    match = re.search(r'::yateto::TensorDescriptor Descriptor{}\{{(.*?)\}};'.format(member),
                      struct(init_h, name))
    assert match is not None, (name, member)
    return [field.strip() for field in match.group(1).split(',')]


def matmul(g):
    a = Tensor('a', (5, 3))
    b = Tensor('b', (3, 4))
    c = Tensor('c', (5, 4))
    g.add('k', c['ij'] <= a['ik'] * b['kj'])


class TestWhatADescriptorSays:
    def test_a_dense_tensor_points_at_the_arrays_of_its_view(self, tmp_path):
        _, files = generate(tmp_path, matmul)
        assert descriptor(files['init.h'], 'a') == [
            '::yateto::Datatype::F64', '::yateto::Storage::Dense', '2', 'Shape',
            'Start', 'Stop', 'Stride', 'nullptr', 'nullptr', 'nullptr', 'Size', '8']
        assert 'unsigned const Stride[] = {1, 5};' in struct(files['init.h'], 'a')

    def test_the_strides_are_the_layouts(self, tmp_path):
        _, files = generate(tmp_path, matmul)
        strides = re.findall(r'unsigned const Stride\[\] = \{(.*?)\};', files['init.h'])
        # a is 5 x 3 and c is 5 x 4, both stored by columns without padding
        assert '1, 5' in strides

    def test_an_aligned_tensor_asks_for_the_alignment_of_its_architecture(self, tmp_path):
        def build(g):
            return [Tensor('p', (5, 3), alignStride=True), Tensor('q', (5, 3))]

        _, files = generate(tmp_path, build)
        padded = descriptor(files['init.h'], 'p')
        assert padded[-1] == '32'  # hsw, 32 bytes
        # padded to eight rows
        assert 'unsigned const Stop[] = {8, 3};' in struct(files['init.h'], 'p')
        assert descriptor(files['init.h'], 'q')[-1] == '8'

    def test_the_element_type_is_the_tensors_own(self, tmp_path):
        def build(g):
            return [Tensor('h', (3, 3), datatype=Datatype.F32)]

        _, files = generate(tmp_path, build)
        fields = descriptor(files['init.h'], 'h')
        assert fields[0] == '::yateto::Datatype::F32'
        assert fields[-1] == '4'

    def test_csc_points_at_rows_and_columns(self, tmp_path):
        def build(g):
            spp = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1]])
            return [Tensor('s', (4, 3), spp=spp, memoryLayoutClass=CSCMemoryLayout)]

        _, files = generate(tmp_path, build)
        assert descriptor(files['init.h'], 's') == [
            '::yateto::Datatype::F64', '::yateto::Storage::CSC', '2', 'Shape',
            'nullptr', 'nullptr', 'nullptr', 'RowInd', 'ColPtr', 'nullptr', 'Size', '8']

    def test_a_pattern_covers_a_box_of_its_own(self, tmp_path):
        def build(g):
            return [Tensor('t', (3, 2), spp=np.array([[1, 0], [0, 1], [1, 1]]),
                           memoryLayoutClass=PatternMemoryLayout)]

        _, files = generate(tmp_path, build)
        assert descriptor(files['init.h'], 't') == [
            '::yateto::Datatype::F64', '::yateto::Storage::Pattern', '2', 'Shape',
            'Start', 'Stop', 'Stride', 'nullptr', 'nullptr', 'Pattern', 'Size', '8']

    def test_without_dimensions_nothing_is_pointed_at(self, tmp_path):
        def build(g):
            return [Tensor('z', ())]

        _, files = generate(tmp_path, build)
        assert descriptor(files['init.h'], 'z')[3:10] == ['nullptr'] * 7

    def test_a_family_has_one_per_member_and_holes_where_it_has_none(self, tmp_path):
        def build(g):
            return [Tensor('F(0)', (2, 2)), Tensor('F(2)', (3, 3))]

        _, files = generate(tmp_path, build)
        body = struct(files['init.h'], 'F')
        assert descriptor(files['init.h'], 'F', '0')[3] == 'Shape[0]'
        assert descriptor(files['init.h'], 'F', '2')[10] == 'Size[2]'
        assert 'Descriptors[] = {&Descriptor0, nullptr, &Descriptor2};' in body
        assert re.search(r'descriptor\(unsigned i0\) \{\s*return Descriptors\[index\(i0\)\];', body)


class TestTable:
    def test_every_tensor_is_listed_by_name(self, tmp_path):
        def build(g):
            return [Tensor('b', (2,)), Tensor('V', (2, 2), namespace='nodal'), Tensor('a', (2,)),
                    Tensor('G(1,0)', (2,)), Tensor('G(0,1)', (2,))]

        result, files = generate(tmp_path, build, namespace='ns')
        assert result['tensorTable'] == [('G', 2), ('a', 0), ('b', 0), ('nodal::V', 0)]
        entries = re.findall(r'\{"([\w:]+)", (\d), (\w+), ([\w:]+)\}', files['init.cpp'])
        assert [(name, rank) for name, rank, _, _ in entries] == \
            [('G', '2'), ('a', '0'), ('b', '0'), ('nodal::V', '0')]
        assert entries[3][3] == '::ns::nodal::init::V::Descriptors'
        assert re.search(r'GroupSize0\[\] = \{2, 2\};', files['init.cpp'])
        assert '::yateto::TensorTable const& tensorTable();' in files['init.h']

    def test_a_tensor_cannot_take_the_name_of_the_table(self, tmp_path):
        def build(g):
            return [Tensor('tensorTable', (2,))]

        with pytest.raises(ValueError, match='tensorTable'):
            generate(tmp_path, build)


def everything(g):
    """A tensor in every layout there is."""
    return [
        Tensor('dense', (5, 3)),
        Tensor('padded', (5, 3), alignStride=True),
        Tensor('cube', (2, 3, 4)),
        Tensor('narrow', (3, 3), datatype=Datatype.F32),
        Tensor('scalar', ()),
        Tensor('csc', (4, 3), spp=np.array([[1, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1]]),
               memoryLayoutClass=CSCMemoryLayout),
        Tensor('cscPadded', (5, 2), spp=np.array([[1, 0], [0, 0], [0, 1], [0, 0], [1, 0]]),
               memoryLayoutClass=CSCMemoryLayout, alignStride=True),
        Tensor('pattern', (3, 2), spp=np.array([[1, 0], [0, 1], [1, 1]]),
               memoryLayoutClass=PatternMemoryLayout),
        Tensor('patternPadded', (5, 2), spp=np.array([[1, 0], [0, 0], [0, 1], [0, 0], [1, 0]]),
               memoryLayoutClass=PatternMemoryLayout, alignStride=True),
        Tensor('F(0)', (2, 2)),
        Tensor('F(2)', (3, 3), alignStride=True),
        Tensor('V', (2, 2), namespace='nodal'),
    ]


def inside(entry, box):
    return all(rng.start <= e < rng.stop for e, rng in zip(entry, box))


def expectations(tensors):
    """Per tensor, every entry of its box and of its shape with the offset its layout gives."""
    lines = []
    for tensor in tensors:
        layout = tensor.memoryLayout()
        shape = tensor.shape()
        box = [range(max(rng.stop, extent)) for rng, extent in zip(layout.bbox(), shape)]
        for entry in itertools.product(*box):
            if isinstance(layout, (CSCMemoryLayout, PatternMemoryLayout)):
                stored = inside(entry, layout.bbox()) and layout.hasValue(entry)
            else:
                stored = inside(entry, layout.bbox())
            offset = layout.address(entry) if stored else -1
            prefix = tensor.prefix()
            name = tensor.baseName()
            group = ', '.join(str(g) for g in tensor.group())
            index = ', '.join(str(e) for e in entry) if entry else '0'
            lines.append('  check(*{}init::{}::descriptor({}), {{{}}}, {}, __LINE__);'.format(
                prefix, name, group, index, offset))
    return '\n'.join(lines)


PROGRAM = """#include "init.h"
#include <cstdio>

using namespace yateto;

static int failures = 0;

static void check(TensorDescriptor const& descriptor, std::initializer_list<unsigned> index,
                  std::ptrdiff_t expected, int line) {
  const std::ptrdiff_t found = offsetOf(descriptor, index.begin());
  if (found != expected) {
    std::printf("line %d: offset %td, expected %td\\n", line, found, expected);
    ++failures;
  }
}

int main() {
@CHECKS@
  auto const& table = init::tensorTable();
  if (table.find("nodal::V", {}) != &nodal::init::V::Descriptor) return 101;
  if (table.find("F", {2}) != &init::F::Descriptor2) return 102;
  if (table.find("F", {1}) != nullptr) return 103;
  if (table.find("dense") == nullptr || table.find("V") != nullptr) return 104;
  if (!sameLayout(init::dense::Descriptor, *table.find("dense", {}))) return 105;
  if (sameLayout(init::dense::Descriptor, init::padded::Descriptor)) return 106;
  static_assert(init::padded::Descriptor.alignment == 32, "hsw aligns to 32 bytes");
  static_assert(init::narrow::Descriptor.datatype == Datatype::F32, "the tensor's own type");
  static_assert(init::F::descriptor(0) == &init::F::Descriptor0, "");
  return failures == 0 ? 0 : 1;
}
"""


@pytest.mark.skipif(shutil.which('c++') is None, reason='needs a C++ compiler')
def test_a_descriptor_finds_every_entry_where_its_layout_put_it(tmp_path):
    out = tmp_path / 'gen'
    out.mkdir()
    tensors = []

    def build(g):
        tensors.extend(everything(g))
        return tensors

    generate(out, build)
    (tmp_path / 'main.cpp').write_text(PROGRAM.replace('@CHECKS@', expectations(tensors)))
    sources = [str(tmp_path / 'main.cpp')] + [str(out / name) for name in ('init.cpp', 'tensor.cpp')]
    built = subprocess.run(['c++', '-std=c++17', '-Wall', '-Wextra', '-Werror', '-Wno-unused-parameter',
                            f'-isystem{INCLUDE}', f'-I{out}', *sources, '-o', str(tmp_path / 'offsets')],
                           capture_output=True, text=True)
    assert built.returncode == 0, built.stderr
    ran = subprocess.run([str(tmp_path / 'offsets')], capture_output=True, text=True)
    assert ran.returncode == 0, (ran.returncode, ran.stdout)
