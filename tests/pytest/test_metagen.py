"""The metagen: several generators behind one set of names.

Every generator is generated into a directory and a namespace of its own, and
the headers in the output directory map a template key to the tensors and
kernels of the generator it was added with. What is pinned here is that the
code of the generators stays apart -- each one is compiled in translation
units of its own, and `sources` names them -- and that all of it still links
into one program. `runtime.h` reaches every generator by the number of its
variant instead of by its key, and includes the code of none of them.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import shutil
import subprocess

import numpy as np
import pytest

from yateto import Generator, GlobalRoutineCache, Scalar, Tensor, simpleParameterSpace, useArchitectureIdentifiedBy
from yateto.gemm_configuration import GeneratorCollection
from yateto.metagen import MetaGenerator


INCLUDE = pathlib.Path(__file__).resolve().parents[2] / 'include'

#: Template key -> (architecture, size). Two precisions and two sizes, so that
#: the generators share no layout and no element type.
VARIANTS = {'Wide': ('dhsw', 4), 'Narrow': ('shsw', 5)}

KEYS_H = """#ifndef KEYS_H_
#define KEYS_H_
namespace test {
struct Wide {};
struct Narrow {};
} // namespace test
#endif
"""


def build(g, n):
    """A kernel with a constant and a scalar, and a family of kernels.

    The wide variant reads a family of tensors in a namespace, the narrow one
    a tensor the wide one does not have.
    """
    A = Tensor('A', (n, n))
    B = Tensor('B', (n, n))
    C = Tensor('C', (n, n))
    K = Tensor('K', (n, n), np.arange(1.0, n * n + 1.0).reshape(n, n))
    alpha = Scalar('alpha')
    g.add('matmul', C['ij'] <= alpha * A['ik'] * B['kj'] + K['ij'])
    g.addFamily('scaled', simpleParameterSpace(2),
                lambda i: C['ij'] <= (i + 1.0) * A['ij'])
    D = [Tensor(f'D({i})', (n, n), namespace='sub') for i in range(2)]
    g.add('pick', C['ij'] <= D[0]['ij'] + D[1]['ij'])
    if n == 5:
        E = Tensor('E', (n,))
        g.add('only', C['ij'] <= E['i'] * A['ij'])


def metagen(tmp_path, variants=VARIANTS, generate=True, **kwargs):
    """Adds every variant under its template key, and generates them into `tmp_path`."""
    (tmp_path / 'keys.h').write_text(KEYS_H)
    m = MetaGenerator(['typename'])
    for name, (arch, n) in variants.items():
        g = Generator(useArchitectureIdentifiedBy(arch))
        build(g, n)
        m.add_generator([f'test::{name}'], g, gemm_cfg=GeneratorCollection([]), name=name, **kwargs)
    if generate:
        m.generate(str(tmp_path), namespace='test', includes=['keys.h'])
    return m


class TestNames:
    def test_the_name_names_directory_and_namespace(self, tmp_path):
        metagen(tmp_path)
        for name in VARIANTS:
            kernel_h = (tmp_path / f'metagen_{name}' / 'kernel.h').read_text()
            assert f'namespace yatetometagen_{name} {{' in kernel_h

    def test_names_default_to_the_position(self):
        m = MetaGenerator(['typename'])
        g = Generator(useArchitectureIdentifiedBy('dhsw'))
        m.add_generator(['A'], g)
        m.add_generator(['B'], g)
        assert [gendata['name'] for gendata in m.generators] == ['0', '1']

    def test_the_name_is_not_passed_to_the_generator(self):
        m = MetaGenerator(['typename'])
        m.add_generator(['A'], Generator(useArchitectureIdentifiedBy('dhsw')), name='a', namespace_hint=1)
        assert m.generators[0]['kwargs'] == {'namespace_hint': 1}

    @pytest.mark.parametrize('name', ['a-b', 'a::b', ''])
    def test_a_name_has_to_fit_into_an_identifier(self, name):
        m = MetaGenerator(['typename'])
        with pytest.raises(ValueError, match='identifier'):
            m.add_generator(['A'], Generator(useArchitectureIdentifiedBy('dhsw')), name=name)

    def test_a_name_is_taken_once(self):
        m = MetaGenerator(['typename'])
        g = Generator(useArchitectureIdentifiedBy('dhsw'))
        m.add_generator(['A'], g, name='x')
        with pytest.raises(ValueError, match='already'):
            m.add_generator(['B'], g, name='x')

    def test_a_key_has_as_many_arguments_as_the_template(self):
        m = MetaGenerator(['typename', 'int'])
        with pytest.raises(ValueError, match='2 arguments'):
            m.add_generator(['A'], Generator(useArchitectureIdentifiedBy('dhsw')))


class TestSources:
    def test_every_generator_is_compiled_on_its_own(self, tmp_path):
        m = metagen(tmp_path)
        sources = m.sources(str(tmp_path))
        assert list(sources) == list(VARIANTS)
        for name, paths in sources.items():
            assert [pathlib.Path(p).name for p in paths] == \
                ['tensor.cpp', 'init.cpp', 'kernel.cpp', 'pool.cpp', 'subroutine.cpp']
            for path in paths:
                assert pathlib.Path(path).parent == tmp_path / f'metagen_{name}'
                assert pathlib.Path(path).is_file()

    def test_device_sources_and_tests_are_listed_apart(self, tmp_path):
        m = metagen(tmp_path)
        for name in VARIANTS:
            assert [pathlib.Path(p).name for p in m.device_sources(str(tmp_path))[name]] == \
                ['gpulike_subroutine.cpp']
            assert [pathlib.Path(p).name for p in m.tests(str(tmp_path))[name]] == ['test-kernel.cpp']
            for path in m.device_sources(str(tmp_path))[name] + m.tests(str(tmp_path))[name]:
                assert pathlib.Path(path).is_file()

    def test_a_shared_routine_cache_keeps_the_routines(self, tmp_path):
        cache = GlobalRoutineCache()
        m = metagen(tmp_path, routine_cache=cache)
        for name in VARIANTS:
            assert [pathlib.Path(p).name for p in m.sources(str(tmp_path))[name]] == \
                ['tensor.cpp', 'init.cpp', 'kernel.cpp', 'pool.cpp']
            assert m.device_sources(str(tmp_path))[name] == []

    def test_nothing_includes_the_code_of_a_generator(self, tmp_path):
        m = metagen(tmp_path)
        assert sorted(p.name for p in tmp_path.iterdir() if p.is_file()) == \
            ['init.h', 'kernel.h', 'keys.h', 'runtime.cpp', 'runtime.h', 'tensor.h']
        for header in ('init.h', 'kernel.h', 'tensor.h'):
            assert '.cpp' not in (tmp_path / header).read_text()
        assert m.shared_sources(str(tmp_path)) == [str(tmp_path / 'runtime.cpp')]
        includes = re.findall(r'#include ["<](.*)[">]', (tmp_path / 'runtime.cpp').read_text() +
                              (tmp_path / 'runtime.h').read_text())
        assert sorted(includes) == ['cstddef', 'initializer_list', 'runtime.h', 'stdexcept', 'string',
                                    'string_view', 'yateto.h']

    def test_what_a_generator_reports_is_plain_data(self, tmp_path):
        # A build that generates each generator in a process of its own hands
        # the reports over to the one that writes the headers.
        at_once = tmp_path / 'at_once'
        apart = tmp_path / 'apart'
        at_once.mkdir()
        apart.mkdir()
        metagen(at_once)
        m = metagen(apart, generate=False)
        reports = [json.loads(json.dumps(m.generate_single(i, str(apart), 'test')))
                   for i in range(len(m.generators))]
        m.generate(str(apart), namespace='test', includes=['keys.h'], precompiled=reports)
        for name in ('init.h', 'kernel.h', 'tensor.h', 'runtime.h', 'runtime.cpp'):
            assert (at_once / name).read_text() == (apart / name).read_text(), name


class TestRuntimeHeader:
    @pytest.fixture
    def out(self, tmp_path):
        metagen(tmp_path)
        return tmp_path

    def test_it_says_what_the_variants_are(self, out):
        runtime_h = (out / 'runtime.h').read_text()
        assert 'constexpr std::size_t VariantCount = 2;' in runtime_h
        assert 'VariantNames[] = {"Wide", "Narrow"};' in runtime_h
        assert 'VariantKeys[] = {"test::Wide", "test::Narrow"};' in runtime_h
        tensor_h = (out / 'tensor.h').read_text()
        assert 'template<> struct VariantOf<test::Wide> { static constexpr std::size_t value = 0; };' in tensor_h
        assert 'template<> struct VariantOf<test::Narrow> { static constexpr std::size_t value = 1; };' in tensor_h

    def test_a_tensor_is_found_where_each_table_lists_it(self, out):
        runtime_cpp = (out / 'runtime.cpp').read_text()
        positions = dict(re.findall(r'(\w+)::descriptor\(std::size_t variant[^)]*\) \{\s*'
                                    r'static constexpr std::ptrdiff_t Positions\[\] = \{(.*?)\};',
                                    runtime_cpp))
        # Wide: A B C K; Narrow: A B C E K -- the family is in a namespace of its own
        assert positions['A'] == '0, 0'
        assert positions['K'] == '3, 4'
        assert positions['E'] == '-1, 3'
        assert positions['D'] == '4, 5'
        assert re.search(r'namespace sub \{\s*namespace init \{\s*struct D \{', (out / 'runtime.h').read_text())
        assert 'descriptor(std::size_t variant, unsigned i0);' in (out / 'runtime.h').read_text()

    @staticmethod
    def named(tmp_path, *tensors):
        """Two generators reading tensors of the given name and namespace."""
        m = MetaGenerator(['typename'])
        for key in ('A', 'B'):
            g = Generator(useArchitectureIdentifiedBy('dhsw'))
            read = [Tensor(name, (2,), namespace=space) for name, space in tensors]
            out = Tensor('out', (2,))
            g.add('k', out['i'] <= sum((t['i'] for t in read[1:]), read[0]['i']))
            m.add_generator([key], g, gemm_cfg=GeneratorCollection([]))
        m.generate(str(tmp_path), namespace='test')
        return m

    def test_a_namespace_cannot_take_a_name_of_the_runtime(self, tmp_path):
        with pytest.raises(ValueError, match='runtime::tensorTable'):
            self.named(tmp_path, ('X', 'tensorTable'))

    @pytest.mark.skipif(shutil.which('c++') is None, reason='needs a C++ compiler')
    def test_a_tensor_may_be_named_like_what_runtime_uses_itself(self, tmp_path):
        m = self.named(tmp_path, ('member', None), ('Tables', 'checkVariant'), ('checkVariant', None))
        built = subprocess.run(['c++', '-std=c++17', '-Wall', '-Wextra', '-Werror', f'-isystem{INCLUDE}',
                                f'-I{tmp_path}', '-c', *m.shared_sources(str(tmp_path)), '-o', os.devnull],
                               capture_output=True, text=True)
        assert built.returncode == 0, built.stderr

    def test_a_family_keeps_its_number_of_indices(self, tmp_path):
        def build(g, rank):
            T = Tensor('T({})'.format(','.join(['0'] * rank)), (2,))
            g.add('k', T['i'] <= T['i'])

        m = MetaGenerator(['typename'])
        for rank in (1, 2):
            g = Generator(useArchitectureIdentifiedBy('dhsw'))
            build(g, rank)
            m.add_generator([f'K{rank}'], g, gemm_cfg=GeneratorCollection([]))
        with pytest.raises(ValueError, match='1 indices in one generator and of 2'):
            m.generate(str(tmp_path), namespace='test')


MAIN = """#include "init.h"
#include "kernel.h"
#include "runtime.h"
#include "tensor.h"
#include <cmath>
#include <stdexcept>
#include <type_traits>
#include <vector>

// The pool is a type of each generator's own, not one the metagen maps a key to.
inline auto poolOf(test::Wide) { return test::yatetometagen_Wide::Pool::host(); }
inline auto poolOf(test::Narrow) { return test::yatetometagen_Narrow::Pool::host(); }

template <typename Key>
int check(unsigned n, int code) {
  using real = std::remove_const_t<std::remove_pointer_t<decltype(test::kernel::matmul<Key>::C)>>;
  std::vector<real> a(test::tensor::A<Key>::size()), b(test::tensor::B<Key>::size()),
      c(test::tensor::C<Key>::size());
  for (unsigned i = 0; i < a.size(); ++i) {
    a[i] = static_cast<real>(i % 7) + 1;
    b[i] = static_cast<real>(i % 5) - 2;
  }
  auto pool = poolOf(Key{});
  test::kernel::matmul<Key> krnl;
  krnl.bindGlobals(pool);
  krnl.alpha = 2;
  krnl.A = a.data();
  krnl.B = b.data();
  krnl.C = c.data();
  krnl.execute();
  auto va = test::init::A<Key>::view::create(a.data());
  auto vb = test::init::B<Key>::view::create(b.data());
  auto vc = test::init::C<Key>::view::create(c.data());
  auto vk = test::init::K<Key>::view::create(test::init::K<Key>::Values);
  for (unsigned i = 0; i < n; ++i) {
    for (unsigned j = 0; j < n; ++j) {
      double expected = vk(i, j);
      for (unsigned k = 0; k < n; ++k) {
        expected += 2.0 * va(i, k) * vb(k, j);
      }
      if (std::abs(vc(i, j) - expected) > 1e-4 * std::abs(expected) + 1e-4) {
        return code;
      }
    }
  }
  test::kernel::scaled<Key> family;
  family.A = a.data();
  family.C = c.data();
  family.execute(1);
  for (unsigned i = 0; i < n; ++i) {
    for (unsigned j = 0; j < n; ++j) {
      if (vc(i, j) != 2 * va(i, j)) {
        return code + 1;
      }
    }
  }
  return 0;
}

int checkRuntime() {
  using namespace test;
  static_assert(runtime::VariantCount == 2, "");
  static_assert(runtime::variantOf<Wide>() == 0 && runtime::variantOf<Narrow>() == 1, "");
  if (runtime::VariantNames[1] != "Narrow" || runtime::VariantKeys[0] != "test::Wide") return 30;
  if (runtime::init::A::descriptor(0) != &yatetometagen_Wide::init::A::Descriptor) return 31;
  if (runtime::init::A::descriptor(1) != &yatetometagen_Narrow::init::A::Descriptor) return 32;
  if (runtime::sub::init::D::descriptor(1, 1) != &yatetometagen_Narrow::sub::init::D::Descriptor1) return 33;
  if (runtime::sub::init::D::descriptor(1, 2) != nullptr) return 34;
  if (runtime::init::E::descriptor(0) != nullptr) return 35;
  if (runtime::init::E::descriptor(1) != &yatetometagen_Narrow::init::E::Descriptor) return 36;
  if (runtime::tensorTable(0).find("E") != nullptr || runtime::tensorTable(1).find("E") == nullptr) return 37;
  if (runtime::init::A::descriptor(0)->datatype != yateto::Datatype::F64 ||
      runtime::init::A::descriptor(1)->datatype != yateto::Datatype::F32) return 38;
  try {
    runtime::init::A::descriptor(2);
    return 39;
  } catch (std::out_of_range const&) {
  }
  return 0;
}

int main() {
  if (int result = check<test::Wide>(4, 10)) return result;
  if (int result = check<test::Narrow>(5, 20)) return result;
  return checkRuntime();
}
"""


@pytest.mark.skipif(shutil.which('c++') is None, reason='needs a C++ compiler')
def test_the_generators_link_into_one_program(tmp_path):
    out = tmp_path / 'gen'
    out.mkdir()
    m = metagen(out)
    (tmp_path / 'main.cpp').write_text(MAIN)
    sources = [str(tmp_path / 'main.cpp')] + m.shared_sources(str(out)) + \
        [path for paths in m.sources(str(out)).values() for path in paths]
    built = subprocess.run(['c++', '-std=c++17', '-fopenmp-simd', '-Wall', '-Wextra', '-Werror', '-Wno-unused-parameter',
                            f'-isystem{INCLUDE}', f'-I{out}', *sources, '-o', str(tmp_path / 'linked')],
                           capture_output=True, text=True)
    assert built.returncode == 0, built.stderr
    ran = subprocess.run([str(tmp_path / 'linked')], capture_output=True, text=True)
    assert ran.returncode == 0, (ran.returncode, ran.stderr)
