"""The metagen: several generators behind one set of names.

Every generator is generated into a directory and a namespace of its own, and
the headers in the output directory map a template key to the tensors and
kernels of the generator it was added with. What is pinned here is that the
code of the generators stays apart -- each one is compiled in translation
units of its own, and `sources` names them -- and that all of it still links
into one program.
"""
from __future__ import annotations

import pathlib
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
    """A kernel with a constant and a scalar, and a family of kernels."""
    A = Tensor('A', (n, n))
    B = Tensor('B', (n, n))
    C = Tensor('C', (n, n))
    K = Tensor('K', (n, n), np.arange(1.0, n * n + 1.0).reshape(n, n))
    alpha = Scalar('alpha')
    g.add('matmul', C['ij'] <= alpha * A['ik'] * B['kj'] + K['ij'])
    g.addFamily('scaled', simpleParameterSpace(2),
                lambda i: C['ij'] <= (i + 1.0) * A['ij'])


def metagen(tmp_path, variants=VARIANTS, **kwargs):
    """Generates every variant under its template key into `tmp_path`."""
    (tmp_path / 'keys.h').write_text(KEYS_H)
    m = MetaGenerator(['typename'])
    for name, (arch, n) in variants.items():
        g = Generator(useArchitectureIdentifiedBy(arch))
        build(g, n)
        m.add_generator([f'test::{name}'], g, gemm_cfg=GeneratorCollection([]), name=name, **kwargs)
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
        metagen(tmp_path)
        assert sorted(p.name for p in tmp_path.iterdir() if p.is_file()) == \
            ['init.h', 'kernel.h', 'keys.h', 'tensor.h']
        for header in ('init.h', 'kernel.h', 'tensor.h'):
            assert '.cpp' not in (tmp_path / header).read_text()


MAIN = """#include "init.h"
#include "kernel.h"
#include "tensor.h"
#include <cmath>
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

int main() {
  if (int result = check<test::Wide>(4, 10)) return result;
  if (int result = check<test::Narrow>(5, 20)) return result;
  return 0;
}
"""


@pytest.mark.skipif(shutil.which('c++') is None, reason='needs a C++ compiler')
def test_the_generators_link_into_one_program(tmp_path):
    out = tmp_path / 'gen'
    out.mkdir()
    m = metagen(out)
    (tmp_path / 'main.cpp').write_text(MAIN)
    sources = [str(tmp_path / 'main.cpp')] + [path for paths in m.sources(str(out)).values() for path in paths]
    built = subprocess.run(['c++', '-std=c++17', '-fopenmp-simd', '-Wall', '-Wextra', '-Werror', '-Wno-unused-parameter',
                            f'-isystem{INCLUDE}', f'-I{out}', *sources, '-o', str(tmp_path / 'linked')],
                           capture_output=True, text=True)
    assert built.returncode == 0, built.stderr
    ran = subprocess.run([str(tmp_path / 'linked')], capture_output=True, text=True)
    assert ran.returncode == 0, (ran.returncode, ran.stderr)
