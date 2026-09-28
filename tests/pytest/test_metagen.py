"""The metagen: several generators behind one set of names.

Every generator is generated into a directory and a namespace of its own, and
the headers in the output directory map a template key to the tensors and
kernels of the generator it was added with. What is pinned here is that the
code of the generators stays apart -- each one is compiled in translation
units of its own, and `sources` names them -- and that all of it still links
into one program. `runtime.h` reaches every generator by the number of its
variant instead of by its key, and includes the code of none of them. A
kernel that asks for it takes views there, which each generator binds to its
own kernel in a translation unit of its own.
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
    a tensor the wide one does not have, and pads what it writes. All kernels
    but `copied` take views as well.
    """
    views = {'operands': 'runtime'}
    A = Tensor('A', (n, n))
    B = Tensor('B', (n, n))
    C = Tensor('C', (n, n), alignStride=(n == 5))
    K = Tensor('K', (n, n), np.arange(1.0, n * n + 1.0).reshape(n, n))
    alpha = Scalar('alpha')
    g.add('matmul', C['ij'] <= alpha * A['ik'] * B['kj'] + K['ij'], attrs=views)
    g.addFamily('scaled', simpleParameterSpace(2),
                lambda i: C['ij'] <= (i + 1.0) * A['ij'], attrs=views)
    # Members at 0, 2 and 4 of a family of 3 x 2.
    g.addFamily('holes', [(0, 0), (2, 0), (1, 1)],
                lambda i, j: C['ij'] <= (1.0 + i + 3 * j) * A['ij'], attrs=views)
    D = [Tensor(f'D({i})', (n, n), namespace='sub') for i in range(2)]
    g.add('pick', C['ij'] <= D[0]['ij'] + D[1]['ij'], attrs=views)
    # Members that use operands, or write them, the others do not.
    g.addFamily('each', simpleParameterSpace(2), lambda i: C['ij'] <= D[i]['ij'], attrs=views)
    g.addFamily('swap', simpleParameterSpace(2),
                lambda i: C['ij'] <= A['ij'] if i == 0 else A['ij'] <= C['ij'], attrs=views)
    g.add('copied', C['ij'] <= A['ij'])
    if n == 5:
        E = Tensor('E', (n,))
        g.add('only', C['ij'] <= E['i'] * A['ij'], attrs=views)


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
                ['tensor.cpp', 'init.cpp', 'kernel.cpp', 'pool.cpp', 'subroutine.cpp', 'runtime.cpp']
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
                ['tensor.cpp', 'init.cpp', 'kernel.cpp', 'pool.cpp', 'runtime.cpp']
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
        assert sorted(includes) == ['cstddef', 'cstdint', 'initializer_list', 'limits', 'runtime.h',
                                    'stdexcept', 'string', 'string_view', 'yateto.h', 'yateto/RuntimeView.h']

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
        for name in ('init.h', 'kernel.h', 'tensor.h', 'runtime.h', 'runtime.cpp',
                     'metagen_Wide/runtime.cpp', 'metagen_Narrow/runtime.cpp'):
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


def body(code, name):
    """The body of `struct name`, of the function `name` or of the block `name`, in generated code."""
    match = re.search(r'(?:struct {0}|void {0}\([^)]*\)[^{{;]*|{0}) \{{\n'.format(re.escape(name)), code)
    assert match is not None, name
    depth = 1
    for position in range(match.end(), len(code)):
        depth += {'{': 1, '}': -1}.get(code[position], 0)
        if depth == 0:
            return code[match.end():position]
    raise AssertionError(name)


def lines(code):
    return [line.strip() for line in code.splitlines() if line.strip()]


class TestRuntimeKernels:
    @pytest.fixture
    def out(self, tmp_path):
        metagen(tmp_path)
        return tmp_path

    def test_a_kernel_takes_views_where_it_asks_for_them(self, out):
        runtime_h = (out / 'runtime.h').read_text()
        assert re.findall(r'struct (\w+) \{\n\s*(?:::yateto::|double|void)', runtime_h) == \
            ['each', 'holes', 'matmul', 'only', 'pick', 'scaled', 'swap']
        assert 'copied' not in runtime_h

    def test_a_caller_sets_what_the_pool_does_not(self, out):
        assert lines(body((out / 'runtime.h').read_text(), 'matmul')) == [
            '::yateto::ConstRuntimeView A;',
            '::yateto::ConstRuntimeView B;',
            '::yateto::RuntimeView C;',
            'double alpha = std::numeric_limits<double>::signaling_NaN();',
            '//! Runs the kernel of a variant on what is set here.',
            'void execute(std::size_t variant) const;']
        binding = body((out / 'metagen_Narrow' / 'runtime.cpp').read_text(), 'matmul')
        assert 'krnl.bindGlobals(pool);' in binding
        assert 'args.K' not in binding
        assert 'krnl.alpha = static_cast<float>(args.alpha);' in binding

    def test_a_family_of_kernels_takes_its_indices(self, out):
        runtime_h = (out / 'runtime.h').read_text()
        assert 'void execute(std::size_t variant, unsigned i0) const;' in body(runtime_h, 'scaled')
        binding = body((out / 'metagen_Wide' / 'runtime.cpp').read_text(), 'scaled')
        assert 'krnl.execute(i0);' in binding
        assert 'throw std::out_of_range' in binding
        # Each index within its extent, and a member there.
        holes = body((out / 'metagen_Wide' / 'runtime.cpp').read_text(), 'holes')
        assert 'const unsigned position = 1u * i0 + 3u * i1;' in holes
        assert 'if (i0 >= 3u || i1 >= 2u || position >= 5u || ' \
            '::test::yatetometagen_Wide::kernel::holes::findExecute(i0, i1) == nullptr) {' in holes

    def test_a_member_of_a_family_of_kernels_takes_what_it_uses(self, out):
        each = body((out / 'metagen_Wide' / 'runtime.cpp').read_text(), 'each')
        assert 'switch (position) {' in each
        assert 'args.D(1)' not in body(each, 'case 0:') and 'args.D(0)' not in body(each, 'case 1:')
        swap = body((out / 'metagen_Wide' / 'runtime.cpp').read_text(), 'swap')
        assert re.findall(r'krnl\.(\w+) = (operand\d)\.data\(\);|(operand\d)\.finish', body(swap, 'case 0:')) == \
            [('A', 'operand0', ''), ('C', 'operand1', ''), ('', '', 'operand1')]
        assert re.findall(r'krnl\.(\w+) = (operand\d)\.data\(\);|(operand\d)\.finish', body(swap, 'case 1:')) == \
            [('A', 'operand0', ''), ('C', 'operand1', ''), ('', '', 'operand0')]

    def test_a_family_of_tensors_is_a_family_of_views(self, out):
        assert '::yateto::Family<::yateto::ConstRuntimeView, 2> D;' in \
            lines(body((out / 'runtime.h').read_text(), 'pick'))
        binding = body((out / 'metagen_Wide' / 'runtime.cpp').read_text(), 'pick')
        assert 'krnl.D(1) = operand2.data();' in binding
        assert '*::test::yatetometagen_Wide::sub::init::D::descriptor(1)' in binding

    def test_only_what_a_kernel_writes_is_written_back(self, out):
        binding = body((out / 'metagen_Wide' / 'runtime.cpp').read_text(), 'matmul')
        assert re.findall(r'(operand\d)\.finish\(\);', binding) == ['operand2']
        assert 'krnl.C = operand2.data();' in binding

    def test_a_variant_without_the_kernel_has_none_to_run(self, out):
        dispatch = body((out / 'runtime.cpp').read_text(), 'only::execute')
        assert 'Functions[] = {nullptr, &::test::yatetometagen_Narrow::_runtime::kernel::only};' in dispatch
        assert 'only' not in (out / 'metagen_Wide' / 'runtime.cpp').read_text()

    @staticmethod
    def merged(tmp_path, builds):
        """One generator per build, under its name."""
        m = MetaGenerator(['typename'])
        for key, build in builds.items():
            g = Generator(useArchitectureIdentifiedBy('dhsw'))
            build(g)
            m.add_generator([key], g, gemm_cfg=GeneratorCollection([]), name=key)
        m.generate(str(tmp_path), namespace='test')
        return m

    def test_a_generator_without_such_kernels_binds_nothing(self, tmp_path):
        def build(views):
            def add(g):
                g.add('k', Tensor('y', (2,))['i'] <= Tensor('x', (2,))['i'],
                      attrs={'operands': 'runtime' if views else 'static'})
            return add

        m = self.merged(tmp_path, {'A': build(True), 'B': build(False)})
        assert (tmp_path / 'metagen_B' / 'runtime.cpp').read_text() == '// No kernel of B takes views.\n'
        assert str(tmp_path / 'metagen_B' / 'runtime.cpp') in m.sources(str(tmp_path))['B']
        assert 'Functions[] = {&::test::yatetometagen_A::_runtime::kernel::k, nullptr};' in \
            (tmp_path / 'runtime.cpp').read_text()

    def test_the_members_are_put_together_over_the_variants(self, tmp_path):
        views = {'operands': 'runtime'}

        def first(g):
            # X is a constant here, and out is written.
            X = Tensor('X', (2,), spp=np.array([1.0, 2.0]))
            g.add('k', Tensor('out', (2,))['i'] <= X['i'] + Tensor('F(0)', (2,))['i'], attrs=views)

        def second(g):
            # Here X is written, out only read, and a scalar comes along.
            s = Scalar('s')
            g.add('k', Tensor('X', (2,))['i'] <= s * Tensor('out', (2,))['i'] + Tensor('F(2)', (2,))['i'],
                  attrs=views)

        m = self.merged(tmp_path, {'A': first, 'B': second})
        assert lines(body((tmp_path / 'runtime.h').read_text(), 'k'))[:4] == [
            '::yateto::Family<::yateto::ConstRuntimeView, 3> F;',
            '::yateto::RuntimeView X;',
            '::yateto::RuntimeView out;',
            'double s = std::numeric_limits<double>::signaling_NaN();']
        a = body((tmp_path / 'metagen_A' / 'runtime.cpp').read_text(), 'k')
        assert 'args.X' not in a and 'args.s' not in a and 'args.F(0)' in a
        b = body((tmp_path / 'metagen_B' / 'runtime.cpp').read_text(), 'k')
        assert 'args.X' in b and 'args.F(2)' in b and 'args.F(0)' not in b
        # What either variant reads, its own binding compiles against.
        if shutil.which('c++') is not None:
            sources = m.shared_sources(str(tmp_path)) + \
                [str(tmp_path / f'metagen_{key}' / 'runtime.cpp') for key in 'AB']
            built = subprocess.run(['c++', '-std=c++17', '-fsyntax-only', '-Wall', '-Wextra', '-Werror',
                                    '-Wno-unused-parameter', f'-isystem{INCLUDE}', f'-I{tmp_path}', *sources],
                                   capture_output=True, text=True)
            assert built.returncode == 0, built.stderr

    def test_the_members_of_a_constant_the_pool_has_no_values_for_are_handed_over(self, tmp_path):
        def build(g):
            # The family is a constant for the kernel as long as its last
            # member is one, and the pool holds the members with values only.
            F = [Tensor('F(0)', (2,)), Tensor('F(1)', (2,), spp=np.array([1.0, 2.0]))]
            g.add('k', Tensor('out', (2,))['i'] <= F[0]['i'] + F[1]['i'], attrs={'operands': 'runtime'})

        self.merged(tmp_path, {'A': build})
        assert '::yateto::Family<::yateto::ConstRuntimeView, 1> F;' in \
            lines(body((tmp_path / 'runtime.h').read_text(), 'k'))
        binding = body((tmp_path / 'metagen_A' / 'runtime.cpp').read_text(), 'k')
        assert binding.index('krnl.bindGlobals(pool);') < binding.index('krnl.F(0) = operand0.data();')
        assert 'args.F(1)' not in binding

    def test_a_member_is_the_same_thing_in_every_variant(self, tmp_path):
        views = {'operands': 'runtime'}

        def tensor(g):
            g.add('k', Tensor('out', (2,))['i'] <= Tensor('x', (2,))['i'], attrs=views)

        def scalar(g):
            g.add('k', Tensor('out', (2,))['i'] <= Scalar('x') * Tensor('y', (2,))['i'], attrs=views)

        with pytest.raises(ValueError, match='takes x as a view of x in one generator and as a float'):
            self.merged(tmp_path, {'A': tensor, 'B': scalar})

    def test_a_family_of_kernels_keeps_its_number_of_indices(self, tmp_path):
        def build(rank):
            def add(g):
                g.addFamily('k', simpleParameterSpace(*([2] * rank)),
                            lambda *i: Tensor('out', (2,))['i'] <= Tensor('x', (2,))['i'],
                            attrs={'operands': 'runtime'})
            return add

        with pytest.raises(ValueError, match='kernel family of 1 indices in one generator and of 2'):
            self.merged(tmp_path, {'A': build(1), 'B': build(2)})


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

// Values the way the caller holds them, whatever the variant: doubles by columns, no padding.
// Wide holds them the same way, and takes them as they are; Narrow copies them.
struct Plain {
  unsigned shape[2];
  unsigned start[2] = {0, 0};
  unsigned stride[2];
  yateto::TensorDescriptor layout;
  Plain(unsigned rank, unsigned n)
      : shape{n, n}, stride{1, n},
        layout{yateto::Datatype::F64, yateto::Storage::Dense, rank, shape, start, shape, stride,
               nullptr, nullptr, nullptr, rank == 1 ? n : n * n, 8} {}
  Plain(Plain const&) = delete;
};

int checkRuntimeKernels() {
  using namespace test::runtime;
  for (std::size_t variant = 0; variant < VariantCount; ++variant) {
    const int code = 40 + 20 * static_cast<int>(variant);
    const unsigned n = init::A::descriptor(variant)->shape[0];
    const Plain matrix(2, n);
    const Plain vector(1, n);
    alignas(64) double a[25], b[25], c[25], e[5];
    for (unsigned i = 0; i < n * n; ++i) {
      a[i] = static_cast<double>(i % 7) + 1;
      b[i] = static_cast<double>(i % 5) - 2;
      c[i] = -1;
    }
    kernel::matmul product;
    product.alpha = 2;
    product.A = {&matrix.layout, a};
    product.B = {&matrix.layout, b};
    product.C = {&matrix.layout, c};
    product.execute(variant);
    for (unsigned i = 0; i < n; ++i) {
      for (unsigned j = 0; j < n; ++j) {
        double expected = 1.0 + n * i + j;
        for (unsigned k = 0; k < n; ++k) {
          expected += 2.0 * a[i + n * k] * b[k + n * j];
        }
        if (std::abs(c[i + n * j] - expected) > 1e-4 * std::abs(expected)) return code;
      }
    }
    kernel::scaled scaled;
    scaled.A = {&matrix.layout, a};
    scaled.C = {&matrix.layout, c};
    scaled.execute(variant, 1);
    for (unsigned i = 0; i < n * n; ++i) {
      if (c[i] != 2 * a[i]) return code + 1;
    }
    try {
      scaled.execute(variant, 2);
      return code + 2;
    } catch (std::out_of_range const&) {
    }
    kernel::holes holes;
    holes.A = {&matrix.layout, a};
    holes.C = {&matrix.layout, c};
    holes.execute(variant, 1, 1);
    for (unsigned i = 0; i < n * n; ++i) {
      if (c[i] != 5 * a[i]) return code + 7;
    }
    const unsigned missing[][2] = {{1, 0}, {3, 0}, {2, 1}, {0, 2}};
    for (auto const& index : missing) {
      try {
        holes.execute(variant, index[0], index[1]);
        return code + 8;
      } catch (std::out_of_range const&) {
      }
    }
    kernel::pick pick;
    pick.C = {&matrix.layout, c};
    pick.D(0) = {&matrix.layout, a};
    try {
      pick.execute(variant);
      return code + 3;
    } catch (std::invalid_argument const&) {
    }
    pick.D(1) = {&matrix.layout, b};
    pick.execute(variant);
    for (unsigned i = 0; i < n * n; ++i) {
      if (c[i] != a[i] + b[i]) return code + 4;
    }
    kernel::each each;
    each.C = {&matrix.layout, c};
    each.D(1) = {&matrix.layout, b};
    each.execute(variant, 1);
    for (unsigned i = 0; i < n * n; ++i) {
      if (c[i] != b[i]) return code + 9;
    }
    try {
      each.execute(variant, 0);
      return code + 10;
    } catch (std::invalid_argument const&) {
    }
    // What a member only reads is not written back, not even rounded.
    kernel::swap swap;
    alignas(64) double x[25], y[25];
    for (unsigned i = 0; i < n * n; ++i) {
      x[i] = 0.1 * (i + 1);
      y[i] = -1;
    }
    swap.A = {&matrix.layout, x};
    swap.C = {&matrix.layout, y};
    swap.execute(variant, 0);
    for (unsigned i = 0; i < n * n; ++i) {
      if (x[i] != 0.1 * (i + 1) || std::abs(y[i] - x[i]) > 1e-6 * x[i]) return code + 11;
      y[i] = 3;
    }
    swap.execute(variant, 1);
    for (unsigned i = 0; i < n * n; ++i) {
      if (x[i] != 3) return code + 12;
    }
    kernel::only only;
    for (unsigned i = 0; i < n; ++i) {
      e[i] = i + 1;
    }
    only.A = {&matrix.layout, a};
    only.E = {&vector.layout, e};
    only.C = {&matrix.layout, c};
    if (variant == 0) {
      try {
        only.execute(variant);
        return code + 5;
      } catch (std::invalid_argument const&) {
      }
    } else {
      only.execute(variant);
      for (unsigned i = 0; i < n * n; ++i) {
        if (c[i] != e[i % n] * a[i]) return code + 6;
      }
    }
  }
  try {
    kernel::matmul{}.execute(2);
    return 90;
  } catch (std::out_of_range const&) {
  }
  return 0;
}

int main() {
  if (int result = check<test::Wide>(4, 10)) return result;
  if (int result = check<test::Narrow>(5, 20)) return result;
  if (int result = checkRuntime()) return result;
  return checkRuntimeKernels();
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
