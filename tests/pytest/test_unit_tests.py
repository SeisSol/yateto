"""The unit tests a generator writes for its kernels.

Every kernel is tested against a reference the test computes itself. For
doctest, the test of a kernel is a function of its own, in the namespace
`unit_test` in that of the kernels, and the one test case of the generator
runs each of them in a subcase: written into the subcases themselves, the
tests of all kernels would be one function, which a compiler takes several
times as long for as for the functions it is made of.
"""
from __future__ import annotations

import re

import pytest

from yateto import Generator, Tensor, simpleParameterSpace, useArchitectureIdentifiedBy
from yateto.gemm_configuration import GeneratorCollection

DOCTEST = 'test-kernel.cpp'
CXXTEST = 'KernelTest.t.h'


def generate(out, **kwargs):
    """A kernel and a family of two, generated into `out`."""
    g = Generator(useArchitectureIdentifiedBy('dhsw'))
    A = Tensor('A', (4, 4))
    B = Tensor('B', (4, 4))
    C = Tensor('C', (4, 4))
    g.add('product', C['ij'] <= A['ik'] * B['kj'])
    g.addFamily('scaled', simpleParameterSpace(2), lambda i: C['ij'] <= (i + 1.0) * A['ij'])
    out.mkdir(exist_ok=True)
    g.generate(str(out), gemm_cfg=GeneratorCollection([]), **kwargs)
    return out


class TestDoctest:
    @pytest.fixture
    def source(self, tmp_path):
        return (generate(tmp_path) / DOCTEST).read_text()

    def test_every_kernel_is_tested_in_a_function_of_its_own(self, source):
        namespace = re.search(r'namespace yateto \{\s*namespace unit_test \{(.*)\} // namespace unit_test',
                              source, re.DOTALL)
        assert namespace is not None
        assert re.findall(r'^ *void (\w+)\(\) \{', namespace.group(1), re.MULTILINE) == \
            ['product', '_scaled_0', '_scaled_1']

    def test_the_test_case_runs_each_function_in_a_subcase(self, source):
        test_case = source[source.index('TEST_CASE("yateto kernels for \\"yateto\\"")'):]
        assert re.findall(r'SUBCASE\("(\w+)"\) \{ yateto::unit_test::(\w+)\(\); \}', test_case) == \
            [('product', 'product'), ('_scaled_0', '_scaled_0'), ('_scaled_1', '_scaled_1')]
