"""The addresses a loop over GEMMs hands its GEMMs to prefetch from.

A kernel that prefetches a tensor hands each GEMM the address of the part of
that tensor its result corresponds to. A loop over GEMMs steps that address
along in pointers of its own, over the indices of the result it loops over
and over the indices it sums over. Not every GEMM generator reads them --
the GEMMs yateto writes itself do not -- and the code compiles without
warnings either way.
"""
from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

from yateto import Generator, Tensor, useArchitectureIdentifiedBy
from yateto.gemm_configuration import GeneratorCollection


INCLUDE = pathlib.Path(__file__).resolve().parents[2] / 'include'


def kernel(loops):
    """C = A B, and a tensor shaped like C to prefetch; the loop over GEMMs
    runs over an index of the result, over an index it sums, or over both."""
    if loops == 'result':
        A, B, C = Tensor('A', (8, 6, 4)), Tensor('B', (6, 5)), Tensor('C', (8, 4, 5))
        expression = C['ijl'] <= A['ikj'] * B['kl']
    elif loops == 'sum':
        A, B, C = Tensor('A', (8, 3, 4)), Tensor('B', (4, 3, 5)), Tensor('C', (8, 5))
        expression = C['ij'] <= A['ilk'] * B['klj']
    else:
        A, B, C = Tensor('A', (8, 3, 4, 2)), Tensor('B', (4, 3, 5)), Tensor('C', (8, 5, 2))
        expression = C['ijm'] <= A['ilkm'] * B['klj']
    return expression, Tensor('P', C.shape())


@pytest.mark.skipif(shutil.which('c++') is None, reason='needs a C++ compiler')
@pytest.mark.parametrize('loops, pointers', [
    ('result', ['_Cprefetch']),
    ('sum', ['_Cprefetchin']),
    ('both', ['_Cprefetch', '_Cprefetchin']),
])
def test_a_gemm_that_does_not_prefetch_leaves_the_address_unread(tmp_path, loops, pointers):
    g = Generator(useArchitectureIdentifiedBy('dhsw'))
    expression, prefetched = kernel(loops)
    g.add('k', expression, prefetch=prefetched)
    # without GEMM tools, yateto writes the GEMMs itself
    g.generate(str(tmp_path), gemm_cfg=GeneratorCollection([]))
    code = (tmp_path / 'kernel.cpp').read_text()
    for pointer in pointers:
        assert f'* {pointer} = ' in code

    built = subprocess.run(['c++', '-std=c++17', '-fsyntax-only', '-Wall', '-Wextra', '-Werror',
                            f'-isystem{INCLUDE}', f'-I{tmp_path}', str(tmp_path / 'kernel.cpp')],
                           capture_output=True, text=True)
    assert built.returncode == 0, built.stderr
