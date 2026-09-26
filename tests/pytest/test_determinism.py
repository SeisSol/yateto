"""The generated code does not depend on the interpreter's hash seed.

Set iteration order varies with PYTHONHASHSEED from one run to the next, so
anything that reaches the emitted code through a set has to be put in order
first -- otherwise two identical invocations produce different files, and
every diff of generated code is noise.
"""

import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]

# Two temporaries of different sizes die at the same statement, and two more
# are born after it: which buffer each of the new ones gets is decided by the
# order in which the dead ones went back on the free list.
PROGRAM = '''
import contextlib, io, sys, tempfile
from yateto import Generator, GeneratorCollection, Tensor
from yateto.arch import useArchitectureIdentifiedBy
import yateto.functions as yf

N, M = 8, 2
A = Tensor('A', (N, N)); B = Tensor('B', (N, N))
P = Tensor('P', (N, M)); Q = Tensor('Q', (M, N))
C = Tensor('C', (N, N)); D = Tensor('D', (N, N))
out = Tensor('out', (N, N)); out2 = Tensor('out2', (N, N))
g = Generator(useArchitectureIdentifiedBy('dhsw'))
g.add('k', [
  out['ij'] <= yf.add(A['ik'] * B['kj'], yf.exp(P['im']) * Q['mj']),
  out2['ij'] <= yf.add(yf.exp(P['im']) * Q['mj'], C['ik'] * D['kj']),
])
d = tempfile.mkdtemp()
with contextlib.redirect_stdout(io.StringIO()):
  g.generate(d, gemm_cfg=GeneratorCollection([]))
sys.stdout.write(open(d + '/kernel.cpp').read())
'''


def generated(seed):
    env = dict(os.environ, PYTHONHASHSEED=str(seed),
               PYTHONPATH=os.pathsep.join([str(ROOT), os.environ.get('PYTHONPATH', '')]))
    return subprocess.run([sys.executable, '-c', PROGRAM], env=env, cwd=ROOT,
                          check=True, capture_output=True, text=True).stdout


@pytest.mark.parametrize('seed', [1, 2, 3, 4, 5, 6, 7])
def test_temporaries_get_the_same_buffers_whatever_the_seed(seed):
    assert generated(seed) == generated(0)
