"""PSpaMM, run in the process that generates the kernels.

The program of PSpaMM is a script that runs the command line of its package.
Where that script runs on the interpreter that runs yateto, yateto runs the
command line in its own process instead of starting the program for every
routine, which is most of the time a routine takes. What is pinned here is
which programs run in the process, and that a routine written there is the
routine the program writes.
"""
from __future__ import annotations

import os
import shutil
import stat
import sys

import pytest

from yateto import Generator, Tensor, useArchitectureIdentifiedBy
from yateto.codegen.gemm import gemmgen
from yateto.gemm_configuration import GeneratorCollection, PSpaMM

#: What the script PSpaMM installs runs.
PSPAMM_SCRIPT = """import re
import sys
from pypspamm.cli import main
if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\\.pyw|\\.exe)?$', '', sys.argv[0])
    sys.exit(main())
"""


@pytest.fixture
def programs(tmp_path, monkeypatch):
    """A directory first on the path, for programs of the test; whether a
    program runs in the process is decided once per program, so the decisions
    are cleared around the test."""
    monkeypatch.setenv('PATH', str(tmp_path) + os.pathsep + os.environ.get('PATH', ''))
    gemmgen._pspammInProcess.cache_clear()
    yield tmp_path
    gemmgen._pspammInProcess.cache_clear()


def program(directory, name, shebang, script=PSPAMM_SCRIPT):
    path = directory / name
    path.write_text(f'#!{shebang}\n{script}')
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return name


class TestWhichProgramRunsInProcess:
    def test_the_script_of_pspamm_on_this_interpreter(self, programs):
        pytest.importorskip('pypspamm.cli')
        assert gemmgen._pspammInProcess(program(programs, 'pspamm-here', sys.executable)) is not None

    def test_the_script_of_pspamm_on_the_interpreter_env_finds(self, programs, monkeypatch):
        pytest.importorskip('pypspamm.cli')
        name = os.path.basename(sys.executable)
        monkeypatch.setenv('PATH', str(programs) + os.pathsep + os.path.dirname(sys.executable))
        assert gemmgen._pspammInProcess(program(programs, 'pspamm-env', f'/usr/bin/env {name}')) is not None

    def test_not_a_program_that_is_not_there(self, programs):
        assert gemmgen._pspammInProcess('pspamm-nowhere') is None

    def test_not_on_another_interpreter(self, programs):
        assert gemmgen._pspammInProcess(program(programs, 'pspamm-there', '/no/such/python3')) is None

    def test_not_on_this_interpreter_with_options(self, programs):
        # An option such as -E or -S changes what the program imports.
        assert gemmgen._pspammInProcess(program(programs, 'pspamm-options', f'{sys.executable} -E')) is None

    def test_not_a_script_that_runs_something_else(self, programs):
        wrapper = 'import subprocess, sys\nsys.exit(subprocess.call(["pspamm-generator"] + sys.argv[1:]))\n'
        assert gemmgen._pspammInProcess(program(programs, 'pspamm-wrapper', sys.executable, wrapper)) is None

    def test_not_a_binary(self, programs):
        (programs / 'pspamm-binary').write_bytes(b'\x7fELF\x02\x01\x01\x00')
        (programs / 'pspamm-binary').chmod(0o755)
        assert gemmgen._pspammInProcess('pspamm-binary') is None


def installed_pspamm():
    """The program of PSpaMM, if it is installed for this interpreter."""
    pytest.importorskip('pypspamm.cli')
    gemmgen._pspammInProcess.cache_clear()
    if shutil.which('pspamm-generator') is None or gemmgen._pspammInProcess('pspamm-generator') is None:
        pytest.skip('needs the program of PSpaMM, installed for this interpreter')
    return 'pspamm-generator'


def routines(tmp_path, name):
    """The routines PSpaMM writes for two GEMMs whose code has loops."""
    arch = useArchitectureIdentifiedBy('dhsw')
    g = Generator(arch)
    A = Tensor('A', (16, 32))
    B = Tensor('B', (32, 24))
    C = Tensor('C', (16, 24))
    D = Tensor('D', (24, 8))
    E = Tensor('E', (16, 8))
    g.add('first', C['ij'] <= A['ik'] * B['kj'])
    g.add('second', E['ij'] <= C['ik'] * D['kj'])
    out = tmp_path / name
    out.mkdir()
    g.generate(str(out), gemm_cfg=GeneratorCollection([PSpaMM(arch)]))
    code = (out / 'subroutine.cpp').read_text()
    assert code.count('pspamm_num_total_flops +=') == 2
    return code


class TestTheRoutine:
    def test_is_the_one_the_program_writes(self, tmp_path, monkeypatch):
        installed_pspamm()
        here = routines(tmp_path, 'here')
        monkeypatch.setattr(gemmgen, '_pspammInProcess', lambda cmd: None)
        assert routines(tmp_path, 'program') == here

    def test_of_a_failing_command_line_is_an_error(self):
        cmd = installed_pspamm()
        argv = list(sys.argv)
        with pytest.raises(RuntimeError, match='in-process'):
            gemmgen._pspammInProcess(cmd)(cmd, [cmd, '--no-such-option'])
        assert sys.argv == argv
