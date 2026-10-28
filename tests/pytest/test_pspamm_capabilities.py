"""What PSpaMM says it supports beyond the GEMM itself.

A PSpaMM that knows the option prints its capabilities as JSON with
--capabilities; one that does not fails on the option. yateto asks once per
program, in its own process where the program would run on this interpreter,
and takes a failure for nothing more than the GEMM.
"""
from __future__ import annotations

import os
import stat
import sys

import pytest

from yateto import useArchitectureIdentifiedBy
from yateto.codegen.gemm import gemmgen
from yateto.gemm_configuration import PSpaMM

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
    """A directory first on the path, for programs of the test; what a program
    is and what it supports are decided once per program, so the decisions are
    cleared around the test."""
    monkeypatch.setenv('PATH', str(tmp_path) + os.pathsep + os.environ.get('PATH', ''))
    gemmgen._pspammInProcess.cache_clear()
    gemmgen.pspammCapabilities.cache_clear()
    yield tmp_path
    gemmgen._pspammInProcess.cache_clear()
    gemmgen.pspammCapabilities.cache_clear()


def program(directory, name, script):
    path = directory / name
    path.write_text(script)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return name


class TestCapabilities:
    def test_what_a_program_prints_is_what_it_supports(self, programs):
        name = program(programs, 'pspamm-new', '#!/bin/sh\necho \'{"abake": true}\'\n')

        assert gemmgen.pspammCapabilities(name) == {'abake': True}

    def test_a_program_that_does_not_know_the_option_supports_nothing_more(self, programs):
        name = program(programs, 'pspamm-old', '#!/bin/sh\necho "unrecognized arguments" >&2\nexit 2\n')

        assert gemmgen.pspammCapabilities(name) == {}

    def test_nor_does_one_that_prints_something_else(self, programs):
        name = program(programs, 'pspamm-chatty', '#!/bin/sh\necho "usage: pspamm-generator"\n')

        assert gemmgen.pspammCapabilities(name) == {}

    def test_nor_one_that_is_not_there(self, programs):
        assert gemmgen.pspammCapabilities('pspamm-nowhere') == {}

    def test_the_tool_asks_its_program(self, programs):
        name = program(programs, 'pspamm-tool', '#!/bin/sh\necho \'{"abake": false}\'\n')

        assert PSpaMM(useArchitectureIdentifiedBy('dhsw'), cmd=name).capabilities() == {'abake': False}

    def test_in_this_process_as_from_the_program(self, programs, monkeypatch):
        """Whatever the installed PSpaMM says, it says it the same either way."""
        pytest.importorskip('pypspamm.cli')
        name = program(programs, 'pspamm-here', f'#!{sys.executable}\n{PSPAMM_SCRIPT}')
        assert gemmgen._pspammInProcess(name) is not None
        here = gemmgen.pspammCapabilities(name)

        gemmgen.pspammCapabilities.cache_clear()
        with monkeypatch.context() as patch:
            patch.setattr(gemmgen, '_pspammInProcess', lambda cmd: None)
            fromProgram = gemmgen.pspammCapabilities(name)

        assert fromProgram == here
