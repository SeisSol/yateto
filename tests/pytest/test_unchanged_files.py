"""A file is written only where its code changes.

A build tool compares the time of a file with the times of what it built from
it, and builds again all that depends on a file that is written anew.
Generating again what is there already leaves the files as they are -- those
of the routines as well, which their generators append to a file one by one.
"""
from __future__ import annotations

import os

import pytest

from yateto import Generator, GlobalRoutineCache, Tensor, useArchitectureIdentifiedBy
from yateto.codegen.cache import RoutineGenerator
from yateto.codegen.code import Cpp, writeIfChanged
from yateto.gemm_configuration import GeneratorCollection

#: A time long before the tests run, in seconds since the epoch.
PAST = 1_000_000_000


def age(directory):
    """Sets the time of every file under `directory` to PAST."""
    for path in directory.rglob('*'):
        if path.is_file():
            os.utime(path, (PAST, PAST))


def written(directory):
    """The files under `directory` written since `age`, by their paths in it."""
    return sorted(str(path.relative_to(directory)) for path in directory.rglob('*')
                  if path.is_file() and path.stat().st_mtime != PAST)


def contents(directory):
    return {str(path.relative_to(directory)): path.read_text()
            for path in directory.rglob('*') if path.is_file()}


class TestWriteIfChanged:
    def test_a_new_file_is_written(self, tmp_path):
        assert writeIfChanged(str(tmp_path / 'a.h'), 'int a;\n')
        assert (tmp_path / 'a.h').read_text() == 'int a;\n'

    def test_a_file_that_holds_the_code_already_is_left_as_it_is(self, tmp_path):
        (tmp_path / 'a.h').write_text('int a;\n')
        age(tmp_path)
        assert not writeIfChanged(str(tmp_path / 'a.h'), 'int a;\n')
        assert written(tmp_path) == []

    @pytest.mark.parametrize('held', ['int b;\n', 'int a;\nint b;\n', ''])
    def test_a_file_of_other_code_is_written(self, tmp_path, held):
        (tmp_path / 'a.h').write_text(held)
        assert writeIfChanged(str(tmp_path / 'a.h'), 'int a;\n')
        assert (tmp_path / 'a.h').read_text() == 'int a;\n'

    def test_a_file_that_is_no_text_is_written(self, tmp_path):
        (tmp_path / 'a.h').write_bytes(b'\xff\xfe')
        assert writeIfChanged(str(tmp_path / 'a.h'), 'int a;\n')
        assert (tmp_path / 'a.h').read_text() == 'int a;\n'


class TestCpp:
    def test_the_file_is_written_at_the_end_of_the_block(self, tmp_path):
        path = tmp_path / 'a.h'
        with Cpp(str(path)) as cpp:
            cpp('int a;')
            assert not path.exists()
        assert path.read_text() == 'int a;\n'

    def test_a_block_that_fails_leaves_the_file_as_it_is(self, tmp_path):
        path = tmp_path / 'a.h'
        path.write_text('int a;\n')
        with pytest.raises(RuntimeError, match='failed'):
            with Cpp(str(path)) as cpp:
                cpp('int b;')
                raise RuntimeError('the generator failed')
        assert path.read_text() == 'int a;\n'


def generate(directory, scale=1.0):
    """The files of a kernel that computes c = scale * a * b."""
    a, b, c = (Tensor(name, (5, 5)) for name in 'abc')
    generator = Generator(useArchitectureIdentifiedBy('dhsw'))
    generator.add('k', c['ij'] <= scale * a['ik'] * b['kj'])
    generator.generate(str(directory), gemm_cfg=GeneratorCollection([]), unit_tests='doctest')


class TestGenerator:
    def test_generating_again_writes_nothing(self, tmp_path):
        generate(tmp_path)
        age(tmp_path)
        generate(tmp_path)
        assert written(tmp_path) == []

    def test_a_changed_kernel_writes_the_files_that_change_alone(self, tmp_path):
        generate(tmp_path)
        before = contents(tmp_path)
        age(tmp_path)
        generate(tmp_path, scale=2.0)
        after = contents(tmp_path)
        changed = sorted(name for name in after if after[name] != before.get(name))
        assert 'kernel.cpp' in changed
        assert len(changed) < len(after)
        assert written(tmp_path) == changed


class Appending(RoutineGenerator):
    """A routine its generator appends to the file it is given, as a GEMM tool
    does."""

    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        return type(self) is type(other) and self.value == other.value

    def header(self, cpp):
        cpp.includeSys('cmath')

    def __call__(self, routineName, fileName):
        with open(fileName, 'a') as file:
            file.write(f'void {routineName}(double* x) {{ x[0] += {self.value}; }}\n')
        return f'void {routineName}(double* x);'


class DeviceAppending(Appending):
    def target(self):
        return 'gpu'


#: In the order they are added.
ROUTINES = {
    'first': Appending(1.0),
    'second': Appending(2.0),
    'third': Appending(3.0),
    'device_first': DeviceAppending(1.0),
    'device_second': DeviceAppending(2.0),
}


def routines(directory, changed={}, shards=None):
    """The routines of ROUTINES, those of `changed` instead, for a generator
    in `directory`/gen."""
    (directory / 'gen').mkdir(exist_ok=True)
    cache = GlobalRoutineCache()
    cache.register(str(directory / 'gen'))
    for name, routine in dict(ROUTINES, **changed).items():
        cache.cache.addRoutine(name, routine)
    cache.generate(str(directory), shards=shards)


class TestRoutines:
    @pytest.mark.parametrize('shards', [None, {'cpu': 2, 'gpu': 2}])
    def test_writing_them_again_writes_nothing(self, tmp_path, shards):
        routines(tmp_path, shards=shards)
        age(tmp_path)
        routines(tmp_path, shards=shards)
        assert written(tmp_path) == []

    def test_a_changed_routine_writes_its_file_alone(self, tmp_path):
        routines(tmp_path)
        age(tmp_path)
        # of the same declaration, and of no other target
        routines(tmp_path, changed={'second': Appending(4.0)})
        assert written(tmp_path) == ['subroutine.cpp']
        assert '{ x[0] += 4.0; }' in (tmp_path / 'subroutine.cpp').read_text()

    def test_a_new_routine_writes_the_file_and_the_declarations(self, tmp_path):
        routines(tmp_path)
        age(tmp_path)
        routines(tmp_path, changed={'fourth': Appending(4.0)})
        assert written(tmp_path) == ['subroutine.cpp', 'subroutine.h']
