"""The routines of several generators, written once for all of them.

A GlobalRoutineCache collects the routines the kernels call -- those of GEMM
tools and of device code generators -- and writes each of them once, with its
declaration in `subroutine.h`. What a process generated can be handed to the
process that writes the routines, as plain data (`export`, `merge`), and the
routines of a target can be spread over several files that compile at the
same time (`shards`).
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess

import pytest

from yateto import GlobalRoutineCache
from yateto.codegen.cache import RoutineGenerator


class Routine(RoutineGenerator):
    """A routine that adds 1 to its argument `size` times, one statement each."""

    INCLUDE = 'cmath'

    def __init__(self, size):
        self.size = size

    def __eq__(self, other):
        return type(self) is type(other) and self.size == other.size

    def header(self, cpp):
        cpp.includeSys(self.INCLUDE)

    def __call__(self, routineName, fileName):
        with open(fileName, 'a') as file:
            file.write(f'void {routineName}(double* x) {{\n' + '  x[0] += 1.0;\n' * self.size + '}\n')
        return f'void {routineName}(double* x);'


class OtherRoutine(Routine):
    """A routine of another kind, which needs includes of its own."""

    INCLUDE = 'cstdio'


class DeviceRoutine(Routine):
    """A routine of a device code generator."""

    INCLUDE = 'cstddef'

    def target(self):
        return 'gpu'


class Unsteady(Routine):
    """A routine its generator writes differently every time -- the same
    routine all the same, as its identity says."""

    written = 0

    def identity(self):
        return f'unsteady {self.size}'

    def __call__(self, routineName, fileName):
        Unsteady.written += 1
        with open(fileName, 'a') as file:
            file.write(f'// written {Unsteady.written} times\n')
        return super().__call__(routineName, fileName)


#: In the order they are added.
ROUTINES = {
    'large': Routine(40),
    'tiny': Routine(1),
    'medium': Routine(25),
    'small': Routine(20),
    'other': OtherRoutine(30),
    'device_a': DeviceRoutine(3),
    'device_b': DeviceRoutine(5),
}


def cache(routines, dirs=()):
    """A cache of `routines`, by name, for the generators in `dirs`."""
    routineCache = GlobalRoutineCache()
    for directory in dirs:
        routineCache.register(str(directory))
    for name, routine in routines.items():
        routineCache.cache.addRoutine(name, routine)
    return routineCache


def defined(path):
    return re.findall(r'^void (\w+)\(', path.read_text(), re.MULTILINE)


class TestHandOver:
    def test_routines_handed_over_are_written_as_where_they_were_generated(self, tmp_path):
        here = tmp_path / 'here'
        there = tmp_path / 'there'
        for root in (here, there):
            (root / 'gen').mkdir(parents=True)
        cache(ROUTINES, dirs=[here / 'gen']).generate(str(here))
        exported = json.loads(json.dumps(cache(ROUTINES, dirs=[there / 'gen']).export(root=str(there))))
        receiving = GlobalRoutineCache()
        receiving.merge(exported, root=str(there))
        receiving.generate(str(there))
        for name in ('subroutine.h', 'subroutine.cpp', 'gpulike_subroutine.cpp', 'gen/subroutine.h'):
            assert (there / name).read_text() == (here / name).read_text(), name

    def test_the_data_does_not_depend_on_where_the_directories_are(self, tmp_path):
        first = cache(ROUTINES, dirs=[tmp_path / 'first' / 'gen']).export(root=str(tmp_path / 'first'))
        second = cache(ROUTINES, dirs=[tmp_path / 'second' / 'gen']).export(root=str(tmp_path / 'second'))
        assert first == second
        assert first['dirs'] == ['gen']

    def test_routines_of_several_processes_are_written_once(self, tmp_path):
        receiving = GlobalRoutineCache()
        receiving.merge(cache({'large': Routine(40), 'tiny': Routine(1)}).export())
        receiving.merge(cache({'tiny': Routine(1), 'small': Routine(20)}).export())
        receiving.generate(str(tmp_path))
        assert defined(tmp_path / 'subroutine.cpp') == ['large', 'tiny', 'small']

    def test_two_routines_of_one_name_are_an_error(self):
        receiving = GlobalRoutineCache()
        receiving.merge(cache({'large': Routine(40)}).export())
        with pytest.raises(RuntimeError, match='large'):
            receiving.merge(cache({'large': Routine(39)}).export())

    def test_a_directory_is_registered_once(self, tmp_path):
        exported = cache({}, dirs=[tmp_path / 'gen']).export()
        receiving = GlobalRoutineCache()
        receiving.merge(exported)
        receiving.merge(exported)
        assert receiving.dirs == [str(tmp_path / 'gen')]

    def test_a_routine_is_the_one_its_identity_says_whatever_its_code(self, tmp_path):
        receiving = GlobalRoutineCache()
        receiving.merge(cache({'unsteady': Unsteady(3)}).export())
        receiving.merge(cache({'unsteady': Unsteady(3)}).export())
        receiving.generate(str(tmp_path))
        assert defined(tmp_path / 'subroutine.cpp') == ['unsteady']

    def test_two_routines_of_one_name_and_two_identities_are_an_error(self):
        receiving = GlobalRoutineCache()
        receiving.merge(cache({'unsteady': Unsteady(3)}).export())
        with pytest.raises(RuntimeError, match='unsteady'):
            receiving.merge(cache({'unsteady': Unsteady(4)}).export())


class TestIdentityOfAGemm:
    """The routines of a GEMM tool are the same where their GEMMs are, as
    ExecuteGemmGen compares them: their code need not be, PSpaMM assigns the
    registers of its routines for ARM differently from one process to the
    next."""

    @staticmethod
    def gemm(sppA=None, **changed):
        from yateto import useArchitectureIdentifiedBy
        from yateto.codegen.gemm.gemmgen import ExecuteGemmGen
        from yateto.gemm_configuration import PSpaMM
        from yateto.type import Datatype

        arch = useArchitectureIdentifiedBy('dhsw')
        descr = {'M': 8, 'N': 4, 'K': 2, 'LDA': 8, 'LDB': 2, 'LDC': 8, 'alpha': 1, 'beta': 0,
                 'alignedA': 1, 'alignedC': 1, 'prefetch': 'nopf', 'transA': False, 'transB': False,
                 'datatypeA': Datatype.F64, 'datatypeB': Datatype.F64, 'datatypeC': Datatype.F64}
        descr.update(changed)
        return ExecuteGemmGen(arch, descr, sppA, None, None, None, PSpaMM(arch))

    def test_one_gemm_has_one_identity(self):
        assert self.gemm().identity() == self.gemm().identity()

    def test_another_gemm_has_another(self):
        assert self.gemm().identity() != self.gemm(K=3).identity()

    def test_a_sparsity_pattern_tells_two_apart(self):
        assert self.gemm(sppA=[(0, 0), (1, 1)]).identity() != self.gemm(sppA=[(0, 0), (2, 1)]).identity()


class TestSources:
    def test_one_file_per_target_by_default(self, tmp_path):
        assert GlobalRoutineCache.sources(str(tmp_path)) == {
            'cpu': [str(tmp_path / 'subroutine.cpp')],
            'gpu': [str(tmp_path / 'gpulike_subroutine.cpp')],
        }

    def test_more_files_are_numbered(self, tmp_path):
        assert GlobalRoutineCache.sources(str(tmp_path), {'gpu': 3}) == {
            'cpu': [str(tmp_path / 'subroutine.cpp')],
            'gpu': [str(tmp_path / f'gpulike_subroutine_{index}.cpp') for index in range(3)],
        }

    def test_a_target_takes_at_least_one_file(self, tmp_path):
        with pytest.raises(ValueError, match='at least one file'):
            GlobalRoutineCache.sources(str(tmp_path), {'cpu': 0})


SHARDS = {'cpu': 2, 'gpu': 2}


class TestShards:
    @pytest.fixture
    def out(self, tmp_path):
        cache(ROUTINES).generate(str(tmp_path), shards=SHARDS)
        return tmp_path

    def test_every_routine_is_written_once(self, out):
        for target, paths in GlobalRoutineCache.sources(str(out), SHARDS).items():
            written = [name for path in paths for name in defined(out / path)]
            assert sorted(written) == sorted(name for name, routine in ROUTINES.items()
                                             if routine.target() == target)

    def test_the_header_declares_them_in_the_order_they_were_added(self, out):
        assert re.findall(r'void (\w+)\(', (out / 'subroutine.h').read_text()) == list(ROUTINES)

    def test_every_file_includes_what_the_routines_of_its_target_need(self, out):
        for target, includes in (('cpu', ['cmath', 'cstdio']), ('gpu', ['cstddef'])):
            for path in GlobalRoutineCache.sources(str(out), SHARDS)[target]:
                assert re.findall(r'#include <(\w+)>', (out / path).read_text()) == includes

    def test_the_largest_go_first_into_the_smallest_file(self, out):
        # by the size of their code: large, other, medium, small, tiny; each
        # into the file that is the smaller one so far
        assert defined(out / 'subroutine_0.cpp') == ['large', 'small']
        # and in a file, in the order they were added
        assert defined(out / 'subroutine_1.cpp') == ['tiny', 'medium', 'other']

    def test_one_file_is_what_the_default_writes(self, tmp_path):
        (tmp_path / 'default').mkdir()
        (tmp_path / 'one').mkdir()
        cache(ROUTINES).generate(str(tmp_path / 'default'))
        cache(ROUTINES).generate(str(tmp_path / 'one'), shards={'cpu': 1, 'gpu': 1})
        for name in ('subroutine.h', 'subroutine.cpp', 'gpulike_subroutine.cpp'):
            assert (tmp_path / 'one' / name).read_text() == (tmp_path / 'default' / name).read_text(), name

    @pytest.mark.skipif(shutil.which('c++') is None, reason='needs a C++ compiler')
    def test_the_files_link_into_one_program(self, out):
        calls = ''.join(f'  {name}(&x);\n' for name in ROUTINES)
        (out / 'main.cpp').write_text(
            f'#include "subroutine.h"\nint main() {{\n  double x = 0.0;\n{calls}'
            f'  return x == {sum(routine.size for routine in ROUTINES.values())}.0 ? 0 : 1;\n}}\n')
        sources = [path for paths in GlobalRoutineCache.sources(str(out), SHARDS).values() for path in paths]
        built = subprocess.run(['c++', '-std=c++17', '-Wall', '-Werror', str(out / 'main.cpp'), *sources,
                                '-o', str(out / 'linked')], capture_output=True, text=True)
        assert built.returncode == 0, built.stderr
        assert subprocess.run([str(out / 'linked')]).returncode == 0
