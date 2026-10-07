"""The routines of several generators, written once for all of them.

A GlobalRoutineCache collects the routines the kernels call -- those of GEMM
tools and of device code generators -- and writes each of them once, with its
declaration in `subroutine.h`. What a process generated can be handed to the
process that writes the routines, as plain data (`export`, `merge`).
"""
from __future__ import annotations

import json
import re

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
