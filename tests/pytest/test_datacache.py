"""
Tests for ``yateto.codegen.datacache`` - the registry of constant arrays.

The cache is what lets the same matrix, requested by two kernels in the same
layout, end up in memory once.  The properties worth pinning down are that
identical requests collapse, that differing ones do not, and that the name a
caller gets back keeps pointing at its own data.
"""
from __future__ import annotations

import collections
import pathlib
import shutil
import subprocess
from io import StringIO

import re

import numpy as np
import pytest

from yateto import Tensor, useArchitectureIdentifiedBy
from yateto.codegen.code import Cpp
from yateto.codegen.arrangement import Arrangement
from yateto.codegen.datacache import DataCache
from yateto.memory import CSCMemoryLayout, DenseMemoryLayout
from yateto.codegen.visitor import POOL_ALIGNMENT, InitializerGenerator, PoolGenerator
from yateto.type import Datatype


@pytest.fixture
def cache():
    return DataCache()


def test_identical_requests_collapse(cache):
    first = cache.add('rDivM', [1.0, 2.0, 0.0], 'double')
    second = cache.add('rDivM', [1.0, 2.0, 0.0], 'double')

    assert first == second
    assert len(cache) == 1


def test_same_values_from_different_callers_collapse(cache):
    """The hint shapes the name but must not take part in the decision."""
    first = cache.add('fMrT', [1.0, 2.0], 'double')
    second = cache.add('someOtherTensor', [1.0, 2.0], 'double')

    assert first == second
    assert len(cache) == 1
    assert first.startswith('fMrT_')


def test_differing_values_stay_apart(cache):
    first = cache.add('rDivM', [1.0, 2.0], 'double')
    second = cache.add('rDivM', [1.0, 3.0], 'double')

    assert first != second
    assert len(cache) == 2


def test_differing_element_type_stays_apart(cache):
    """Same numbers, different type: different bytes, hence different entries."""
    first = cache.add('pattern', [1, 2], 'double')
    second = cache.add('pattern', [1, 2], 'unsigned')

    assert first != second
    assert len(cache) == 2


def test_order_is_not_padding(cache):
    """Padding sits at a position, so moving it changes the image."""
    first = cache.add('m', [1.0, 0.0], 'double')
    second = cache.add('m', [0.0, 1.0], 'double')

    assert first != second


def test_shared_entry_takes_the_stricter_alignment(cache):
    cache.add('rDivM', [1.0], 'double', alignment=16)
    cache.add('rDivM', [1.0], 'double', alignment=128)

    entry, = cache.entries()
    assert entry.alignment() == 128


def test_alignment_does_not_split_entries(cache):
    first = cache.add('rDivM', [1.0], 'double', alignment=16)
    second = cache.add('rDivM', [1.0], 'double', alignment=128)

    assert first == second
    assert len(cache) == 1


def test_entries_keep_registration_order(cache):
    cache.add('a', [1.0], 'double')
    cache.add('b', [2.0], 'double')
    cache.add('c', [3.0], 'double')

    assert [entry.hint() for entry in cache.entries()] == ['a', 'b', 'c']


def test_entry_carries_what_the_emitter_needs(cache):
    name = cache.add('rDivM', [1.0, 2.0, 3.0], 'double', alignment=128)

    entry, = cache.entries()
    assert entry.name() == name
    assert entry.values() == [1.0, 2.0, 3.0]
    assert entry.elements() == 3
    assert entry.typename() == 'double'
    assert entry.alignment() == 128


def test_names_are_valid_cxx_identifiers(cache):
    name = cache.add('nodal::rDivM', [1.0], 'double')

    assert '::' not in name
    assert name.replace('_', 'x').isalnum()


class TestCollectPool:
    """``collectPool`` decides what goes into the image and under which type."""

    @staticmethod
    def _collect(tensors):
        arch = useArchitectureIdentifiedBy('dhsw')
        generator = InitializerGenerator(arch, tensors, [])
        cache = DataCache()
        return generator.collectPool(cache), cache

    def test_entry_takes_the_tensor_datatype(self):
        """Not the architecture's: init binds a reference of the tensor's type."""
        tensors = [
            Tensor('A', (2, 2), np.eye(2)),
            Tensor('P', (2, 2), np.eye(2), datatype=Datatype.F32),
        ]
        _, cache = self._collect(tensors)

        assert {entry.typename() for entry in cache.entries()} == {'double', 'float'}

    def test_same_numbers_in_two_types_stay_apart(self):
        tensors = [
            Tensor('A', (2, 2), np.eye(2)),
            Tensor('P', (2, 2), np.eye(2), datatype=Datatype.F32),
        ]
        _, cache = self._collect(tensors)

        assert len(cache) == 2

    def test_values_are_rendered_for_their_type(self):
        tensors = [Tensor('P', (2, 2), np.eye(2), datatype=Datatype.F32)]
        _, cache = self._collect(tensors)

        entry, = cache.entries()
        assert all(value.endswith('f') for value in entry.values())

    def test_tensors_without_values_are_absent(self):
        pool, cache = self._collect([Tensor('A', (2, 2))])

        assert pool == {}
        assert len(cache) == 0

    def test_group_reports_its_datatype(self):
        tensors = [Tensor('F({})'.format(i), (2, 2), np.eye(2) * (i + 1)) for i in range(2)]
        pool, _ = self._collect(tensors)

        entry = next(e for e in pool.values() if e.baseName == 'F')
        assert entry.datatype == Datatype.F64
        assert entry.groupSize == (2,)
        assert len(entry.symbols) == 2

    def test_one_arrangement_is_one_entry(self):
        tensors = [Tensor('F({})'.format(i), (2, 2), np.eye(2) * (i + 1)) for i in range(2)]
        pool, _ = self._collect(tensors)

        arrangement = Arrangement({tensor.group(): tensor.memoryLayout()
                                   for tensor in tensors})
        assert [key for key in pool] == [('F', arrangement.tag())]

    def test_members_laid_out_differently_are_one_entry(self):
        """A family is one table of pointers, and each of them has its own array."""
        tensors = [Tensor('F(0)', (2, 2), np.eye(2)),
                   Tensor('F(1)', (4, 4), np.eye(4))]
        pool, cache = self._collect(tensors)

        entry, = pool.values()
        assert len(entry.symbols) == 2
        assert sorted(e.elements() for e in cache.entries()) == [4, 16]

    def test_mixed_datatypes_in_one_group_are_rejected(self):
        tensors = [
            Tensor('F(0)', (2, 2), np.eye(2)),
            Tensor('F(1)', (2, 2), np.eye(2), datatype=Datatype.F32),
        ]
        with pytest.raises(ValueError, match='Mixed datatypes'):
            self._collect(tensors)


class TestAFamilyIsArrangedAsAWhole:
    """A family is one table of pointers, so it is held one way, not per member.

    Its members need not agree on how each of them is laid out -- each has an
    array of its own -- and a kernel that reads one of them still speaks about
    the whole family, because that is what it is handed.
    """

    @staticmethod
    def _generate(tmp_path, reads):
        from yateto import Generator
        from yateto.gemm_configuration import GeneratorCollection

        arch = useArchitectureIdentifiedBy('dhsw')
        # Two members of one family, of different heights, so that they are
        # laid out differently and cannot be held in one shape.
        heights = {0: 9, 1: 5}
        F = {i: Tensor('F({})'.format(i), (rows, 3), np.ones((rows, 3)),
                       alignStride=True)
             for i, rows in heights.items()}
        B = Tensor('B', (3, 3))
        g = Generator(arch)
        for i in reads:
            C = Tensor('C{}'.format(i), (heights[i], 3))
            g.add('krnl{}'.format(i), C['ij'] <= F[i]['ik'] * B['kj'])
        g.generate(str(tmp_path), gemm_cfg=GeneratorCollection([]),
                   include_tensors=set(F.values()))
        return ((tmp_path / 'pool.h').read_text(),
                (tmp_path / 'kernel.h').read_text())

    def test_members_laid_out_differently_are_one_entry(self, tmp_path):
        pool_h, _ = self._generate(tmp_path, reads=(0, 1))

        members = re.findall(r'Container<double const\*> (F_\w+)', pool_h)
        assert len(members) == 1
        assert re.search(r'double const F_0_\w+\[36\]', pool_h)
        assert re.search(r'double const F_1_\w+\[24\]', pool_h)

    def test_reading_one_member_names_the_whole_family(self, tmp_path):
        """Two kernels, one member each, still bind the same pool member."""
        _, kernel_h = self._generate(tmp_path, reads=(0, 1))

        bound = set(re.findall(r'F = pool\.(\w+);', kernel_h))
        assert len(bound) == 1

    def test_a_partly_read_family_is_held_whole(self, tmp_path):
        """The unread member is in the table too; the caller is handed all of it."""
        pool_h, _ = self._generate(tmp_path, reads=(0,))

        assert re.search(r'double const F_0_\w+\[36\]', pool_h)
        assert re.search(r'double const F_1_\w+\[24\]', pool_h)


class TestArrangementIdentity:
    """A tag says how a family is held, so it covers every member of it."""

    @staticmethod
    def _layout(rows, alignStride=True):
        from yateto.memory import DenseMemoryLayout
        values = np.ones((rows, 3))
        return DenseMemoryLayout.fromSpp(
            Tensor('t', (rows, 3), values).spp(), alignStride=alignStride)

    def test_the_same_members_are_the_same_arrangement(self):
        first = Arrangement({(0,): self._layout(9), (1,): self._layout(5)})
        second = Arrangement({(1,): self._layout(5), (0,): self._layout(9)})

        assert first == second
        assert first.tag() == second.tag()

    def test_one_member_held_differently_is_another_arrangement(self):
        first = Arrangement({(0,): self._layout(9), (1,): self._layout(5)})
        second = Arrangement({(0,): self._layout(9),
                              (1,): self._layout(5, alignStride=False)})

        assert first != second

    def test_a_family_that_gains_a_member_is_held_differently(self):
        one = Arrangement({(0,): self._layout(9)})
        two = Arrangement({(0,): self._layout(9), (1,): self._layout(5)})

        assert one != two


class TestPoolMembers:
    """Pool members are flat, so two tensor names can meet in one identifier."""

    def test_namespaces_are_flattened(self):
        assert PoolGenerator.memberName('nodal::rDivM') == 'nodal_rDivM'

    def test_each_tensor_gets_its_own_member(self):
        members = PoolGenerator.assignMembers(['nodal::rDivM', 'fMrT'])

        assert members == {'nodal::rDivM': 'nodal_rDivM', 'fMrT': 'fMrT'}

    def test_a_collision_is_refused(self):
        """`a::b` and `a_b` flatten alike; one would write into the other."""
        with pytest.raises(ValueError, match='a_b'):
            PoolGenerator.assignMembers(['a::b', 'a_b'])


class TestPoolAlignment:
    """Every entry meets the pool's floor, whatever its own layout asks for."""

    @staticmethod
    def _entries(tensor):
        arch = useArchitectureIdentifiedBy('dhsw')
        cache = DataCache()
        InitializerGenerator(arch, [tensor], []).collectPool(cache)
        return cache.entries()

    def test_an_unaligned_layout_still_meets_the_floor(self):
        entry, = self._entries(Tensor('v', (3, 3), np.eye(3)))

        assert entry.alignment() >= POOL_ALIGNMENT

    def test_an_aligned_layout_meets_it_too(self):
        entry, = self._entries(Tensor('m', (8, 8), np.eye(8), alignStride=True))

        assert entry.alignment() >= POOL_ALIGNMENT

    def test_a_wider_cache_line_wins_over_the_floor(self):
        """a64fx reads 256-byte lines; the floor must not cap that."""
        arch = useArchitectureIdentifiedBy('da64fx')
        cache = DataCache()
        tensor = Tensor('m', (8, 8), np.eye(8), alignStride=True)
        InitializerGenerator(arch, [tensor], []).collectPool(cache)

        entry, = cache.entries()
        assert entry.alignment() == max(POOL_ALIGNMENT, arch.cacheline)


class TestImageAlignment:
    """The image is declared with the alignment it actually has."""

    @staticmethod
    def _generator(archName, tensors):
        arch = useArchitectureIdentifiedBy(archName)
        cache = DataCache()
        pool = InitializerGenerator(arch, tensors, []).collectPool(cache)
        return PoolGenerator(arch, cache, pool), cache

    def test_the_floor_holds_with_nothing_in_the_image(self):
        generator, _ = self._generator('dhsw', [])

        assert generator.imageAlignment() == POOL_ALIGNMENT

    def test_the_floor_holds_when_no_entry_asks_for_more(self):
        generator, _ = self._generator('dhsw', [Tensor('v', (3, 3), np.eye(3))])

        assert generator.imageAlignment() == POOL_ALIGNMENT

    def test_the_strictest_entry_raises_the_image(self):
        """Declaring less than a member asks for would say 128 and mean 256."""
        tensor = Tensor('m', (8, 8), np.eye(8), alignStride=True)
        generator, cache = self._generator('da64fx', [tensor])

        entry, = cache.entries()
        assert entry.alignment() > POOL_ALIGNMENT
        assert generator.imageAlignment() == entry.alignment()


class TestReservations:
    """An entry can be announced before there is anything to put in it.

    A generator that decides how it wants an operand arranged while a kernel
    is being written out has a symbol to emit and no image yet. It reserves,
    keeps writing, and fills once it produces the values.
    """

    def test_a_reservation_names_itself_immediately(self, cache):
        reservation = cache.reserve('kDivM')

        assert reservation.name()
        assert not reservation.isFilled()

    def test_the_symbol_survives_being_filled(self, cache):
        reservation = cache.reserve('kDivM')
        name = reservation.name()
        cache.fill(reservation, [1.0, 2.0], 'double')

        assert reservation.name() == name
        assert reservation.entry().values() == [1.0, 2.0]

    def test_two_reservations_of_one_hint_stay_apart(self, cache):
        first = cache.reserve('kDivM')
        second = cache.reserve('kDivM')

        assert first.name() != second.name()

    def test_equal_images_share_one_array_under_two_symbols(self, cache):
        first = cache.reserve('kDivM')
        second = cache.reserve('kDivMT')
        cache.fill(first, [1.0, 2.0], 'double')
        cache.fill(second, [1.0, 2.0], 'double')

        assert len(cache) == 1
        assert first.entry() is second.entry()
        assert first.name() != second.name()

    def test_differing_images_stay_apart(self, cache):
        first = cache.reserve('kDivM')
        second = cache.reserve('kDivMT')
        cache.fill(first, [1.0, 2.0], 'double')
        cache.fill(second, [1.0, 3.0], 'double')

        assert len(cache) == 2
        assert first.entry() is not second.entry()

    def test_a_reservation_lands_on_an_array_a_tensor_already_asked_for(self, cache):
        added = cache.add('rDivM', [1.0, 2.0], 'double')
        reservation = cache.reserve('mine')
        cache.fill(reservation, [1.0, 2.0], 'double')

        assert len(cache) == 1
        assert reservation.entry().name() == added

    def test_filling_twice_is_refused(self, cache):
        reservation = cache.reserve('kDivM')
        cache.fill(reservation, [1.0], 'double')

        with pytest.raises(RuntimeError, match='twice'):
            cache.fill(reservation, [2.0], 'double')

    def test_an_unfilled_reservation_is_refused(self, cache):
        cache.reserve('kDivM')

        with pytest.raises(RuntimeError, match='kDivM'):
            cache.reservations()

    def test_reservations_keep_their_order(self, cache):
        names = [cache.reserve(hint).name() for hint in ('a', 'b', 'c')]
        for hint, name in zip(('a', 'b', 'c'), names):
            cache.fill(cache._reservations[name], [1.0, float(ord(hint))], 'double')

        assert [r.name() for r in cache.reservations()] == names

    def test_names_are_valid_cxx_identifiers(self, cache):
        reservation = cache.reserve('nodal::rDivM(0)')

        assert reservation.name().replace('_', 'x').isalnum()


class TestReservationsInTheEmittedPool:
    """A reserved entry is a pool member like any other."""

    @staticmethod
    def _pool(reservationHints=()):
        arch = useArchitectureIdentifiedBy('dhsw')
        cache = DataCache()
        tensor = Tensor('m', (8, 8), np.eye(8), alignStride=True)
        initGen = InitializerGenerator(arch, [tensor], [])
        poolMap = initGen.collectPool(cache)
        for i, hint in enumerate(reservationHints):
            reservation = cache.reserve(hint)
            cache.fill(reservation, [float(i)] * 4, 'float')
        generator = PoolGenerator(arch, cache, poolMap)

        headerIO, cppIO = StringIO(), StringIO()
        with Cpp(headerIO) as header:
            generator.generateH(header)
            headerText = headerIO.getvalue()
        with Cpp(cppIO) as cpp:
            generator.generateCpp(cpp)
            cppText = cppIO.getvalue()
        return headerText, cppText

    def test_the_member_is_declared(self):
        header, _ = self._pool(['scratch'])

        assert 'float const* scratch{};' in header

    def test_the_member_points_into_the_image(self):
        _, cpp = self._pool(['scratch'])

        assert 'result.scratch = reinterpret_cast<float const*>(' in cpp
        assert 'offsetof(poolstorage::Storage, scratch_' in cpp

    def test_a_shared_array_is_stored_once_and_pointed_at_twice(self):
        header, cpp = self._pool(['first', 'second'])

        assert 'float const* first{};' in header
        assert 'float const* second{};' in header
        assert cpp.count('result.first = ') == 1
        assert cpp.count('result.second = ') == 1

    def test_a_tensor_member_carries_its_arrangement(self):
        header, _ = self._pool()

        assert 'const* m{};' not in header
        assert any(line.strip().startswith('double const* m_') and line.strip().endswith('{};')
                   for line in header.splitlines())


class TestReservingWhileKernelsAreWritten:
    """The cache is reachable from the factories, not only afterwards."""

    def test_a_factory_can_put_an_array_into_the_pool(self, tmp_path, monkeypatch):
        import numpy as np

        from yateto import Generator
        from yateto.codegen.factory import OptimizedKernelFactory

        original = OptimizedKernelFactory.post_generate

        def post_generate(self, routine_cache):
            reservation = self._dataCache.reserve('chosenLayout')
            self._dataCache.fill(reservation, [1.0, 2.0, 3.0, 4.0], 'double')
            return original(self, routine_cache)

        monkeypatch.setattr(OptimizedKernelFactory, 'post_generate', post_generate)

        arch = useArchitectureIdentifiedBy('dhsw')
        g = Generator(arch)
        A = Tensor('A', (4, 4), np.eye(4))
        B = Tensor('B', (4, 4))
        C = Tensor('C', (4, 4))
        g.add('krnl', C['ij'] <= A['ik'] * B['kj'])
        g.generate(str(tmp_path))

        pool_h = (tmp_path / 'pool.h').read_text()
        pool_cpp = (tmp_path / 'pool.cpp').read_text()

        assert 'double const* chosenLayout{};' in pool_h
        assert 'result.chosenLayout = reinterpret_cast<double const*>(' in pool_cpp
        assert '1.0, 2.0, 3.0, 4.0' in pool_cpp


class TestViewArrayPool:
    """The index arrays the views need are spelled once and referred to.

    They have to stay constant expressions: a lookup with constant indices
    folds to a single address only while the compiler can see the pattern.
    """

    @staticmethod
    def _initH(tensors):
        arch = useArchitectureIdentifiedBy('dhsw')
        gen = InitializerGenerator(arch, tensors, [])
        out = StringIO()
        with Cpp(out) as header:
            gen.generateInitH(header)
            return out.getvalue()

    def test_a_shared_pattern_is_spelled_once(self):
        spp = np.zeros((4, 4))
        spp[0, 0] = spp[1, 1] = spp[3, 2] = 1.0
        a = Tensor('a', (4, 4), spp, CSCMemoryLayout)
        b = Tensor('b', (4, 4), spp, CSCMemoryLayout)

        header = self._initH([a, b])

        rows = [line for line in header.splitlines() if 'RowInd_' in line and '= {' in line]
        assert len(rows) == 1
        assert header.count('(&RowInd)[3] = viewdata::') == 2

    def test_differing_patterns_stay_apart(self):
        first = np.zeros((4, 4))
        first[0, 0] = first[1, 1] = 1.0
        second = np.zeros((4, 4))
        second[0, 0] = second[2, 1] = 1.0

        header = self._initH([Tensor('a', (4, 4), first, CSCMemoryLayout),
                              Tensor('b', (4, 4), second, CSCMemoryLayout)])

        rows = [line for line in header.splitlines() if 'RowInd_' in line and '= {' in line]
        assert len(rows) == 2

    def test_the_arrays_stand_before_the_structs_that_name_them(self):
        spp = np.zeros((4, 4))
        spp[0, 0] = spp[1, 1] = 1.0

        header = self._initH([Tensor('a', (4, 4), spp, CSCMemoryLayout)])

        assert header.index('struct viewdata') < header.index('(&RowInd)')

    def test_the_references_are_constant_expressions(self):
        spp = np.zeros((4, 4))
        spp[0, 0] = spp[1, 1] = 1.0

        header = self._initH([Tensor('a', (4, 4), spp, CSCMemoryLayout)])

        for line in header.splitlines():
            if '(&RowInd)' in line or 'RowInd_' in line:
                assert line.strip().startswith('constexpr static')

    def test_dense_bounds_are_shared_too(self):
        header = self._initH([Tensor('a', (4, 4)), Tensor('b', (4, 4))])

        starts = [line for line in header.splitlines() if 'Start_' in line and '= {' in line]
        assert len(starts) == 1
        assert header.count('(&Start)[2] = viewdata::') == 2

    def test_a_pool_with_nothing_in_it_emits_nothing(self):
        header = self._initH([])

        assert 'viewdata' not in header


class TestTwoArrangementsOfOneTensor:
    """A matrix read two ways is two arrays, and two members pointing at them."""

    @staticmethod
    def _arrangements(tensor, archNames):
        from yateto.arch import getArchitectureIdentifiedBy
        from yateto.memory import DenseMemoryLayout

        out = collections.OrderedDict()
        for name in archNames:
            arch = getArchitectureIdentifiedBy(name)
            layout = DenseMemoryLayout.fromSpp(tensor.spp(), alignStride=True,
                                               alignmentArch=arch)
            arrangement = Arrangement({tensor.group(): layout})
            out[arrangement.tag()] = arrangement
        return out

    def _collect(self, archNames):
        arch = useArchitectureIdentifiedBy('dhsw')
        cache = DataCache()
        # Ten rows: padded to twelve on one of the two and to sixteen on the
        # other, so the two arrangements really are different sequences.
        values = np.zeros((10, 3))
        values[0, 0] = values[3, 1] = values[9, 2] = 1.0
        tensor = Tensor('m', (10, 3), values, alignStride=True)
        gen = InitializerGenerator(arch, [tensor], [])
        pool = gen.collectPool(cache, {'m': self._arrangements(tensor, archNames)})
        return cache, pool

    def test_each_arrangement_gets_its_own_entry(self):
        cache, pool = self._collect(['dhsw', 'dskx'])

        assert len(pool) == 2
        assert {entry.baseName for entry in pool.values()} == {'m'}
        assert len(cache) == 2

    def test_the_entries_are_different_lengths(self):
        cache, _ = self._collect(['dhsw', 'dskx'])

        lengths = sorted(entry.elements() for entry in cache.entries())
        assert lengths[0] != lengths[1]

    def test_each_arrangement_gets_its_own_member(self):
        _, pool = self._collect(['dhsw', 'dskx'])

        members = PoolGenerator.assignMembers(pool)

        assert len(set(members.values())) == 2
        assert all(member.startswith('m_') for member in members.values())

    def test_one_arrangement_is_one_entry_and_one_member(self):
        cache, pool = self._collect(['dhsw'])

        assert len(pool) == 1
        assert len(cache) == 1
        assert len(set(PoolGenerator.assignMembers(pool).values())) == 1


class TestATensorCanBeHeldAtTheHostsWidth:
    """A run configured for a device pads to the device's width.

    That is right for what the device reads. The same matrix read on the host
    carries the padding along, and the host has no use for it: the rows are
    dead, they are loaded, and they are multiplied. Which width a tensor is
    held at is the caller's to say, because it is the caller who knows which
    machine reads it -- and a kernel reads a constant at the width the tensor
    was declared with, so the two cannot drift apart.
    """

    @staticmethod
    def _generate(tmp_path, alignmentArch=None, archNames=('shsw', 'sgfx90a', 'hip')):
        from yateto import Generator
        from yateto.gemm_configuration import GeneratorCollection

        arch = useArchitectureIdentifiedBy(*archNames)
        # 35 rows: padded to 40 at the host's width and to 64 at the device's.
        values = np.zeros((35, 3))
        values[0, 0] = values[17, 1] = values[34, 2] = 1.0
        A = Tensor('A', (35, 3), values, alignStride=True)
        if alignmentArch is not None:
            A.setMemoryLayout(DenseMemoryLayout, alignStride=True,
                              alignmentArch=alignmentArch)
        B = Tensor('B', (3, 3))
        C = Tensor('C', (35, 3))
        g = Generator(arch)
        g.add('krnl', C['ij'] <= A['ik'] * B['kj'])
        g.generate(str(tmp_path), gemm_cfg=GeneratorCollection([]))
        return ((tmp_path / 'pool.h').read_text(),
                (tmp_path / 'kernel.cpp').read_text())

    def test_the_run_s_width_is_what_a_tensor_takes(self, tmp_path):
        pool_h, _ = self._generate(tmp_path)

        assert '[192]' in pool_h
        assert '[120]' not in pool_h

    def test_a_tensor_can_be_declared_at_the_host_s_width(self, tmp_path):
        arch = useArchitectureIdentifiedBy('shsw', 'sgfx90a', 'hip')
        pool_h, _ = self._generate(tmp_path, alignmentArch=arch.hostAlignment)

        assert '[120]' in pool_h
        assert '[192]' not in pool_h

    def test_the_kernel_reads_it_at_the_width_it_is_held_at(self, tmp_path):
        """Whatever the tensor was declared with, the strides follow it."""
        arch = useArchitectureIdentifiedBy('shsw', 'sgfx90a', 'hip')
        _, kernel_cpp = self._generate(tmp_path, alignmentArch=arch.hostAlignment)

        assert '40*k' in kernel_cpp
        assert '64*k' not in kernel_cpp


class TestTheHostsWidth:
    """An architecture says what the host aligns to, apart from the run's width."""

    @staticmethod
    def _arch(archNames):
        return useArchitectureIdentifiedBy(*archNames)

    def test_a_device_run_still_knows_the_host_s_width(self):
        assert self._arch(('shsw', 'sgfx90a', 'hip')).hostAlignment.alignment == 32

    def test_a_host_only_run_aligns_to_the_same_width_either_way(self):
        arch = self._arch(('shsw',))

        assert arch.hostAlignment.alignment == arch.alignment


class TestPoolMemberOfATensor:
    """Where a pool holds a tensor as it lays itself out, named stably.

    The member of `Pool` carries a hash of the arrangement, which is nothing
    code outside the generated kernels can spell. Such code -- a hand-written
    device routine reading a constant from a pool on the device -- reaches it
    through `init::X::PoolMember`, a pointer to that member.
    """

    @staticmethod
    def _generate(tmp_path):
        from yateto import Generator
        from yateto.gemm_configuration import GeneratorCollection

        arch = useArchitectureIdentifiedBy('dhsw')
        A = Tensor('A', (4, 4), np.arange(1.0, 17.0).reshape(4, 4))
        F = {i: Tensor('F({})'.format(i), (4, 4), np.full((4, 4), float(i + 2)))
             for i in range(2)}
        B = Tensor('B', (4, 4))
        C = Tensor('C', (4, 4))
        g = Generator(arch)
        g.add('k', C['ij'] <= A['ik'] * F[0]['kl'] * F[1]['lm'] * B['mj'])
        g.generate(str(tmp_path), gemm_cfg=GeneratorCollection([]))
        return tmp_path

    def test_it_points_at_the_member_that_holds_it(self, tmp_path):
        out = self._generate(tmp_path)
        init_h = (out / 'init.h').read_text()
        pool_h = (out / 'pool.h').read_text()
        found = re.findall(r'constexpr static auto PoolMember = &Pool::((\w+?)_[0-9a-f]{8});', init_h)
        members = {base: member for member, base in found}
        assert set(members) == {'A', 'F'}
        for member in members.values():
            assert re.search(r'\b{}\{{\}};'.format(member), pool_h), member

    @pytest.mark.skipif(shutil.which('c++') is None, reason='needs a C++ compiler')
    def test_it_reaches_the_same_numbers_as_init(self, tmp_path):
        (tmp_path / 'gen').mkdir()
        out = self._generate(tmp_path / 'gen')
        include = pathlib.Path(__file__).resolve().parents[2] / 'include'
        (tmp_path / 'main.cpp').write_text(
            '#include "init.h"\n'
            'int main() {\n'
            '  auto pool = yateto::Pool::host();\n'
            '  if (pool.*yateto::init::A::PoolMember != yateto::init::A::Values) return 1;\n'
            '  if ((pool.*yateto::init::F::PoolMember)(1) != yateto::init::F::Values1) return 2;\n'
            '  if ((pool.*yateto::init::F::PoolMember)(1)[0] != 3.0) return 3;\n'
            '  return 0;\n'
            '}\n')
        sources = [str(tmp_path / 'main.cpp')] + [str(out / name) for name in
                                                  ('pool.cpp', 'init.cpp', 'tensor.cpp')]
        subprocess.run(['c++', '-std=c++17', f'-I{include}', f'-I{out}', *sources,
                        '-o', str(tmp_path / 'poolmember')], check=True,
                       capture_output=True, text=True)
        assert subprocess.run([str(tmp_path / 'poolmember')]).returncode == 0
