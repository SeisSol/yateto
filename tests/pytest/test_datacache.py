"""
Tests for ``yateto.codegen.datacache`` - the registry of constant arrays.

The cache is what lets the same matrix, requested by two kernels in the same
layout, end up in memory once.  The properties worth pinning down are that
identical requests collapse, that differing ones do not, and that the name a
caller gets back keeps pointing at its own data.
"""
from __future__ import annotations

import numpy as np
import pytest

from yateto import Tensor, useArchitectureIdentifiedBy
from yateto.codegen.datacache import DataCache
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

        entry = pool['F']
        assert entry.datatype == Datatype.F64
        assert entry.groupSize == (2,)
        assert len(entry.symbols) == 2

    def test_mixed_datatypes_in_one_group_are_rejected(self):
        tensors = [
            Tensor('F(0)', (2, 2), np.eye(2)),
            Tensor('F(1)', (2, 2), np.eye(2), datatype=Datatype.F32),
        ]
        with pytest.raises(ValueError, match='Mixed datatypes'):
            self._collect(tensors)


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
