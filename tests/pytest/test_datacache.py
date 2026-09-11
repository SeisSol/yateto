"""
Tests for ``yateto.codegen.datacache`` - the registry of constant arrays.

The cache is what lets the same matrix, requested by two kernels in the same
layout, end up in memory once.  The properties worth pinning down are that
identical requests collapse, that differing ones do not, and that the name a
caller gets back keeps pointing at its own data.
"""
from __future__ import annotations

import pytest

from yateto.codegen.datacache import DataCache


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
