"""
Tests for ``tools/kernel_report.py``, which reads what a generated kernel set
costs out of the generated files.

It parses, so what is worth testing is the shapes it has to recognise: a
kernel states its numbers one per line, a family states them per member as an
array, and a family's members each have an ``execute`` of their own on the one
struct they share.
"""
from __future__ import annotations

import importlib.util
import pathlib

import pytest

_TOOL = (pathlib.Path(__file__).resolve().parents[2] / 'tools' / 'kernel_report.py')


@pytest.fixture(scope='module')
def report():
    spec = importlib.util.spec_from_file_location('kernel_report', _TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HEADER = '''
namespace yateto {
  namespace kernel {
    struct plain {
      constexpr static unsigned long const NonZeroFlops = 100;
      constexpr static unsigned long const HardwareFlops = 120;
      constexpr static unsigned long const TmpMemRequiredInBytes = 64;
    };
    struct family {
      constexpr static unsigned long const NonZeroFlops[] = {0, 10, 20};
      constexpr static unsigned long const HardwareFlops[] = {0, 11, 22};
    };
  }
}
'''

SOURCE = '''
namespace yateto {
  void kernel::plain::execute() {
    double* _tmp0;
    memset(_tmp0, 0, 8 * sizeof(double));
    #pragma omp simd collapse(2)
    for (int _b = 0; _b < 2; ++_b) {
      for (int _a = 0; _a < 2; ++_a) {
        _tmp0[1*_a] = 1.0;
      }
    }
  }
  void kernel::family::execute1() {
    #pragma omp simd
    for (int _a = 0; _a < 2; ++_a) {
    }
  }
  void kernel::family::execute2() {
    #pragma omp simd
    for (int _a = 0; _a < 2; ++_a) {
    }
  }
}
'''


class TestReads:
    def test_a_kernel_states_its_numbers_one_per_line(self, report):
        assert report.counters(HEADER)['plain']['HardwareFlops'] == 120
        assert report.counters(HEADER)['plain']['TmpMemRequiredInBytes'] == 64

    def test_a_family_states_them_per_member(self, report):
        assert report.counters(HEADER)['family']['HardwareFlops'] == 33
        assert report.counters(HEADER)['family']['NonZeroFlops'] == 30

    def test_a_family_has_one_execute_per_member(self, report):
        found = report.bodies(SOURCE)
        assert set(found) == {'plain', 'family'}
        assert report.shape(found['family'])['nests'] == 2

    def test_what_a_body_does_is_counted(self, report):
        shape = report.shape(report.bodies(SOURCE)['plain'])
        assert shape['nests'] == 1
        assert shape['loops'] == 2
        assert shape['buffers'] == 1
        assert shape['memsets'] == 1


class TestCompares:
    def test_only_what_moved_is_reported(self, report):
        before = {'a': {'nests': 3, 'buffers': 1}, 'b': {'nests': 1}}
        after = {'a': {'nests': 1, 'buffers': 0}, 'b': {'nests': 1}}
        moved = report.compare(before, after)
        assert [name for name, _ in moved] == ['a']
        assert moved[0][1]['nests'] == (3, 1)

    def test_a_field_may_be_asked_for_on_its_own(self, report):
        before = {'a': {'nests': 3, 'buffers': 1}}
        after = {'a': {'nests': 1, 'buffers': 0}}
        moved = report.compare(before, after, only={'buffers'})
        assert set(moved[0][1]) == {'buffers'}
