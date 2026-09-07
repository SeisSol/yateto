"""
Per-kernel attributes and the one thing they currently switch: the batch
flags.

``flags`` is a mask of elements a batched GPU kernel should skip.  Every
GPU kernel used to carry it -- a member on the kernel struct, a parameter
on every external call, a runtime check on every element -- although
almost none are ever called with one.  It is now something a kernel asks
for at ``Generator.add`` time.

The asymmetry worth keeping in mind while reading these: a kernel that
does *not* ask for the mask has no ``flags`` member at all, rather than an
ignored one.  Assigning to it is then a compile error, which is the only
way a caller that meant to mask elements off finds out that it forgot the
attribute -- an ignored member would silently compute every element.

These tests stop before any external code generator runs (GemmForge and
ChainForge are not installed in this suite), so the emitted *call sites*
are covered indirectly, through ``BatchedOperationsAux.flags_arg``, which
is what those generators put in the argument list.
"""
from __future__ import annotations

import collections
from io import StringIO

import pytest

from yateto import Generator, Tensor, simpleParameterSpace
from yateto.codegen.cache import RoutineCache
from yateto.codegen.code import Cpp
from yateto.codegen.common import BatchedOperationsAux, KernelAttributes
from yateto.codegen.factory import ExportFactory, OptimizedKernelFactory
from yateto.codegen.visitor import OptimizedKernelGenerator


# ---------------------------------------------------------------------------
# KernelAttributes
# ---------------------------------------------------------------------------


class TestKernelAttributes:
    def test_no_attributes_means_no_flags(self):
        assert KernelAttributes().flags is False
        assert KernelAttributes(None).flags is False

    def test_flags_is_what_it_was_set_to(self):
        assert KernelAttributes({"flags": True}).flags is True
        assert KernelAttributes({"flags": False}).flags is False

    def test_unknown_attribute_is_rejected(self):
        # An attribute that does nothing is indistinguishable from a typo in
        # one that would have done something, and the typo is the likelier
        # of the two.
        with pytest.raises(ValueError, match="flagz"):
            KernelAttributes({"flagz": True})

    def test_error_names_what_is_available(self):
        with pytest.raises(ValueError, match="'flags'"):
            KernelAttributes({"nonsense": 1})

    def test_equality_is_by_content(self):
        assert KernelAttributes({"flags": True}) == KernelAttributes({"flags": True})
        assert KernelAttributes({"flags": True}) != KernelAttributes()

    def test_as_dict_does_not_alias_the_input(self):
        source = {"flags": True}
        attrs = KernelAttributes(source)
        attrs.as_dict()["flags"] = False
        source["flags"] = False
        assert attrs.flags is True


# ---------------------------------------------------------------------------
# The path from Generator.add to the kernel
# ---------------------------------------------------------------------------


class TestAttributesReachTheKernel:
    def _tensors(self):
        N = 8
        return Tensor("A", (N, N)), Tensor("B", (N, N)), Tensor("C", (N, N))

    def test_default_is_no_flags(self, arch):
        A, B, C = self._tensors()
        g = Generator(arch)
        g.add("krnl", C["ij"] <= A["ik"] * B["kj"])
        assert g.kernels()[0].attrs.flags is False

    def test_add_forwards_attributes(self, arch):
        A, B, C = self._tensors()
        g = Generator(arch)
        g.add("krnl", C["ij"] <= A["ik"] * B["kj"], target="gpu",
              attrs={"flags": True})
        assert g.kernels()[0].attrs.flags is True

    def test_add_family_forwards_attributes(self, arch):
        A, B, C = self._tensors()
        g = Generator(arch)
        g.addFamily("fam", simpleParameterSpace(2),
                    lambda i: C["ij"] <= A["ik"] * B["kj"],
                    target="gpu", attrs={"flags": True})
        kernels = g.kernels()
        assert len(kernels) == 2
        assert all(kernel.attrs.flags for kernel in kernels)

    def test_typo_is_reported_at_the_add_that_made_it(self, arch):
        A, B, C = self._tensors()
        g = Generator(arch)
        with pytest.raises(ValueError):
            g.add("krnl", C["ij"] <= A["ik"] * B["kj"], attrs={"flag": True})


# ---------------------------------------------------------------------------
# What the attribute changes in the generated C++
# ---------------------------------------------------------------------------


def _outline(attrs, target="gpu"):
    """A minimal KernelOutline: enough to emit a struct, no tensors."""
    return OptimizedKernelGenerator.KernelOutline(
        nonZeroFlops=1, hwFlops=1, inConstBytes=0, inBytes=0, outBytes=0,
        tensors=collections.OrderedDict(), writable={},
        prefetch=collections.OrderedDict(), scalars=collections.OrderedDict(),
        function="  // body\n", tmp_mem_size=0, is_compute_constant_tensors={},
        datatype={}, target=target, attrs=attrs)


def _struct(arch, outlines, familyStride=None):
    headerIO, cppIO = StringIO(), StringIO()
    generator = OptimizedKernelGenerator(arch, RoutineCache(), {})
    with Cpp(cppIO) as cpp:
        with Cpp(headerIO) as header:
            generator.generate(cpp, header, "krnl", outlines, familyStride)
            return headerIO.getvalue()


class TestGeneratedStruct:
    def test_without_the_attribute_there_is_no_member(self, arch):
        assert "flags" not in _struct(arch, [_outline(KernelAttributes())])

    def test_with_the_attribute_the_member_is_there(self, arch):
        header = _struct(arch, [_outline(KernelAttributes({"flags": True}))])
        assert "unsigned *flags = nullptr;" in header

    def test_the_other_batch_members_do_not_depend_on_it(self, arch):
        for attrs in (KernelAttributes(), KernelAttributes({"flags": True})):
            header = _struct(arch, [_outline(attrs)])
            assert "unsigned numElements = 0;" in header
            assert "void *streamPtr" in header

    def test_a_cpu_kernel_has_none_of_it(self, arch):
        header = _struct(arch, [_outline(KernelAttributes(), target="cpu")])
        assert "flags" not in header
        assert "numElements" not in header

    def test_a_family_must_agree(self, arch):
        # One struct carries the family, so one flags member -- or none --
        # has to describe every member of it.
        outlines = [_outline(KernelAttributes({"flags": True})),
                    _outline(KernelAttributes())]
        with pytest.raises(RuntimeError, match="different attributes"):
            _struct(arch, outlines, familyStride=(1,))


class TestResetFlags:
    """``execute()`` clears the batch members it consumed."""

    def _emit(self, arch, attrs):
        out = StringIO()
        with Cpp(out) as cpp:
            OptimizedKernelFactory(cpp, arch, "gpu", attrs).reset_flags()
            return out.getvalue()

    def test_reset_only_where_there_is_a_member(self, arch):
        assert self._emit(arch, KernelAttributes()) .strip() == ""
        assert "flags = nullptr;" in self._emit(
            arch, KernelAttributes({"flags": True}))


class TestExternalCallArgument:
    """External kernels take a flags parameter whether or not it is used."""

    def test_the_member_when_the_kernel_has_one(self):
        assert BatchedOperationsAux.flags_arg(
            KernelAttributes({"flags": True})) == "flags"

    def test_a_literal_null_otherwise(self):
        # GemmForge and ChainForge kernels always take the parameter, so the
        # call site has to pass something, and there is no member to pass.
        assert BatchedOperationsAux.flags_arg(KernelAttributes()) == "nullptr"


# ---------------------------------------------------------------------------
# Routine exporters
# ---------------------------------------------------------------------------


class TestExporterHandover:
    def test_an_exporter_is_told_the_attributes(self, arch):
        seen = {}

        class Exporter:
            INTERFACE_VERSION = 5

            def __init__(self, arch, attrs=None):
                seen["attrs"] = attrs

        ExportFactory.makeFactory(Exporter)(
            Cpp(StringIO()), arch, "gpu", KernelAttributes({"flags": True}))
        assert seen["attrs"] == {"flags": True}

    def test_an_exporter_speaking_an_older_interface_is_refused(self, arch):
        # It would read a description whose per-occurrence bounding box it
        # does not know about, and run every operation over the whole storage
        # instead -- which for an assignment writes over entries the operation
        # was never meant to touch. Nothing about that shows up as an error
        # later, so it has to show up here.
        class OldExporter:
            def __init__(self, arch, attrs=None):
                pass

        with pytest.raises(RuntimeError, match="interface version"):
            ExportFactory.makeFactory(OldExporter)(
                Cpp(StringIO()), arch, "gpu", KernelAttributes())

    def test_an_exporter_speaking_a_newer_interface_is_accepted(self, arch):
        """Fields it knows and this yateto does not send do not appear."""
        class NewExporter:
            INTERFACE_VERSION = 99

            def __init__(self, arch, attrs=None):
                pass

        ExportFactory.makeFactory(NewExporter)(
            Cpp(StringIO()), arch, "gpu", KernelAttributes())

    def test_the_version_is_asked_of_the_exporter_not_of_its_factory(self, arch):
        """A factory function is a fine way to register one, and it carries
        no version of its own."""
        class Exporter:
            INTERFACE_VERSION = 5

            def __init__(self, arch, attrs=None):
                pass

        def make(arch, attrs=None):
            return Exporter(arch, attrs)

        ExportFactory.makeFactory(make)(
            Cpp(StringIO()), arch, "gpu", KernelAttributes())

    def test_an_exporter_without_the_channel_is_refused(self, arch):
        # It would generate a kernel taking flags while this side emits no
        # member to pass, so the failure belongs here and not in a C++
        # compiler two repositories away.
        class OldExporter:
            def __init__(self, arch):
                pass

        with pytest.raises(RuntimeError, match="kernel attributes"):
            ExportFactory.makeFactory(OldExporter)(
                Cpp(StringIO()), arch, "gpu", KernelAttributes())
