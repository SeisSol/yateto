"""
Tests for ``yateto.generator`` - the orchestrator that ties everything
together.

``Generator`` is the user-facing top-level object.  Users call
``g.add("mykernel", C['ij'] <= A['ik'] * B['kj'])`` to register kernels
and ``g.generate(outdir)`` to spit out C++.  Between those two,
``Kernel`` objects carry the ASTs and run them through
``prepareUntilUnitTest`` and ``prepareUntilCodeGen``.

These tests exercise the Python side of that pipeline - stopping
**before** any C++ emission - so they can run without a compiler
toolchain.  The C++ generation side is tested by the GitHub Actions
``yateto-cpu.yml`` workflow.
"""
from __future__ import annotations

import os
import tempfile

import pytest

from yateto import Tensor, Generator, simpleParameterSpace, parameterSpaceFromRanges
from yateto.generator import Kernel, KernelFamily


# ---------------------------------------------------------------------------
# Kernel name validation
# ---------------------------------------------------------------------------


class TestKernelNames:
    @pytest.mark.parametrize(
        "name",
        ["matmul", "K1", "my_kernel", "foo0", "A"],
    )
    def test_valid_names(self, name):
        assert Kernel.isValidName(name)

    @pytest.mark.parametrize(
        "name",
        ["0foo", "foo(1)", "foo-bar", ""],
    )
    def test_invalid_names(self, name):
        assert not Kernel.isValidName(name)


class TestKernelFamilyNames:
    @pytest.mark.parametrize(
        "name",
        ["family(0)", "family(1)", "kfam(12)"],
    )
    def test_valid_family_names(self, name):
        # A KernelFamily name must be ``<base>(<nat>)``.
        assert KernelFamily.isValidName(name)

    @pytest.mark.parametrize(
        "name",
        ["family", "family()", "family(0,1)"],
    )
    def test_invalid_family_names(self, name):
        # Note: ``family(0,1)`` is a *tensor* group name, not a family
        # name.  Family names have exactly one parenthesised nat.
        assert not KernelFamily.isValidName(name)


# ---------------------------------------------------------------------------
# Kernel construction & prepareUntilUnitTest
# ---------------------------------------------------------------------------


class TestKernelPreparation:
    def test_construction_stores_ast_as_list(self):
        A = Tensor("A", (4, 4))
        B = Tensor("B", (4, 4))
        C = Tensor("C", (4, 4))
        kernel = Kernel("k", C["ij"] <= A["ik"] * B["kj"])
        # Internally the AST is always stored as a list (even for a
        # single-statement kernel).
        assert isinstance(kernel.ast, list)
        assert len(kernel.ast) == 1

    def test_construction_accepts_list_of_asts(self):
        A = Tensor("A", (4, 4))
        B = Tensor("B", (4, 4))
        C = Tensor("C", (4, 4))
        ast_list = [
            C["ij"] <= A["ij"],
            B["ij"] <= C["ij"],
        ]
        kernel = Kernel("k", ast_list)
        assert len(kernel.ast) == 2

    def test_default_target_is_cpu(self):
        A = Tensor("A", (2, 2))
        kernel = Kernel("k", A["ij"] <= A["ij"])
        assert kernel.target == "cpu"

    def test_rejects_invalid_target(self):
        A = Tensor("A", (2, 2))
        with pytest.raises(ValueError, match="target platform"):
            Kernel("k", A["ij"] <= A["ij"], target="fpga")

    def test_prepareUntilUnitTest_populates_cfg(self, arch):
        A = Tensor("A", (4, 4))
        B = Tensor("B", (4, 4))
        C = Tensor("C", (4, 4))
        kernel = Kernel("k", C["ij"] <= A["ik"] * B["kj"])
        assert kernel.cfg is None
        kernel.prepareUntilUnitTest()
        # After prepare, cfg is populated and each ProgramPoint has a
        # live set (LivenessAnalysis has run).
        assert kernel.cfg is not None
        assert all(pp.live is not None for pp in kernel.cfg)

    def test_prepareUntilCodeGen_populates_nonzero_flops(self, arch):
        from yateto.ast.cost import BoundingBoxCostEstimator
        A = Tensor("A", (8, 8))
        B = Tensor("B", (8, 8))
        C = Tensor("C", (8, 8))
        kernel = Kernel("k", C["ij"] <= A["ik"] * B["kj"])
        kernel.prepareUntilUnitTest()
        kernel.prepareUntilCodeGen(BoundingBoxCostEstimator, enableFusedGemm=False)
        # The exact flop count for a dense 8x8 matmul is 960 - same
        # value we pinned down in test_ast_visitor.
        assert kernel.nonZeroFlops == 960


# ---------------------------------------------------------------------------
# Prefetch argument
# ---------------------------------------------------------------------------


class TestKernelPrefetch:
    def test_accepts_tensor(self):
        A = Tensor("A", (4, 4))
        P = Tensor("P", (4, 4))
        kernel = Kernel("k", A["ij"] <= A["ij"], prefetch=P)
        # _prefetch is stored as a list internally.
        assert kernel._prefetch == [P]

    def test_accepts_list_of_tensors(self):
        A = Tensor("A", (4, 4))
        P1 = Tensor("P1", (4, 4))
        P2 = Tensor("P2", (4, 4))
        kernel = Kernel("k", A["ij"] <= A["ij"], prefetch=[P1, P2])
        assert kernel._prefetch == [P1, P2]

    def test_rejects_invalid_prefetch(self):
        A = Tensor("A", (4, 4))
        with pytest.raises(ValueError, match="Prefetch must"):
            Kernel("k", A["ij"] <= A["ij"], prefetch="some string")


# ---------------------------------------------------------------------------
# Parameter spaces (helpers)
# ---------------------------------------------------------------------------


class TestParameterSpace:
    def test_simple_parameter_space(self):
        # simpleParameterSpace(a, b) -> cartesian product range(a) x range(b)
        ps = simpleParameterSpace(2, 3)
        assert len(ps) == 6
        assert (0, 0) in ps
        assert (1, 2) in ps
        assert (1, 3) not in ps

    def test_parameter_space_from_ranges(self):
        ps = parameterSpaceFromRanges([0, 2], [1, 3])
        assert sorted(ps) == [(0, 1), (0, 3), (2, 1), (2, 3)]


# ---------------------------------------------------------------------------
# Generator - registration
# ---------------------------------------------------------------------------


class TestGeneratorRegistration:
    def test_add_kernel(self, arch):
        A = Tensor("A", (4, 4))
        B = Tensor("B", (4, 4))
        C = Tensor("C", (4, 4))
        g = Generator(arch)
        g.add("matmul", C["ij"] <= A["ik"] * B["kj"])
        assert len(g.kernels()) == 1
        assert g.kernels()[0].name == "matmul"

    def test_add_family_member_creates_family(self, arch):
        A = Tensor("A", (4, 4))
        B = Tensor("B", (4, 4))
        C = Tensor("C", (4, 4))
        g = Generator(arch)
        # "foo(0)" is a family-indexed kernel; the generator should detect
        # that and create a ``KernelFamily`` rather than a single Kernel.
        g.add("foo(0)", C["ij"] <= A["ij"] + B["ij"])
        g.add("foo(1)", C["ij"] <= A["ij"] - B["ij"])
        # foo(0) and foo(1) both belong to the "foo" family.
        assert len(g.kernels()) == 2
        # Family dispatches by internal renaming: we don't see "foo(0)"
        # as a top-level kernel.
        assert not any(k.name == "foo(0)" for k in g.kernels())

    def test_add_rejects_invalid_kernel_name(self, arch):
        A = Tensor("A", (4, 4))
        g = Generator(arch)
        with pytest.raises(ValueError, match="Kernel name invalid"):
            g.add("0bad", A["ij"] <= A["ij"])

    def test_arch_is_attached(self, arch):
        g = Generator(arch)
        assert g.arch() is arch


# ---------------------------------------------------------------------------
# KernelFamily
# ---------------------------------------------------------------------------


class TestKernelFamily:
    def test_linear_dispatch_math(self):
        # The family's linear index formula: index = sum_i p_i * stride_i
        # Used to map a multi-index group to a single numeric id.
        stride = (1, 3, 9)
        assert KernelFamily.linear(stride, (0, 0, 0)) == 0
        assert KernelFamily.linear(stride, (1, 0, 0)) == 1
        assert KernelFamily.linear(stride, (0, 1, 0)) == 3
        assert KernelFamily.linear(stride, (1, 1, 1)) == 1 + 3 + 9

    def test_family_base_name(self):
        assert KernelFamily.baseName("foo(3)") == "foo"

    def test_family_group_extraction(self):
        assert KernelFamily.group("foo(5)") == 5

    def test_stride_default(self):
        f = KernelFamily()
        assert f.stride() == (1,)


# ---------------------------------------------------------------------------
# Generator.generate smoke test
# ---------------------------------------------------------------------------


class TestGeneratorGenerateSmoke:
    """``Generator.generate`` emits C++ files.  We don't compile them
    here, but the call should run end-to-end without error on a simple
    kernel and the expected output files must materialise.
    """

    def test_generate_writes_expected_files(self, arch, tmp_path):
        A = Tensor("A", (8, 8))
        B = Tensor("B", (8, 8))
        C = Tensor("C", (8, 8))
        g = Generator(arch)
        g.add("matmul", C["ij"] <= A["ik"] * B["kj"])
        g.generate(str(tmp_path))

        # These are the canonical files emitted by the generator.
        # (The exact list is documented in ``Generator.FileNames``.)
        expected = {
            "tensor.h",
            "tensor.cpp",
            "init.h",
            "init.cpp",
            "kernel.h",
            "kernel.cpp",
        }
        present = set(os.listdir(str(tmp_path)))
        missing = expected - present
        assert not missing, f"missing files: {missing}"

    def test_generate_writes_kernel_class(self, arch, tmp_path):
        # Regression: the generated ``kernel.h`` must declare a struct
        # named like our kernel.  This catches gross breakage in the
        # codegen without requiring a C++ compiler.
        A = Tensor("A", (8, 8))
        B = Tensor("B", (8, 8))
        C = Tensor("C", (8, 8))
        g = Generator(arch)
        g.add("matmul", C["ij"] <= A["ik"] * B["kj"])
        g.generate(str(tmp_path))

        kernel_h = (tmp_path / "kernel.h").read_text()
        # The generator emits a ``struct matmul`` in a ``namespace kernel``.
        assert "matmul" in kernel_h
        assert "namespace kernel" in kernel_h or "kernel::" in kernel_h
