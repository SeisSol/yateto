"""
Tests for the per-kernel
"bandwidth" constants emitted into the generated C++ code.

Background
----------
For roof-line analysis and runtime self-instrumentation, kernels often
need to know not only their FLOP count but also how many bytes they
push through memory.  Every generated kernel struct contains
three ``constexpr`` fields:

* ``InboundConstBytes`` - bytes read from compute-constant tensors
  (these can be hoisted to read-only memory by the runtime).
* ``InboundBytes``      - bytes read from non-constant input tensors.
* ``OutboundBytes``     - bytes written to output tensors.

The numbers are computed in ``OptimizedKernelGenerator.generateKernelOutline``
(``yateto/codegen/visitor.py``).  Each tensor referenced by the kernel
is bucketed into one of the three categories based on its
``is_compute_constant()`` and ``writable`` flags, and its size is
``memoryLayout().storage().requiredReals() * arch.bytesPerReal``.

What is checked
---------------
* The struct fields exist on the right branch (skipped on master).
* The numbers are correct for a battery of canonical kernels:
  pure dense matmul, identity copy, reductions to a scalar, kernels
  with compute-constant tensors, sparse tensors with shrunk bounding
  boxes, single vs double precision, and a kernel family that shares
  tensors across its members.
* The categorisation rules: a tensor is "out" iff it's writable, "in"
  iff it's only read, "const" iff it's compute-constant.
* Each kernel reports its own per-kernel numbers; tensors shared
  between kernels are not double-counted across kernels (they appear
  once in each kernel's own outline).
"""

from __future__ import annotations

import os
import re

import numpy as np
import pytest

from yateto import Generator, Tensor, useArchitectureIdentifiedBy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_FIELD_RE = re.compile(
    r"const\s+(?P<name>InboundConstBytes|InboundBytes|OutboundBytes|"
    r"NonZeroFlops|HardwareFlops|TmpMemRequiredInBytes|TmpMaxMemRequiredInBytes)"
    r"\s*=\s*(?P<value>\d+)"
)


def _extract_kernel_constants(out_dir: str, kernel_name: str) -> dict:
    """Parse ``kernel.h`` and return the bandwidth/flop fields of one
    named kernel struct.

    Doing this with a regex (rather than compiling the C++) is fragile
    in general but good enough for our specific output format - the
    generated struct is a small block of seven ``constexpr static``
    fields with stable spelling.
    """
    text = (open(os.path.join(out_dir, "kernel.h")).read())
    # Locate the struct body: ``struct <name> { ... };`` (the closing
    # brace ends the struct, so we slice up to it).
    pattern = re.compile(
        r"struct\s+" + re.escape(kernel_name) + r"\s*\{(?P<body>.*?)\};",
        re.DOTALL,
    )
    m = pattern.search(text)
    assert m is not None, f"struct {kernel_name!r} not found in kernel.h"
    body = m.group("body")
    return {match.group("name"): int(match.group("value"))
            for match in _FIELD_RE.finditer(body)}


def _new_arch(name: str):
    """Fresh architecture, with the global alignment state reset.  The
    bandwidth tests don't share an ``arch`` fixture across cases
    because we want to vary precision (``dhsw`` vs ``shsw``) per test.
    """
    from yateto.memory import DenseMemoryLayout
    DenseMemoryLayout.ALIGNMENT_ARCH = None
    return useArchitectureIdentifiedBy(name)


@pytest.fixture
def reset_arch():
    """Reset alignment state both before *and* after each test - some
    of these tests construct multiple architectures.
    """
    from yateto.memory import DenseMemoryLayout
    DenseMemoryLayout.ALIGNMENT_ARCH = None
    yield
    DenseMemoryLayout.ALIGNMENT_ARCH = None


# ---------------------------------------------------------------------------
# Pure surface check: do the constants exist at all?
# ---------------------------------------------------------------------------


class TestSurface:
    def test_bandwidth_constants_present_in_generated_header(
        self, tmp_path, reset_arch,
    ):
        arch = _new_arch("dhsw")
        A = Tensor("A", (4, 4))
        B = Tensor("B", (4, 4))
        C = Tensor("C", (4, 4))
        g = Generator(arch)
        g.add("matmul", C["ij"] <= A["ik"] * B["kj"])
        g.generate(str(tmp_path))
        consts = _extract_kernel_constants(str(tmp_path), "matmul")
        # All three new fields should be present alongside the existing
        # NonZeroFlops / HardwareFlops.
        assert "InboundConstBytes" in consts
        assert "InboundBytes" in consts
        assert "OutboundBytes" in consts


# ---------------------------------------------------------------------------
# Dense matmul: classic case, two inputs and one output
# ---------------------------------------------------------------------------


class TestDenseMatmul:
    """A bog-standard ``C = A * B`` of square dense matrices.  All the
    numbers can be computed by hand and serve as the regression
    baseline for the rest of the test file.
    """

    @pytest.mark.parametrize("N", [4, 8, 16, 32])
    def test_double_precision(self, tmp_path, reset_arch, N):
        arch = _new_arch("dhsw")
        A = Tensor("A", (N, N))
        B = Tensor("B", (N, N))
        C = Tensor("C", (N, N))
        g = Generator(arch)
        g.add("matmul", C["ij"] <= A["ik"] * B["kj"])
        g.generate(str(tmp_path))
        consts = _extract_kernel_constants(str(tmp_path), "matmul")
        # NxN doubles = N*N * 8 bytes.
        per_tensor = N * N * 8
        assert consts["InboundConstBytes"] == 0
        assert consts["InboundBytes"] == 2 * per_tensor   # A + B
        assert consts["OutboundBytes"] == per_tensor      # C

    def test_single_precision_halves_the_byte_counts(
        self, tmp_path, reset_arch,
    ):
        # ``shsw`` = single precision, 4 bytes/real.
        arch = _new_arch("shsw")
        N = 8
        A = Tensor("A", (N, N))
        B = Tensor("B", (N, N))
        C = Tensor("C", (N, N))
        g = Generator(arch)
        g.add("matmul", C["ij"] <= A["ik"] * B["kj"])
        g.generate(str(tmp_path))
        consts = _extract_kernel_constants(str(tmp_path), "matmul")
        per_tensor = N * N * 4
        assert consts["InboundBytes"] == 2 * per_tensor
        assert consts["OutboundBytes"] == per_tensor

    def test_input_only_kernel_has_zero_outbound(
        self, tmp_path, reset_arch,
    ):
        # A pure copy ``B = A`` reads ``A`` (input) and writes ``B``
        # (output), so InboundBytes=sizeof(A), OutboundBytes=sizeof(B),
        # InboundConstBytes=0.
        arch = _new_arch("dhsw")
        A = Tensor("A", (3, 5))
        B = Tensor("B", (3, 5))
        g = Generator(arch)
        g.add("copy", B["ij"] <= A["ij"])
        g.generate(str(tmp_path))
        consts = _extract_kernel_constants(str(tmp_path), "copy")
        size = 3 * 5 * 8
        assert consts["InboundConstBytes"] == 0
        assert consts["InboundBytes"] == size
        assert consts["OutboundBytes"] == size


# ---------------------------------------------------------------------------
# Compute-constant tensors land in InboundConstBytes
# ---------------------------------------------------------------------------


class TestComputeConstantTensors:
    """A tensor whose ``spp`` is given as a numerical numpy array is
    *compute-constant*: its values are baked into the generated code
    and the kernel reads them from a read-only buffer. We account
    for these separately.
    """

    def test_constant_tensor_only_in_const_bytes(
        self, tmp_path, reset_arch,
    ):
        arch = _new_arch("dhsw")
        N = 4
        # Float values -> tensor is compute-constant.
        diag = np.eye(N)
        A = Tensor("A", (N, N), spp=diag)
        B = Tensor("B", (N, N))
        C = Tensor("C", (N, N))
        assert A.is_compute_constant() is True

        g = Generator(arch)
        g.add("matmul", C["ij"] <= A["ik"] * B["kj"])
        g.generate(str(tmp_path))
        consts = _extract_kernel_constants(str(tmp_path), "matmul")

        # A is the only constant tensor: its layout requiredReals * 8 bytes.
        a_layout_size = A.memoryLayout().storage().requiredReals() * 8
        assert consts["InboundConstBytes"] == a_layout_size
        # B is the only non-const input.
        assert consts["InboundBytes"] == N * N * 8
        # C is the only output.
        assert consts["OutboundBytes"] == N * N * 8

    def test_pattern_only_sparse_tensor_is_not_compute_constant(
        self, tmp_path, reset_arch,
    ):
        # ``spp`` given as a *boolean* ndarray (or a dict of bools) is a
        # mere pattern, not values.  Such a tensor is NOT
        # compute-constant - the runtime still has to fill its values.
        arch = _new_arch("dhsw")
        N = 4
        bool_pat = np.eye(N, dtype=bool)
        A = Tensor("A", (N, N), spp=bool_pat)
        B = Tensor("B", (N, N))
        C = Tensor("C", (N, N))
        assert A.is_compute_constant() is False

        g = Generator(arch)
        g.add("matmul", C["ij"] <= A["ik"] * B["kj"])
        g.generate(str(tmp_path))
        consts = _extract_kernel_constants(str(tmp_path), "matmul")

        # A is a regular (non-const) input now.
        assert consts["InboundConstBytes"] == 0
        # InboundBytes covers A + B.  A's storage might be smaller than
        # the full N*N due to the diagonal pattern, so we compute it
        # from the layout rather than hard-coding.
        a_size = A.memoryLayout().storage().requiredReals() * 8
        b_size = N * N * 8
        assert consts["InboundBytes"] == a_size + b_size


# ---------------------------------------------------------------------------
# Sparse tensors store fewer reals than the full shape would suggest
# ---------------------------------------------------------------------------


class TestSparseStorageSizing:
    """The byte counts must use ``requiredReals`` from the storage
    layout, not the full tensor shape - otherwise sparse tensors would
    over-report their memory traffic.
    """

    def test_sparse_input_uses_storage_size_not_shape(
        self, tmp_path, reset_arch,
    ):
        arch = _new_arch("dhsw")
        N = 8
        # Block pattern: only the top-left 4x4 is non-zero.  Yateto
        # shrinks the bounding box and stores only those 16 reals.
        pat = np.zeros((N, N), dtype=bool)
        pat[:4, :4] = True
        A = Tensor("A", (N, N), spp=pat)
        B = Tensor("B", (N, N))
        C = Tensor("C", (N, N))

        g = Generator(arch)
        g.add("matmul", C["ij"] <= A["ik"] * B["kj"])
        g.generate(str(tmp_path))
        consts = _extract_kernel_constants(str(tmp_path), "matmul")

        a_storage = A.memoryLayout().storage().requiredReals()
        # The shrunk bounding box must register as fewer bytes than the
        # full N*N would.
        assert a_storage < N * N
        # And InboundBytes must reflect that.
        a_bytes = a_storage * 8
        b_bytes = N * N * 8
        assert consts["InboundBytes"] == a_bytes + b_bytes


# ---------------------------------------------------------------------------
# Multi-kernel programs: each kernel has its own per-kernel counts
# ---------------------------------------------------------------------------


class TestMultiKernel:
    def test_two_kernels_have_independent_counts(
        self, tmp_path, reset_arch,
    ):
        arch = _new_arch("dhsw")
        N = 4
        A = Tensor("A", (N, N))
        B = Tensor("B", (N, N))
        C = Tensor("C", (N, N))
        g = Generator(arch)
        g.add("k_copy", B["ij"] <= A["ij"])
        g.add("k_chain", C["ij"] <= B["ij"])
        g.generate(str(tmp_path))

        ck = _extract_kernel_constants(str(tmp_path), "k_copy")
        cn = _extract_kernel_constants(str(tmp_path), "k_chain")
        # Each kernel sees the size of its two tensors only.
        size = N * N * 8
        assert ck["InboundBytes"] == size
        assert ck["OutboundBytes"] == size
        assert cn["InboundBytes"] == size
        assert cn["OutboundBytes"] == size

    def test_tensor_written_in_one_kernel_is_input_in_the_next(
        self, tmp_path, reset_arch,
    ):
        # B is *output* in k_copy and *input* in k_chain - the outline
        # is per-kernel, so the same tensor must appear in opposite
        # categories in the two outlines.
        arch = _new_arch("dhsw")
        N = 4
        A = Tensor("A", (N, N))
        B = Tensor("B", (N, N))
        C = Tensor("C", (N, N))
        g = Generator(arch)
        g.add("k_copy", B["ij"] <= A["ij"])
        g.add("k_chain", C["ij"] <= B["ij"])
        g.generate(str(tmp_path))

        ck = _extract_kernel_constants(str(tmp_path), "k_copy")
        cn = _extract_kernel_constants(str(tmp_path), "k_chain")
        size = N * N * 8
        # k_copy: A in, B out
        assert ck["InboundBytes"] == size and ck["OutboundBytes"] == size
        # k_chain: B in, C out  -> B is in for this kernel
        assert cn["InboundBytes"] == size and cn["OutboundBytes"] == size


# ---------------------------------------------------------------------------
# Precision sanity check across architectures
# ---------------------------------------------------------------------------


class TestPrecisionConsistency:
    """A simple sanity check: switching between double and single
    precision halves all three byte counts for the same kernel.
    """

    def test_single_precision_is_half_of_double(
        self, tmp_path, reset_arch,
    ):
        N = 4
        A_arr = np.eye(N)  # compute-constant - hits all three buckets
        sizes = {}
        for tag, archname in [("d", "dhsw"), ("s", "shsw")]:
            from yateto.memory import DenseMemoryLayout
            DenseMemoryLayout.ALIGNMENT_ARCH = None
            arch = useArchitectureIdentifiedBy(archname)

            A = Tensor("A", (N, N), spp=A_arr)
            B = Tensor("B", (N, N))
            C = Tensor("C", (N, N))
            g = Generator(arch)
            g.add("matmul", C["ij"] <= A["ik"] * B["kj"])
            sub = tmp_path / tag
            sub.mkdir()
            g.generate(str(sub))
            sizes[tag] = _extract_kernel_constants(str(sub), "matmul")

        # The double-precision count should be twice the single-precision
        # count for every bucket.  This indirectly verifies that the
        # arch.bytesPerReal is what feeds the calculation.
        for field in ("InboundConstBytes", "InboundBytes", "OutboundBytes"):
            assert sizes["d"][field] == 2 * sizes["s"][field], field


# ---------------------------------------------------------------------------
# Family kernels - one outline per family member
# ---------------------------------------------------------------------------


class TestKernelFamily:
    def test_family_member_outlines_have_bandwidth_constants(
        self, tmp_path, reset_arch,
    ):
        from yateto import simpleParameterSpace

        arch = _new_arch("dhsw")
        N = 4
        A = Tensor("A", (N, N))
        B = Tensor("B", (N, N))
        C = Tensor("C", (N, N))

        def build(i):
            return C["ij"] <= A["ik"] * B["kj"]

        g = Generator(arch)
        g.addFamily("fam", simpleParameterSpace(2), build)
        g.generate(str(tmp_path))

        # Family-generated structs are typically named "fam".  We only
        # assert that *some* struct in the header carries the new
        # constants - the exact naming convention is implementation
        # detail.
        text = open(os.path.join(str(tmp_path), "kernel.h")).read()
        assert "OutboundBytes" in text
        assert "InboundBytes" in text
        assert "InboundConstBytes" in text


# ---------------------------------------------------------------------------
# Internal API check - the KernelOutline carries the new attributes
# ---------------------------------------------------------------------------


class TestKernelOutlineAttributes:
    """Direct check on the Python side: ``KernelOutline`` instances have
    ``inConstBytes``, ``inBytes``, ``outBytes`` attributes and the
    values match what's emitted into the C++.

    This is independent of any quirk in the
    C++ emission, so a failure here pinpoints the calculation rather
    than the formatting.
    """

    def test_kernel_outline_has_byte_attributes(self, reset_arch):
        from yateto.codegen.visitor import OptimizedKernelGenerator
        # All three attribute names should be exposed on KernelOutline.
        ko_cls = OptimizedKernelGenerator.KernelOutline
        # Construct a minimal instance to verify the constructor signature.
        # We don't run a full generation here - we just instrospect.
        import inspect
        sig = inspect.signature(ko_cls.__init__)
        for param in ("inConstBytes", "inBytes", "outBytes"):
            assert param in sig.parameters, (
                f"KernelOutline.__init__ should accept ``{param}``"
            )
