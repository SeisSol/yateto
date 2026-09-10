"""
End-to-end pipeline tests on the actual example kernels shipped with
Yateto.

These are **integration** tests at the Python level: they load each
example from ``tests/code-gen/*.py`` (the same scripts that the CI
builds and runs with real C++ compilers), and push them through the
full Python pipeline up to ``prepareUntilCodeGen``.  They also invoke
``Generator.generate`` into a scratch directory to make sure the C++
emission itself doesn't crash.

They **do not** compile or run the generated C++ - that's what the
``yateto-cpu.yml`` GitHub Actions workflow does.  Our job here is to
catch Python-side regressions much faster.

The tests are written to work on ``master``.
"""
from __future__ import annotations

import importlib.util
import inspect
import os
import sys

import pytest

from yateto import Generator, Tensor
from yateto.generator import Kernel


TEST_SCRIPTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "code-gen")
)


def _available_scripts(*candidates):
    """Filter a candidate list of example scripts to those that
    actually exist in ``tests/code-gen/``.
    """
    return [s for s in candidates
            if os.path.isfile(os.path.join(TEST_SCRIPTS_DIR, f"{s}.py"))]


def _load_example_module(name):
    """Load ``tests/code-gen/<n>.py`` as a fresh module."""
    path = os.path.join(TEST_SCRIPTS_DIR, f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"_example_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Candidate scripts that may exist on either branch
# ---------------------------------------------------------------------------

BASIC_SCRIPTS = _available_scripts("matmul", "minimal", "indices", "slicing")
REGRESSION_SCRIPTS = _available_scripts("regress")  # master only

# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------


class TestExampleScripts:
    """Each example registers its kernels via an ``add(g)`` entry point.
    We feed each example into a fresh ``Generator`` and then push it
    through the pipeline.
    """

    @pytest.mark.parametrize(
        "script",
        BASIC_SCRIPTS + REGRESSION_SCRIPTS,
    )
    def test_example_prepares_without_error(self, arch, script):
        mod = _load_example_module(script)
        g = Generator(arch)
        mod.add(g)

        # Every example must successfully run ``prepareUntilUnitTest``
        # on all its kernels.
        for kernel in g.kernels():
            kernel.prepareUntilUnitTest(arch)
            assert kernel.cfg is not None

    @pytest.mark.parametrize("script", BASIC_SCRIPTS)
    def test_example_generates_cpp(self, arch, tmp_path, script, request):
        mod = _load_example_module(script)
        g = Generator(arch)
        mod.add(g)

        g.generate(str(tmp_path))
        out = os.listdir(str(tmp_path))
        # Always produced:
        for expected in ("kernel.h", "kernel.cpp", "tensor.h", "init.h"):
            assert expected in out, f"{expected} missing for {script}"


# ---------------------------------------------------------------------------
# Flop counts (regression)
# ---------------------------------------------------------------------------


class TestFlopRegression:
    """Lock in the Yateto flop counts for a few canonical kernels.

    These are **regression** tests: if the cost model changes in a way
    that alters the counts, the test will flag it.  A change here is
    not necessarily a bug - it just demands human attention and a
    justified update.
    """

    def test_matmul_32x32(self, arch):
        from yateto.ast.cost import BoundingBoxCostEstimator
        mod = _load_example_module("matmul")
        g = Generator(arch)
        mod.add(g)

        # matmul.py registers 4 kernels (AB / ATB / ABT / ATBT) - all
        # 32x32x32 dense matmuls; Yateto should report the same flop
        # count for each.
        counts = []
        for kernel in g.kernels():
            kernel.prepareUntilUnitTest(arch)
            kernel.prepareUntilCodeGen(BoundingBoxCostEstimator)
            counts.append(kernel.nonZeroFlops)

        # 2*N^3 - N^2 = 65536 - 1024 = 64512 (Yateto's accounting).
        assert all(c == 64512 for c in counts), f"counts={counts}"

    def test_minimal_kernel(self, arch):
        from yateto.ast.cost import BoundingBoxCostEstimator
        mod = _load_example_module("minimal")
        g = Generator(arch)
        mod.add(g)
        kernel = g.kernels()[0]
        kernel.prepareUntilUnitTest(arch)
        kernel.prepareUntilCodeGen(BoundingBoxCostEstimator)
        # The ``minimal`` example's single kernel should come out with
        # a sensible (positive) flop count.
        assert kernel.nonZeroFlops > 0


# ---------------------------------------------------------------------------
# The regression bundle
# ---------------------------------------------------------------------------

class TestRegressions:
    def test_regress_script_runs(self, arch, tmp_path):
        # This script collects bug-fix regressions.  Running it end to
        # end protects against reintroducing those bugs.
        mod = _load_example_module("regress")
        g = Generator(arch)
        mod.add(g)
        g.generate(str(tmp_path))

# ---------------------------------------------------------------------------
# Family kernels via addFamily
# ---------------------------------------------------------------------------


class TestKernelFamily:
    def test_add_family_iterates_over_parameter_space(self, arch):
        from yateto import simpleParameterSpace

        A = Tensor("A", (4, 4))
        B = Tensor("B", (4, 4))
        C = Tensor("C", (4, 4))

        def build(i, j):
            # The specific formula doesn't matter - what matters is that
            # the family machinery produces one valid AST per parameter.
            if (i + j) % 2 == 0:
                return C["ij"] <= A["ik"] * B["kj"]
            return C["ij"] <= A["ij"] + B["ij"]

        g = Generator(arch)
        g.addFamily("fam", simpleParameterSpace(2, 2), build)
        # 2x2 parameter space -> 4 kernels.
        assert len(g.kernels()) == 4
        # We don't run ``generate`` here because on the nonlinearity
        # branch the Broadcast codegen is broken (see TestKnownBugs).
        for kernel in g.kernels():
            kernel.prepareUntilUnitTest(arch)


class TestLoopShape:
    """What the emitted loop nest looks like, which the compiler is sensitive to."""

    @staticmethod
    def _emit(statements, arch=None):
        import pathlib
        import tempfile
        from yateto import useArchitectureIdentifiedBy
        from yateto.gemm_configuration import GeneratorCollection
        arch = arch or useArchitectureIdentifiedBy('dhsw')
        generator = Generator(arch)
        for i, statement in enumerate(statements):
            generator.add(f'k{i}', statement)
        with tempfile.TemporaryDirectory() as out:
            generator.generate(out, gemm_cfg=GeneratorCollection([]))
            return (pathlib.Path(out) / 'kernel.cpp').read_text()

    def test_a_nest_is_one_iteration_space(self):
        """A short innermost loop is unrolled away before a vectoriser sees it,
        so the whole nest is marked rather than that one loop."""
        A = Tensor('A', (20, 9))
        C = Tensor('C', (20, 9))
        code = self._emit([C['ij'] <= 2.0 * A['ij']])
        assert '#pragma omp simd collapse(2)' in code

    def test_a_named_factor_is_read_once(self):
        """It is a member of the kernel object, and a store through one of its
        pointer members may, as far as the compiler can tell, land on it."""
        from yateto import Scalar
        A = Tensor('A', (20, 9))
        C = Tensor('C', (20, 9))
        alpha = Scalar('alpha')
        code = self._emit([C['ij'] <= alpha * A['ij']])
        assert 'double const _alpha = alpha;' in code
        body = code[code.index('k0::execute'):]
        body = body[:body.index('\n  }\n')]
        # once into the local, and only the local inside the nest
        assert body.count('= alpha;') == 1
        assert '_alpha * A' in body

    def test_a_pinned_index_rules_the_clause_out(self):
        """Unrolling puts a declaration between the loops; the nest is no
        longer perfect and collapse would not be legal."""
        import numpy as np
        import yateto.functions as yf
        from yateto.memory import CSCMemoryLayout
        pattern = np.zeros((6, 6), dtype=bool)
        pattern[0, 0] = pattern[1, 1] = pattern[3, 2] = True
        A = Tensor('A', (6, 6), spp=pattern)
        A.setMemoryLayout(CSCMemoryLayout)
        B = Tensor('B', (6, 6), spp=pattern)
        B.setMemoryLayout(CSCMemoryLayout)
        C = Tensor('C', (6, 6))
        code = self._emit([C['ij'] <= yf.add(A['ij'], B['ij'])])
        assert 'collapse' not in code
