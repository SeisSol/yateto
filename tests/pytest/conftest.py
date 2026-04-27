"""
Pytest configuration and fixtures for the Yateto Python unit-test suite.

These tests exercise Yateto's compiler-style pipeline purely in Python
(frontend DSL -> AST passes -> control-flow graph).  They intentionally
stop before C++ code generation / compilation, so they are fast, do not
depend on libxsmm / PSpaMM / CxxTest / a C++ toolchain, and can run
everywhere the `yateto` package imports.

The matching C++/code-gen integration tests live under ``tests/code-gen``
and are driven by the GitHub Actions workflow ``yateto-cpu.yml`` - we
do not duplicate them here.
"""
from __future__ import annotations

import os
import sys

import pytest


# Make the yateto source tree importable even when the package has not been
# installed via ``pip install -e .`` (e.g. when running the tests locally
# straight from a clone).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ---------------------------------------------------------------------------
# Common fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def arch():
    """A host architecture used when passes need alignment info.

    ``dhsw`` = double precision on Haswell.  Same default as the example
    scripts.  We re-set the layout's global alignment reference on every
    test so no state leaks between tests.
    """
    from yateto import useArchitectureIdentifiedBy
    from yateto.memory import DenseMemoryLayout

    a = useArchitectureIdentifiedBy("dhsw")
    yield a
    # Reset global alignment state to keep tests hermetic.
    DenseMemoryLayout.ALIGNMENT_ARCH = None


@pytest.fixture
def square_tensors():
    """A handful of 8x8 tensors, useful for most elementwise/matmul tests."""
    from yateto import Tensor

    N = 8
    return {
        "N": N,
        "A": Tensor("A", (N, N)),
        "B": Tensor("B", (N, N)),
        "C": Tensor("C", (N, N)),
    }


@pytest.fixture
def deduced():
    """Helper that runs ``DeduceIndices`` on an AST and returns the result.

    ``DeduceIndices`` is the first mandatory pass after the DSL builds the
    tree - without it most other visitors/transformers are not meaningful,
    so almost every test needs it.
    """
    from yateto.ast.transformer import DeduceIndices

    def _deduce(ast, target=None):
        return DeduceIndices(target).visit(ast)

    return _deduce


@pytest.fixture
def run_ast_pipeline(arch):
    """Push an AST through the middle-end up to the point where flops are
    countable.  Returns the (transformed) AST so tests can inspect it.
    """
    from yateto.ast.transformer import (
        DeduceIndices,
        EquivalentSparsityPattern,
        SetSparsityPattern,
        StrengthReduction,
    )
    from yateto.ast.cost import BoundingBoxCostEstimator

    def _run(ast):
        ast = DeduceIndices().visit(ast)
        ast = EquivalentSparsityPattern().visit(ast)
        ast = StrengthReduction(BoundingBoxCostEstimator).visit(ast)
        ast = SetSparsityPattern().visit(ast)
        return ast

    return _run
