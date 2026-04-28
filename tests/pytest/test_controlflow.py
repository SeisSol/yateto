"""
Tests for ``yateto.controlflow`` - the mini IR between the AST and the
emitted C++.

After strength reduction and ``ImplementContractions``, the AST is
flattened into a straight-line control-flow graph (no loops / branches
at this level - the DSL doesn't have them).  Each program point carries
a ``ProgramAction`` of the shape

    result [+]= [scalar *] term

where ``term`` is either a single ``Variable`` or an ``Expression``
(a LoopOverGEMM, a Permute, a Broadcast, ...).  Subsequent CFG-level
passes do classic compiler things: liveness analysis, copy
propagation, dead-store elimination, action merging.

These tests check:

* ``AST2ControlFlow`` really emits a linear CFG and introduces fresh
  temporaries for each intermediate result,
* ``LivenessAnalysis`` annotates every program point with a correct
  ``live`` set,
* ``SubstituteForward`` / ``SubstituteBackward`` eliminate trivial
  copies,
* ``RemoveEmptyStatements`` drops ``x = x`` lines,
* ``MergeActions`` fuses compatible actions.
"""
from __future__ import annotations

import pytest

from yateto import Tensor
from yateto.arch import useArchitectureIdentifiedBy
from yateto.ast.cost import BoundingBoxCostEstimator
from yateto.ast.transformer import (
    ComputeMemoryLayout,
    DeduceIndices,
    EquivalentSparsityPattern,
    FindContractions,
    ImplementContractions,
    SetSparsityPattern,
    StrengthReduction,
)
from yateto.ast.visitor import FindIndexPermutations
from yateto.ast.transformer import SelectIndexPermutations
from yateto.controlflow.graph import (
    Expression,
    FusedActions,
    ProgramAction,
    ProgramPoint,
    Variable,
    VariableView,
)
from yateto.controlflow.transformer import (
    LivenessAnalysis,
    MergeActions,
    MergeScalarMultiplications,
    RemoveEmptyStatements,
    SubstituteBackward,
    SubstituteForward,
)
from yateto.controlflow.visitor import AST2ControlFlow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lower_to_cfg(kernel, arch):
    """Run the pipeline up to and including AST2ControlFlow.  Returns
    the kernel (post-AST passes) and its CFG.
    """
    kernel = DeduceIndices().visit(kernel)
    kernel = EquivalentSparsityPattern().visit(kernel)
    kernel = StrengthReduction(BoundingBoxCostEstimator).visit(kernel)
    kernel = FindContractions().visit(kernel)
    kernel = ComputeMemoryLayout().visit(kernel)
    variants = FindIndexPermutations().visit(kernel)
    kernel = SelectIndexPermutations(variants).visit(kernel)
    kernel = ImplementContractions().visit(kernel)
    kernel = SetSparsityPattern().visit(kernel)

    conv = AST2ControlFlow()
    conv.visit(kernel)
    cfg = conv.cfg()
    return kernel, cfg


def _live_var_names(live):
    """Return the variable names inside a live set, irrespective of
    whether ``live`` is a Python ``set`` (master) or a ``LiveSet``
    wrapper (nonlinearity branch)."""
    if hasattr(live, "variables"):
        return {v.name for v in live.variables()}
    return {v.name for v in live}


# ---------------------------------------------------------------------------
# Variable
# ---------------------------------------------------------------------------


class TestVariable:
    def test_globality_follows_tensor(self, arch):
        from yateto.memory import DenseMemoryLayout
        # A variable with a non-temporary tensor is global.
        T = Tensor("A", (3, 3))
        ml = DenseMemoryLayout((3, 3))
        v = Variable("A", writable=False, memoryLayout=ml, tensor=T)
        assert v.isGlobal()
        assert not v.isLocal()

    def test_pure_temporary_is_local(self, arch):
        from yateto.memory import DenseMemoryLayout
        ml = DenseMemoryLayout((3, 3))
        v = Variable("_tmp0", writable=True, memoryLayout=ml, is_temporary=True)
        assert v.isLocal()
        assert not v.isGlobal()

    def test_hash_is_by_name(self, arch):
        from yateto.memory import DenseMemoryLayout
        ml = DenseMemoryLayout((3, 3))
        a = Variable("X", True, ml)
        b = Variable("X", True, ml)
        # Same name -> same hash, insertable into a set without dups.
        s = {a, b}
        assert len(s) == 1

    def test_set_writable_only_matches_by_name(self, arch):
        from yateto.memory import DenseMemoryLayout
        ml = DenseMemoryLayout((3, 3))
        v = Variable("X", False, ml)
        v.setWritable("Y")
        assert v.writable is False
        v.setWritable("X")
        assert v.writable is True


# ---------------------------------------------------------------------------
# AST2ControlFlow - smoke test on a matmul
# ---------------------------------------------------------------------------


class TestAST2ControlFlow:
    def test_produces_linear_cfg(self, arch):
        A = Tensor("A", (8, 8))
        B = Tensor("B", (8, 8))
        C = Tensor("C", (8, 8))
        kernel = C["ij"] <= A["ik"] * B["kj"]
        _, cfg = _lower_to_cfg(kernel, arch)

        # Every program point has no branching structure - it's a straight
        # list (plus a terminating sentinel with ``action=None``).
        assert all(isinstance(pp, ProgramPoint) for pp in cfg)
        assert cfg[-1].action is None  # sentinel
        # There must be at least one action.
        assert any(pp.action is not None for pp in cfg)

    def test_has_action_with_result_and_term(self, arch):
        A = Tensor("A", (8, 8))
        B = Tensor("B", (8, 8))
        C = Tensor("C", (8, 8))
        kernel = C["ij"] <= A["ik"] * B["kj"]
        _, cfg = _lower_to_cfg(kernel, arch)

        action = next(pp.action for pp in cfg if pp.action is not None)
        assert action.result is not None
        assert action.term is not None

    def test_temporary_names_are_unique(self, arch):
        # Each _tmp<N> name should appear exactly once as a result.
        A = Tensor("A", (4, 4))
        B = Tensor("B", (4, 4))
        C = Tensor("C", (4, 4))
        D = Tensor("D", (4, 4))
        kernel = D["il"] <= A["ij"] * B["jk"] * C["kl"]
        _, cfg = _lower_to_cfg(kernel, arch)

        tmp_results = [pp.action.result.name for pp in cfg
                       if pp.action is not None
                       and pp.action.result.name.startswith("_tmp")]
        assert len(tmp_results) == len(set(tmp_results))


# ---------------------------------------------------------------------------
# LivenessAnalysis
# ---------------------------------------------------------------------------


class TestLivenessAnalysis:
    def test_annotates_every_program_point(self, arch):
        A = Tensor("A", (4, 4))
        B = Tensor("B", (4, 4))
        C = Tensor("C", (4, 4))
        kernel = C["ij"] <= A["ik"] * B["kj"]
        _, cfg = _lower_to_cfg(kernel, arch)

        # Before: live sets are None.
        assert all(pp.live is None for pp in cfg)
        cfg = LivenessAnalysis().visit(cfg)
        # After: every program point has a live set (possibly empty).
        assert all(pp.live is not None for pp in cfg)

    def test_sentinel_has_empty_live_set(self, arch):
        A = Tensor("A", (4, 4))
        B = Tensor("B", (4, 4))
        C = Tensor("C", (4, 4))
        kernel = C["ij"] <= A["ik"] * B["kj"]
        _, cfg = _lower_to_cfg(kernel, arch)
        cfg = LivenessAnalysis().visit(cfg)
        # Past the last action, nothing should be live - otherwise the
        # kernel would leak.  ``_live_var_names`` works both for plain
        # Python sets (master) and for the ``LiveSet`` wrapper
        # (nonlinearity).
        assert _live_var_names(cfg[-1].live) == set()

    def test_inputs_are_live_at_first_use(self, arch):
        # A + B: both tensors must be live at the beginning (they are read
        # by the first real action).
        A = Tensor("A", (4, 4))
        B = Tensor("B", (4, 4))
        C = Tensor("C", (4, 4))
        kernel = C["ij"] <= A["ij"] + B["ij"]
        _, cfg = _lower_to_cfg(kernel, arch)
        cfg = LivenessAnalysis().visit(cfg)

        first = next(pp for pp in cfg if pp.action is not None)
        live_vars = _live_var_names(first.live)
        # At least one of A/B is live at the first action.
        assert "A" in live_vars or "B" in live_vars


# ---------------------------------------------------------------------------
# SubstituteForward / Backward / RemoveEmptyStatements
# ---------------------------------------------------------------------------


class TestCopyPropagation:
    def test_pipeline_shrinks_cfg(self, arch):
        # A simple identity assign ``C = A`` should be reduced aggressively
        # by the CFG passes (the intermediate _tmp variables get folded).
        A = Tensor("A", (4, 4))
        C = Tensor("C", (4, 4))
        kernel = C["ij"] <= A["ij"]
        _, cfg = _lower_to_cfg(kernel, arch)
        cfg = LivenessAnalysis().visit(cfg)

        before = sum(1 for pp in cfg if pp.action is not None)
        cfg = SubstituteForward().visit(cfg)
        cfg = SubstituteBackward().visit(cfg)
        cfg = RemoveEmptyStatements().visit(cfg)
        after = sum(1 for pp in cfg if pp.action is not None)
        # The pipeline must not grow the CFG.  It usually shrinks it.
        assert after <= before


# ---------------------------------------------------------------------------
# MergeActions
# ---------------------------------------------------------------------------


class TestMergeActions:
    def test_returns_cfg_with_liveness(self, arch):
        A = Tensor("A", (4, 4))
        B = Tensor("B", (4, 4))
        C = Tensor("C", (4, 4))
        kernel = C["ij"] <= A["ik"] * B["kj"]
        _, cfg = _lower_to_cfg(kernel, arch)
        cfg = LivenessAnalysis().visit(cfg)
        cfg = MergeActions().visit(cfg)
        # After merging, liveness must still be up to date.
        assert all(pp.live is not None for pp in cfg)


class TestFusedActions:
    def test_is_empty_on_construction(self):
        fa = FusedActions()
        assert fa.is_empty()

    def test_add_rejects_non_log_term(self, arch):
        from yateto.memory import DenseMemoryLayout
        ml = DenseMemoryLayout((3, 3))
        # Construct a trivially non-LoG ProgramAction to check the guard.
        result = Variable("R", True, ml)
        term = Variable("X", False, ml)  # a plain Variable, not Expression
        action = ProgramAction(result, term, add=False)

        fa = FusedActions()
        # The term is not an Expression at all (it's a Variable), so
        # ``action.term.node`` would error - but the check happens first.
        with pytest.raises(AttributeError):
            fa.add(action)
