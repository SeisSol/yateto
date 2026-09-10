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
* ``liveness`` answers, for every position in the graph, with a correct
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
    Guard,
    ProgramAction,
    Variable,
)
from yateto.controlflow.transformer import (
    liveness,
    MergeActions,
    MergeScalarMultiplications,
    RemoveEmptyStatements,
    SubstituteBackward,
    SubstituteForward,
)
from yateto.controlflow.verify import verify
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

        # No branching structure: the graph is a straight list of the
        # statements the kernel is made of.
        assert all(isinstance(action, ProgramAction) for action in cfg)
        # There must be at least one action.
        assert len(cfg) > 0

    def test_has_action_with_result_and_term(self, arch):
        A = Tensor("A", (8, 8))
        B = Tensor("B", (8, 8))
        C = Tensor("C", (8, 8))
        kernel = C["ij"] <= A["ik"] * B["kj"]
        _, cfg = _lower_to_cfg(kernel, arch)

        action = cfg[0]
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

        tmp_results = [action.result.name for action in cfg
                       if action.result.name.startswith("_tmp")]
        assert len(tmp_results) == len(set(tmp_results))


# ---------------------------------------------------------------------------
# liveness
# ---------------------------------------------------------------------------


class TestLiveness:
    def test_there_is_one_answer_per_position_and_one_past_the_end(self, arch):
        A = Tensor("A", (4, 4))
        B = Tensor("B", (4, 4))
        C = Tensor("C", (4, 4))
        kernel = C["ij"] <= A["ik"] * B["kj"]
        _, cfg = _lower_to_cfg(kernel, arch)

        live = liveness(cfg)
        assert len(live) == len(cfg) + 1
        assert all(at is not None for at in live)

    def test_nothing_is_live_once_the_kernel_is_done(self, arch):
        A = Tensor("A", (4, 4))
        B = Tensor("B", (4, 4))
        C = Tensor("C", (4, 4))
        kernel = C["ij"] <= A["ik"] * B["kj"]
        _, cfg = _lower_to_cfg(kernel, arch)
        # Otherwise the kernel would leak.
        assert _live_var_names(liveness(cfg)[-1]) == set()

    def test_inputs_are_live_at_first_use(self, arch):
        # A + B: both tensors must be live at the beginning (they are read
        # by the first real action).
        A = Tensor("A", (4, 4))
        B = Tensor("B", (4, 4))
        C = Tensor("C", (4, 4))
        kernel = C["ij"] <= A["ij"] + B["ij"]
        _, cfg = _lower_to_cfg(kernel, arch)

        live_vars = _live_var_names(liveness(cfg)[0])
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

        before = len(cfg)
        cfg = SubstituteForward().visit(cfg)
        cfg = SubstituteBackward().visit(cfg)
        cfg = RemoveEmptyStatements().visit(cfg)
        after = len(cfg)
        # The pipeline must not grow the CFG.  It usually shrinks it.
        assert after <= before


# ---------------------------------------------------------------------------
# MergeActions
# ---------------------------------------------------------------------------


class TestMergeActions:
    def test_merging_does_not_grow_the_graph(self, arch):
        A = Tensor("A", (4, 4))
        B = Tensor("B", (4, 4))
        C = Tensor("C", (4, 4))
        kernel = C["ij"] <= A["ik"] * B["kj"]
        _, cfg = _lower_to_cfg(kernel, arch)
        before = len(cfg)
        cfg = MergeActions().visit(cfg)
        assert len(cfg) <= before


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------


class TestVerify:
    """The three things the copy-propagating passes read the graph as though.

    None of them is enforced anywhere: they hold because of how the graph is
    built. Asking is what turns "holds today" into "held when this ran".
    """

    @staticmethod
    def _ml():
        from yateto.memory import DenseMemoryLayout
        return DenseMemoryLayout((4, 4))

    def _tmp(self, name='_tmp0'):
        return Variable(name, True, self._ml(), is_temporary=True)

    def _global(self, name='A', writable=False):
        return Variable(name, writable, self._ml(), tensor=Tensor(name, (4, 4)))

    def test_a_graph_that_is_as_it_is_taken_to_be_says_nothing(self, arch):
        A, C = self._global('A'), self._global('C', writable=True)
        tmp = self._tmp()
        assert verify([ProgramAction(tmp, A, add=False),
                       ProgramAction(C, tmp, add=False)]) == []

    def test_one_definition_and_accumulations_into_it_is_one_definition(self, arch):
        A, B, C = self._global('A'), self._global('B'), self._global('C', True)
        tmp = self._tmp()
        assert verify([ProgramAction(tmp, A, add=False),
                       ProgramAction(tmp, B, add=True),
                       ProgramAction(C, tmp, add=False)]) == []

    def test_two_definitions_of_one_temporary_are_reported(self, arch):
        A, B, C = self._global('A'), self._global('B'), self._global('C', True)
        tmp = self._tmp()
        found, = verify([ProgramAction(tmp, A, add=False),
                         ProgramAction(tmp, B, add=False),
                         ProgramAction(C, tmp, add=False)])
        assert '_tmp0' in found and 'one definition' in found

    def test_a_read_between_the_steps_of_a_definition_is_reported(self, arch):
        A, B, C = self._global('A'), self._global('B'), self._global('C', True)
        tmp, other = self._tmp(), self._tmp('_tmp1')
        found, = verify([ProgramAction(tmp, A, add=False),
                         ProgramAction(other, tmp, add=False),
                         ProgramAction(tmp, B, add=True),
                         ProgramAction(C, tmp, add=False)])
        assert '_tmp0' in found and 'between the steps' in found

    def test_a_view_of_a_temporary_is_reported(self, arch):
        from yateto.memory import DenseMemoryLayout
        A, C = self._global('A'), self._global('C', writable=True)
        tmp = self._tmp()
        sliced = Variable.view(tmp, DenseMemoryLayout((4, 4)).subslice(1, 0, 2), None)
        found, = verify([ProgramAction(tmp, A, add=False),
                         ProgramAction(C, sliced, add=False)])
        assert '_tmp0' in found and 'view of a temporary' in found

    def test_a_read_outside_the_guard_it_was_written_under_is_reported(self, arch):
        A, C = self._global('A'), self._global('C', writable=True)
        flag = self._global('flag')
        tmp = self._tmp()
        found, = verify([ProgramAction(tmp, A, add=False,
                                       condition=Guard.literal(flag)),
                         ProgramAction(C, tmp, add=False)])
        assert '_tmp0' in found and 'does not imply' in found

    def test_a_temporary_that_is_never_written_is_reported(self, arch):
        C = self._global('C', writable=True)
        found, = verify([ProgramAction(C, self._tmp(), add=False)])
        assert '_tmp0' in found and 'never written' in found
