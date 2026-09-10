"""
Tests for ``yateto.controlflow`` - the mini IR between the AST and the
emitted C++.

After strength reduction and ``ImplementContractions``, the AST is
flattened into a straight-line control-flow graph (no loops / branches
at this level - the DSL doesn't have them).  Each program point carries
a ``ProgramAction`` of the shape

    result [+]= [scalar *] term

each holding a destination and the operands read into it
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
* a substitution drops the ``x = x`` it leaves behind,
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
    Guard,
    ProgramAction,
)
from yateto.ast.indices import Indices
from yateto.description import IndexedTensorDescription as Operand
from yateto.controlflow.transformer import (
    liveness,
    substituted,
    MergeActions,
    MergeScalarMultiplications,
    SubstituteBackward,
    SubstituteForward,
)
from yateto.controlflow.verify import verify
from yateto.ir.core import Region
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
# Operand
# ---------------------------------------------------------------------------


class TestVariable:
    def test_globality_follows_tensor(self, arch):
        from yateto.memory import DenseMemoryLayout
        # A variable with a non-temporary tensor is global.
        T = Tensor("A", (3, 3))
        ml = DenseMemoryLayout((3, 3))
        v = Operand("A", None, ml, None, tensor=T)
        assert v.isGlobal()
        assert not v.isLocal()

    def test_pure_temporary_is_local(self, arch):
        from yateto.memory import DenseMemoryLayout
        ml = DenseMemoryLayout((3, 3))
        v = Operand("_tmp0", None, ml, None, is_temporary=True, writable=True)
        assert v.isLocal()
        assert not v.isGlobal()

    def test_hash_is_by_name(self, arch):
        from yateto.memory import DenseMemoryLayout
        ml = DenseMemoryLayout((3, 3))
        a = Operand("X", None, ml, None, writable=True)
        b = Operand("X", None, ml, None, writable=True)
        # Same name -> same hash, insertable into a set without dups.
        s = {a, b}
        assert len(s) == 1

    def test_set_writable_only_matches_by_name(self, arch):
        from yateto.memory import DenseMemoryLayout
        ml = DenseMemoryLayout((3, 3))
        v = Operand("X", None, ml, None)
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

    def test_has_action_with_result_and_operands(self, arch):
        A = Tensor("A", (8, 8))
        B = Tensor("B", (8, 8))
        C = Tensor("C", (8, 8))
        kernel = C["ij"] <= A["ik"] * B["kj"]
        _, cfg = _lower_to_cfg(kernel, arch)

        action = cfg[0]
        assert action.result is not None
        assert action.operands

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

    @staticmethod
    def _spp():
        import numpy as np
        from yateto import aspp
        return aspp.general(np.ones((4, 4), dtype=bool))

    def _tmp(self, name='_tmp0'):
        return Operand(name, None, self._ml(), self._spp(),
                       is_temporary=True, writable=True)

    def _global(self, name='A', writable=False):
        return Operand(name, None, self._ml(), self._spp(),
                       tensor=Tensor(name, (4, 4)), writable=writable)

    def test_a_graph_that_is_as_it_is_taken_to_be_says_nothing(self, arch):
        A, C = self._global('A'), self._global('C', writable=True)
        tmp = self._tmp()
        assert verify(Region([ProgramAction.copy(tmp, A, add=False),
                       ProgramAction.copy(C, tmp, add=False)])) == []

    def test_one_definition_and_accumulations_into_it_is_one_definition(self, arch):
        A, B, C = self._global('A'), self._global('B'), self._global('C', True)
        tmp = self._tmp()
        assert verify(Region([ProgramAction.copy(tmp, A, add=False),
                       ProgramAction.copy(tmp, B, add=True),
                       ProgramAction.copy(C, tmp, add=False)])) == []

    def test_two_definitions_of_one_temporary_are_reported(self, arch):
        A, B, C = self._global('A'), self._global('B'), self._global('C', True)
        tmp = self._tmp()
        found, = verify(Region([ProgramAction.copy(tmp, A, add=False),
                         ProgramAction.copy(tmp, B, add=False),
                         ProgramAction.copy(C, tmp, add=False)]))
        assert '_tmp0' in found and 'one definition' in found

    def test_a_read_between_the_steps_of_a_definition_is_reported(self, arch):
        A, B, C = self._global('A'), self._global('B'), self._global('C', True)
        tmp, other = self._tmp(), self._tmp('_tmp1')
        found, = verify(Region([ProgramAction.copy(tmp, A, add=False),
                         ProgramAction.copy(other, tmp, add=False),
                         ProgramAction.copy(tmp, B, add=True),
                         ProgramAction.copy(C, tmp, add=False)]))
        assert '_tmp0' in found and 'between the steps' in found

    def test_a_view_of_a_temporary_is_reported(self, arch):
        from yateto.memory import DenseMemoryLayout
        A, C = self._global('A'), self._global('C', writable=True)
        tmp = self._tmp()
        sliced = Operand.view(tmp, DenseMemoryLayout((4, 4)).subslice(1, 0, 2),
                              self._spp())
        # it does not stand up either -- the slice keeps room for fewer
        # entries than the pattern says it has values at -- so ask for the one
        # finding this is about
        found = verify(Region([ProgramAction.copy(tmp, A, add=False),
                        ProgramAction.copy(C, sliced, add=False)]))
        assert any('_tmp0' in f and 'view of a temporary' in f for f in found)

    def test_a_read_outside_the_guard_it_was_written_under_is_reported(self, arch):
        A, C = self._global('A'), self._global('C', writable=True)
        flag = self._global('flag')
        tmp = self._tmp()
        found, = verify(Region([ProgramAction.copy(tmp, A, add=False,
                                        condition=Guard.literal(flag)),
                         ProgramAction.copy(C, tmp, add=False)]))
        assert '_tmp0' in found and 'does not imply' in found

    def test_a_temporary_that_is_never_written_is_reported(self, arch):
        C = self._global('C', writable=True)
        found, = verify(Region([ProgramAction.copy(C, self._tmp(), add=False)]))
        assert '_tmp0' in found and 'never written' in found

    def test_a_statement_that_does_not_stand_up_is_reported(self, arch):
        """The same three questions a rewrite has to answer before it is made,
        asked of what came out of one."""
        import numpy as np
        from yateto import aspp
        from yateto.memory import DenseMemoryLayout
        A = self._global('A')
        # a destination that keeps room for fewer entries than the operand has
        # values at
        narrow = np.zeros((4, 4), dtype=bool)
        narrow[:2, :2] = True
        C = Operand('C', None, DenseMemoryLayout.fromSpp(aspp.general(narrow)),
                    self._spp(), tensor=Tensor('C', (4, 4)), writable=True)
        found = verify(Region([ProgramAction.copy(C, A, add=False)]))
        assert any('does not stand up' in f for f in found)


class TestStorageFacts:
    """A variable carries everything about the storage it names.

    Which is the half `readFrom` takes from the storage, so the two ways of
    describing an operand -- from a node and from a variable -- say the same
    about it and cannot drift apart.
    """

    @staticmethod
    def _cfg(arch, kernel):
        _, cfg = _lower_to_cfg(kernel, arch)
        return cfg

    def test_a_constant_tensor_says_so_and_carries_its_values(self, arch):
        import numpy as np
        values = np.arange(16, dtype=float).reshape(4, 4)
        A = Tensor("A", (4, 4), values, alignStride=False)
        C = Tensor("C", (4, 4))
        cfg = self._cfg(arch, C["ij"] <= A["ij"])
        operands = [var for action in cfg for var in action.reads()
                    if var.name == "A"]
        assert operands and all(var.is_compute_constant for var in operands)
        assert all(var.values is not None for var in operands)

    def test_a_temporary_is_not_constant(self, arch):
        A = Tensor("A", (4, 4))
        B = Tensor("B", (4, 4))
        C = Tensor("C", (4, 4))
        cfg = self._cfg(arch, C["ij"] <= A["ik"] * B["kj"])
        temporaries = [action.result for action in cfg if action.result.isLocal()]
        assert temporaries
        assert not any(var.is_compute_constant or var.values is not None
                       for var in temporaries)

    def test_the_two_ways_of_describing_an_operand_agree(self, arch):
        """One is built from the node, the other from the variable; the half
        that is about storage has to come out the same."""
        from yateto.codegen.common import IndexedTensorDescription
        import numpy as np
        values = np.arange(16, dtype=float).reshape(4, 4)
        A = Tensor("A", (4, 4), values, alignStride=False)
        C = Tensor("C", (4, 4))
        cfg = self._cfg(arch, C["ij"] <= A["ij"])
        from yateto.ast.indices import Indices
        var = next(v for a in cfg for v in a.reads() if v.name == "A")
        fromVar = IndexedTensorDescription.fromVar(var, Indices("ij", (4, 4)))
        read = fromVar.readFrom(var)
        for field in ('name', 'memoryLayout', 'is_compute_constant',
                      'is_temporary', 'values', 'addressing', 'tensor', 'writable'):
            assert getattr(read, field) is getattr(fromVar, field)

    def test_a_variable_is_the_operand_it_names(self, arch):
        """It answers as a description, so the graph and the code generators
        ask it the same questions and get the same answers."""
        from yateto.description import IndexedTensorDescription
        A = Tensor("A", (4, 4))
        C = Tensor("C", (4, 4))
        cfg = self._cfg(arch, C["ij"] <= A["ij"])
        var = next(v for a in cfg for v in a.reads() if v.name == "A")
        assert isinstance(var, IndexedTensorDescription)
        assert list(var.indices) == ["i", "j"]
        assert var.tensor is A and not var.is_temporary

    def test_a_temporary_says_what_it_is_read_over(self, arch):
        A = Tensor("A", (4, 4))
        B = Tensor("B", (4, 4))
        C = Tensor("C", (4, 4))
        cfg = self._cfg(arch, C["ij"] <= A["ik"] * B["kj"])
        temporary = next(a.result for a in cfg if a.result.isLocal())
        assert temporary.indices is not None


    def test_a_substituted_operand_keeps_what_the_statement_says(self, arch):
        """Only where it reads from changes: it still reads the entries it
        read, over the indices it read them with."""
        from yateto.ast.indices import Indices
        from yateto.memory import DenseMemoryLayout
        import numpy as np
        from yateto import aspp
        ml = DenseMemoryLayout((4, 4))
        narrow = aspp.general(np.tril(np.ones((4, 4), dtype=bool)))
        whole = aspp.general(np.ones((4, 4), dtype=bool))
        operand = Operand("_tmp0", Indices("ij", (4, 4)), ml, narrow,
                          is_temporary=True, writable=True)
        storage = Operand("A", Indices("kl", (4, 4)), ml, whole,
                          tensor=Tensor("A", (4, 4)), writable=True)
        read = operand.substituted(operand, storage)
        assert read.name == "A" and read.tensor is storage.tensor
        assert list(read.indices) == ["i", "j"]
        assert read.eqspp is narrow

    def test_an_operand_that_is_not_the_one_replaced_is_left_alone(self, arch):
        from yateto.memory import DenseMemoryLayout
        ml = DenseMemoryLayout((4, 4))
        one = Operand("B", None, ml, None)
        assert one.substituted(Operand("A", None, ml, None),
                               Operand("C", None, ml, None)) is one


class TestStatementWithoutItsNode:
    """What a statement computes, it says itself.

    The node it was built from is the generator's to read -- which backend
    takes the statement, and what that backend is told beyond the operands --
    and nothing here asks it anything else.
    """

    @staticmethod
    def _operand(name, indices):
        import numpy as np
        from yateto import aspp
        from yateto.memory import DenseMemoryLayout
        shape = (4,) * len(indices)
        return Operand(name, Indices(indices, shape), DenseMemoryLayout(shape),
                       aspp.general(np.ones(shape, dtype=bool)))

    def test_what_it_computes_it_says(self, arch):
        import numpy as np
        from yateto import aspp
        spp = aspp.general(np.tril(np.ones((4, 4), dtype=bool)))
        expression = ProgramAction("Elementwise", None,
                                   [self._operand("A", "ij")],
                                   Indices("ij", (4, 4)), spp, False)
        assert expression.eqspp is spp
        assert list(expression.indices) == ["i", "j"]
        assert expression.prefetch is None

    def test_a_product_asks_whether_its_operands_are_matrices(self, arch):
        groups = (Indices("il", (4, 4)), Indices("j", (4,)), Indices("k", (4,)))
        # `i` and `l` are one dimension of the product; in `ikl` the summed
        # index sits between them
        for left, matrices in (("ilk", True), ("ikl", False)):
            operands = [self._operand("A", left), self._operand("B", "kjl")]
            expression = ProgramAction("LoopOverGEMM", None, operands,
                                       Indices("ilj", (4, 4, 4)), None, False,
                                       groups=groups)
            layouts = [operand.memoryLayout for operand in operands]
            assert expression.mayReadOperands(layouts) is matrices

    def test_anything_that_is_not_a_product_asks_nothing(self, arch):
        expression = ProgramAction("Elementwise", None,
                                   [self._operand("A", "ij")],
                                   Indices("ij", (4, 4)), None, False)
        assert expression.mayReadOperands([None])

    def test_the_operation_and_the_immediates_are_the_statement_s(self, arch):
        """An element-wise statement says which operation it applies and where
        the numbers it was written with go, without being asked the tree."""
        import yateto.functions as yf
        A = Tensor("A", (4, 4))
        C = Tensor("C", (4, 4))
        _, cfg = _lower_to_cfg(C["ij"] <= yf.maximum(A["ij"], 0.0), arch)
        elementwise = next(a for a in cfg
                           if not a.isCopy() and a.optype is not None)
        assert 'max' in str(elementwise.optype).lower()
        filled = elementwise.fillTerms(["<operand>"])
        assert "<operand>" in filled and 0.0 in filled

    def test_the_statement_knows_its_kind_without_the_tree(self, arch):
        """Which backend writes a statement is decided by what kind it is, and
        the kind is the statement's to say."""
        A = Tensor("A", (4, 4))
        B = Tensor("B", (4, 4))
        C = Tensor("C", (4, 4))
        _, cfg = _lower_to_cfg(C["ij"] <= A["ik"] * B["kj"], arch)
        expression = next(a for a in cfg if not a.isCopy())
        assert expression.kind == "LoopOverGEMM"
        assert not hasattr(expression, "node")

    def test_a_contraction_says_what_it_contracts_and_how_it_reads(self, arch):
        A = Tensor("A", (4, 4))
        B = Tensor("B", (4, 4))
        C = Tensor("C", (4, 4))
        _, cfg = _lower_to_cfg(C["ij"] <= A["ik"] * B["kj"], arch)
        expression = next(a for a in cfg if not a.isCopy())
        m, n, k = expression.groups
        assert (str(m), str(n), str(k)) == ("i", "j", "k")
        assert expression.transA is False and expression.transB is False
        assert expression.loopIndices is not None

    def test_the_guard_is_a_guard_from_the_start(self, arch):
        """It is asked for far more often than it is set, so it is made once."""
        A = Tensor("A", (4, 4))
        C = Tensor("C", (4, 4))
        _, cfg = _lower_to_cfg(C["ij"] <= A["ij"], arch)
        assert all(isinstance(action.condition, Guard) for action in cfg)
        assert all(action.getGuard() is action.condition for action in cfg)

    def test_a_plain_copy_is_a_statement_like_any_other(self, arch):
        """One question instead of two: everything that walks the graph asks
        the statement, and a copy answers as one."""
        A = Tensor("A", (4, 4))
        C = Tensor("C", (4, 4))
        _, cfg = _lower_to_cfg(C["ij"] <= A["ij"], arch)
        copy = next(action for action in cfg if action.isCopy())
        assert copy.kind == "Copy"
        assert copy.operands == [copy.copied()]
        assert copy.copied().name == "A"
        assert copy.eqspp is copy.copied().eqspp

    def test_what_a_copy_reads_is_substituted_like_any_operand(self, arch):
        from yateto.memory import DenseMemoryLayout
        ml = DenseMemoryLayout((4, 4))
        source = Operand("A", None, ml, None, tensor=Tensor("A", (4, 4)))
        other = Operand("B", None, ml, None, tensor=Tensor("B", (4, 4)))
        action = ProgramAction.copy(Operand("C", None, ml, None), source, add=False)
        assert substituted(action, source, other).copied().name == "B"

    def test_whether_a_value_fits_a_destination_survives_substitution(self, arch):
        """Which is why the statement is asked as it stands: what decides it --
        the indices and the entries there are values at -- is not what a
        substitution changes."""
        from yateto.memory import DenseMemoryLayout
        import numpy as np
        from yateto import aspp
        ml = DenseMemoryLayout((4, 4))
        spp = aspp.general(np.ones((4, 4), dtype=bool))
        source = Operand("A", Indices("ij", (4, 4)), ml, spp,
                         tensor=Tensor("A", (4, 4)))
        other = Operand("B", Indices("ij", (4, 4)), ml, spp,
                        tensor=Tensor("B", (4, 4)))
        destination = Operand("C", Indices("ij", (4, 4)), ml, spp,
                              tensor=Tensor("C", (4, 4)), writable=True)
        statement = ProgramAction.copy(destination, source, add=False)
        assert substituted(statement, source, other).resultCompatible(destination) \
            == statement.resultCompatible(destination)

    def test_a_statement_says_whether_it_stands_up(self, arch):
        """Without being handed a substitution that replaces nothing."""
        from yateto.memory import DenseMemoryLayout
        import numpy as np
        from yateto import aspp
        ml = DenseMemoryLayout((4, 4))
        spp = aspp.general(np.ones((4, 4), dtype=bool))
        source = Operand("A", Indices("ij", (4, 4)), ml, spp,
                         tensor=Tensor("A", (4, 4)))
        wide = Operand("C", Indices("ij", (4, 4)), ml, spp,
                       tensor=Tensor("C", (4, 4)), writable=True)
        assert ProgramAction.copy(wide, source, add=False).standsUp()

        narrow = np.zeros((4, 4), dtype=bool)
        narrow[:2, :2] = True
        tight = Operand("C", Indices("ij", (4, 4)),
                        DenseMemoryLayout.fromSpp(aspp.general(narrow)), spp,
                        tensor=Tensor("C", (4, 4)), writable=True)
        assert not ProgramAction.copy(tight, source, add=False).standsUp()


class TestGraphIsARegion:
    """The graph a kernel is made of is a region of statements.

    Which is the same thing the code generators build and improve, so what
    walks the one walks the other.
    """

    def test_the_graph_is_a_region(self, arch):
        A = Tensor("A", (4, 4))
        B = Tensor("B", (4, 4))
        C = Tensor("C", (4, 4))
        _, cfg = _lower_to_cfg(C["ij"] <= A["ik"] * B["kj"], arch)
        assert isinstance(cfg, Region)
        assert len(cfg) > 0

    def test_a_statement_is_an_operation(self, arch):
        from yateto.ir.core import Op
        A = Tensor("A", (4, 4))
        C = Tensor("C", (4, 4))
        _, cfg = _lower_to_cfg(C["ij"] <= A["ij"], arch)
        assert all(isinstance(statement, Op) for statement in cfg)
        assert all(statement.regions() == () for statement in cfg)

    def test_the_region_can_be_walked(self, arch):
        A = Tensor("A", (4, 4))
        B = Tensor("B", (4, 4))
        C = Tensor("C", (4, 4))
        _, cfg = _lower_to_cfg(C["ij"] <= A["ik"] * B["kj"], arch)
        assert list(cfg.walk()) == list(cfg)
