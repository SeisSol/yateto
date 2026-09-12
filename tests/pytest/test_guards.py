"""Guards: conjunctions of literals over versioned condition values."""

import pytest

from yateto import Tensor
from yateto.aspp import dense
from yateto.controlflow.graph import Guard, LiveSet, Variable
from yateto.memory import DenseMemoryLayout


def var(name, shape=()):
    return Variable(name, True, DenseMemoryLayout.fromSpp(dense(shape)), dense(shape))


@pytest.fixture
def abc():
    return var('a'), var('b'), var('c')


class TestGuardAlgebra:
    def test_always_and_never(self):
        assert Guard.always().isAlways()
        assert not Guard.always().isNever()
        assert Guard.never().isNever()
        assert not Guard.never().isAlways()

    def test_coerce_from_bool(self):
        assert Guard.coerce(True).isAlways()
        assert Guard.coerce(False).isNever()
        assert Guard.coerce(Guard.always()).isAlways()

    def test_conjunction_is_idempotent(self, abc):
        a, _, _ = abc
        g = Guard.literal(a)
        assert g & g == g
        assert g & g & g == g

    def test_conjunction_is_commutative(self, abc):
        a, b, _ = abc
        assert Guard.literal(a) & Guard.literal(b) == Guard.literal(b) & Guard.literal(a)

    def test_always_is_the_unit(self, abc):
        a, _, _ = abc
        g = Guard.literal(a)
        assert g & Guard.always() == g
        assert Guard.always() & g == g

    def test_never_absorbs(self, abc):
        a, _, _ = abc
        assert (Guard.literal(a) & Guard.never()).isNever()
        assert (Guard.never() & Guard.literal(a)).isNever()

    def test_contradictory_polarities_are_never(self, abc):
        a, _, _ = abc
        yes = Guard.literal(a, polarity=True)
        no = Guard.literal(a, polarity=False)
        assert (yes & no).isNever()

    def test_hashable_and_usable_as_key(self, abc):
        a, b, _ = abc
        seen = {Guard.literal(a): 1, Guard.literal(b): 2}
        seen[Guard.literal(a)] = 3
        assert len(seen) == 2

    def test_guard_is_not_a_bool(self, abc):
        a, _, _ = abc
        # `if guard:` would silently be true for every guard, so it is refused
        with pytest.raises(TypeError):
            bool(Guard.literal(a))


class TestImplication:
    def test_everything_implies_always(self, abc):
        a, _, _ = abc
        assert Guard.literal(a).implies(Guard.always())
        assert Guard.always().implies(Guard.always())

    def test_always_implies_only_always(self, abc):
        a, _, _ = abc
        assert not Guard.always().implies(Guard.literal(a))

    def test_never_implies_everything(self, abc):
        a, _, _ = abc
        assert Guard.never().implies(Guard.literal(a))

    def test_stronger_implies_weaker(self, abc):
        a, b, _ = abc
        both = Guard.literal(a) & Guard.literal(b)
        assert both.implies(Guard.literal(a))
        assert both.implies(Guard.literal(b))
        assert not Guard.literal(a).implies(both)

    def test_is_reflexive(self, abc):
        a, b, _ = abc
        g = Guard.literal(a) & Guard.literal(b)
        assert g.implies(g)

    def test_unrelated_guards_do_not_imply(self, abc):
        a, b, _ = abc
        assert not Guard.literal(a).implies(Guard.literal(b))

    def test_opposite_polarity_does_not_imply(self, abc):
        a, _, _ = abc
        assert not Guard.literal(a, polarity=True).implies(Guard.literal(a, polarity=False))


class TestVersioning:
    """A condition tensor may be rewritten, so a version is part of the literal."""

    def test_versions_are_distinct_literals(self, abc):
        a, _, _ = abc
        assert Guard.literal(a, 1) != Guard.literal(a, 2)

    def test_a_later_version_does_not_imply_an_earlier_one(self, abc):
        a, _, _ = abc
        assert not Guard.literal(a, 2).implies(Guard.literal(a, 1))
        assert not Guard.literal(a, 1).implies(Guard.literal(a, 2))

    def test_versions_do_not_contradict_each_other(self, abc):
        a, _, _ = abc
        # different values of the same tensor may well both be true
        assert not (Guard.literal(a, 1) & Guard.literal(a, 2)).isNever()

    def test_same_version_conjoins_to_one_literal(self, abc):
        a, _, _ = abc
        assert len((Guard.literal(a, 1) & Guard.literal(a, 1)).literals()) == 1


class TestGuardEmission:
    def test_always_and_never(self):
        assert Guard.always().ccode() == 'true'
        assert Guard.never().ccode() == 'false'

    def test_single_literal(self, abc):
        a, _, _ = abc
        assert Guard.literal(a).ccode() == '(a[0])'

    def test_conjunction_is_anded(self, abc):
        a, b, _ = abc
        code = (Guard.literal(a) & Guard.literal(b)).ccode()
        assert code.count('&&') == 1
        assert 'a[0]' in code and 'b[0]' in code

    def test_repeated_literal_is_emitted_once(self, abc):
        a, _, _ = abc
        assert (Guard.literal(a) & Guard.literal(a)).ccode().count('a[0]') == 1

    def test_emission_order_is_stable(self, abc):
        a, b, _ = abc
        assert (Guard.literal(a) & Guard.literal(b)).ccode() \
            == (Guard.literal(b) & Guard.literal(a)).ccode()

    def test_variables_are_reported(self, abc):
        a, b, _ = abc
        assert (Guard.literal(a) & Guard.literal(b)).variables() == {a, b}
        assert Guard.always().variables() == set()


class TestLiveSet:
    def test_unconditional_write_kills(self, abc):
        a, _, _ = abc
        live = LiveSet({a: Guard.always()})
        assert a not in (live - {a: Guard.always()}).variables()

    def test_conditional_write_keeps_the_variable_live(self, abc):
        a, b, _ = abc
        # under the negation of the guard the old value survives
        live = LiveSet({a: Guard.always()})
        assert a in (live - {a: Guard.literal(b)}).variables()

    def test_conditional_write_widens_to_unconditional(self, abc):
        a, b, _ = abc
        live = LiveSet({a: Guard.literal(b)})
        assert (live - {a: Guard.literal(b)}).guardOf(a).isAlways()

    def test_killing_an_absent_variable_is_a_no_op(self, abc):
        a, b, _ = abc
        live = LiveSet({a: Guard.always()})
        assert (live - {b: Guard.always()}).variables() == {a}

    def test_join_of_equal_guards_keeps_the_guard(self, abc):
        a, b, _ = abc
        joined = LiveSet({a: Guard.literal(b)}) | {a: Guard.literal(b)}
        assert joined.guardOf(a) == Guard.literal(b)

    def test_join_of_different_guards_widens(self, abc):
        a, b, c = abc
        joined = LiveSet({a: Guard.literal(b)}) | {a: Guard.literal(c)}
        assert joined.guardOf(a).isAlways()

    def test_join_adds_new_variables(self, abc):
        a, b, _ = abc
        assert (LiveSet({a: Guard.always()}) | {b: Guard.always()}).variables() == {a, b}

    def test_membership_of_a_dead_variable(self, abc):
        a, b, _ = abc
        assert (a, Guard.always()) not in LiveSet({b: Guard.always()})

    def test_membership_under_a_satisfiable_guard(self, abc):
        a, b, _ = abc
        assert (a, Guard.literal(b)) in LiveSet({a: Guard.always()})

    def test_membership_under_a_contradictory_guard(self, abc):
        a, b, _ = abc
        live = LiveSet({a: Guard.literal(b, polarity=True)})
        assert (a, Guard.literal(b, polarity=False)) not in live
