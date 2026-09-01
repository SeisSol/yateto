import numpy as np
from . import aspp
from .type import Datatype

# Datatypes ordered by "rank" for the default promotion rules below.
_PROMOTION_ORDER = [
    Datatype.BOOL,
    Datatype.I8,
    Datatype.I16,
    Datatype.I32,
    Datatype.I64,
    Datatype.BF16,
    Datatype.F16,
    Datatype.F32,
    Datatype.F64,
    Datatype.F128,
]


def promote(argtypes):
    """Default result datatype of an operation: the widest of its arguments."""
    known = [t for t in argtypes if t is not None]
    if not known:
        return None
    return max(known, key=lambda t: _PROMOTION_ORDER.index(t))


def _unionSpp(spps):
    spp = spps[0]
    for other in spps[1:]:
        spp = aspp.add(spp, other)
    return spp


def _intersectSpp(spps):
    spp = spps[0]
    for other in spps[1:]:
        spp = aspp.multiply(spp, other)
    return spp


def _denseSpp(spps):
    return aspp.dense(spps[0].shape)


class Operation:
    # Number of arguments the operation takes; None means "variadic".
    ARITY = None

    def call(self, *args):
        raise NotImplementedError()

    def callstr(self, *args) -> str:
        raise NotImplementedError()

    def datatypeResult(self, argtypes):
        return promote(argtypes)

    def sparsityResult(self, spps):
        """Sparsity pattern of the result, given the (already aligned) operand patterns.

        Implementations must over-approximate: claiming a zero that is in fact
        non-zero silently drops the value during code generation.
        """
        return _denseSpp(spps)

    def checkArity(self, nargs):
        if self.ARITY is not None and nargs != self.ARITY:
            raise ValueError(f'{self} takes {self.ARITY} argument(s), got {nargs}.')

    def _key(self):
        # Sub-classes carrying state (e.g. Typecast) must extend this.
        return (type(self).__name__,)

    def __str__(self):
        return type(self).__name__

    def __eq__(self, other):
        return isinstance(other, Operation) and self._key() == other._key()

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        # NOTE: defining __eq__ without __hash__ makes the class unhashable;
        #       operations are used as dict keys / in sets, so keep them hashable.
        return hash(self._key())


class CommutativeMonoidMixin:
    def neutral(self, datatype=None):
        """Neutral element of the monoid, as a Python value.

        Takes the datatype because the neutral element is not always
        datatype-independent (e.g. bitwise `and` over intN needs all-ones,
        while over bool it needs `True`).
        """
        raise NotImplementedError()

    def neutralLiteral(self, datatype) -> str:
        return datatype.literal(self.neutral(datatype))


class RingMixin:
    def formsRing(self, op):
        raise NotImplementedError()


class UnaryArgsMixin:
    ARITY = 1


class BinaryArgsMixin:
    ARITY = 2


class ZeroPreservingMixin:
    """op(0, ..., 0) == 0, hence the result pattern is the union of the operands'."""
    def sparsityResult(self, spps):
        return _unionSpp(spps)


class DenseResultMixin:
    """The result may be non-zero even where all operands are zero."""
    def sparsityResult(self, spps):
        return _denseSpp(spps)


class CFunctionMixin:
    def cppname(self) -> str:
        raise NotImplementedError()

    def callstr(self, *args) -> str:
        return f'{self.cppname()}({", ".join(str(arg) for arg in args)})'


class CUnaryOperatorMixin:
    def cppname(self) -> str:
        raise NotImplementedError()

    def callstr(self, *args) -> str:
        return f'{self.cppname()}({args[0]})'


class CBinaryOperatorMixin:
    def cppname(self) -> str:
        raise NotImplementedError()

    def callstr(self, *args) -> str:
        return f'({args[0]}) {self.cppname()} ({args[1]})'


class _FloatFunction(CFunctionMixin, UnaryArgsMixin, Operation):
    """Common base for the libm-backed unary functions."""
    NPFUN = None
    CPPNAME = None

    def cppname(self):
        return self.CPPNAME

    def call(self, *args):
        return type(self).NPFUN(args[0])

    def datatypeResult(self, argtypes):
        # <cmath> promotes integers to double; we keep the argument type so the
        # AST stays self-consistent and an explicit cast stays the user's job.
        return argtypes[0]


# ---------------------------------------------------------------------------
# trigonometry / elementary functions
# NOTE: np.asin/acos/atan/asinh/acosh/atanh/astype are numpy>=2.0 aliases; the
#       arc* spellings below keep numpy 1.x working as well.
# ---------------------------------------------------------------------------

class Sin(ZeroPreservingMixin, _FloatFunction):
    NPFUN, CPPNAME = staticmethod(np.sin), 'std::sin'
class Cos(DenseResultMixin, _FloatFunction):
    NPFUN, CPPNAME = staticmethod(np.cos), 'std::cos'
class Tan(ZeroPreservingMixin, _FloatFunction):
    NPFUN, CPPNAME = staticmethod(np.tan), 'std::tan'
class Asin(ZeroPreservingMixin, _FloatFunction):
    NPFUN, CPPNAME = staticmethod(np.arcsin), 'std::asin'
class Acos(DenseResultMixin, _FloatFunction):
    NPFUN, CPPNAME = staticmethod(np.arccos), 'std::acos'
class Atan(ZeroPreservingMixin, _FloatFunction):
    NPFUN, CPPNAME = staticmethod(np.arctan), 'std::atan'

class Sinh(ZeroPreservingMixin, _FloatFunction):
    NPFUN, CPPNAME = staticmethod(np.sinh), 'std::sinh'
class Cosh(DenseResultMixin, _FloatFunction):
    NPFUN, CPPNAME = staticmethod(np.cosh), 'std::cosh'
class Tanh(ZeroPreservingMixin, _FloatFunction):
    NPFUN, CPPNAME = staticmethod(np.tanh), 'std::tanh'
class Asinh(ZeroPreservingMixin, _FloatFunction):
    NPFUN, CPPNAME = staticmethod(np.arcsinh), 'std::asinh'
class Acosh(DenseResultMixin, _FloatFunction):
    NPFUN, CPPNAME = staticmethod(np.arccosh), 'std::acosh'
class Atanh(ZeroPreservingMixin, _FloatFunction):
    NPFUN, CPPNAME = staticmethod(np.arctanh), 'std::atanh'

class Log(DenseResultMixin, _FloatFunction):
    NPFUN, CPPNAME = staticmethod(np.log), 'std::log'
class Exp(DenseResultMixin, _FloatFunction):
    NPFUN, CPPNAME = staticmethod(np.exp), 'std::exp'
class Log1p(ZeroPreservingMixin, _FloatFunction):
    NPFUN, CPPNAME = staticmethod(np.log1p), 'std::log1p'
class Expm1(ZeroPreservingMixin, _FloatFunction):
    NPFUN, CPPNAME = staticmethod(np.expm1), 'std::expm1'
class Sqrt(ZeroPreservingMixin, _FloatFunction):
    NPFUN, CPPNAME = staticmethod(np.sqrt), 'std::sqrt'
class Cbrt(ZeroPreservingMixin, _FloatFunction):
    NPFUN, CPPNAME = staticmethod(np.cbrt), 'std::cbrt'
class Abs(ZeroPreservingMixin, _FloatFunction):
    NPFUN, CPPNAME = staticmethod(np.abs), 'std::abs'


class Max(CFunctionMixin, BinaryArgsMixin, CommutativeMonoidMixin, ZeroPreservingMixin, Operation):
    def neutral(self, datatype=None):
        if datatype is not None and datatype.isBool():
            return False
        if datatype is not None and datatype.isInteger():
            return datatype.limits()[0]
        return -float('inf')
    def cppname(self):
        return 'std::max'
    def call(self, *args):
        # NOTE: the builtin max() is ambiguous on arrays
        return np.maximum(args[0], args[1])

class Min(CFunctionMixin, BinaryArgsMixin, CommutativeMonoidMixin, ZeroPreservingMixin, Operation):
    def neutral(self, datatype=None):
        if datatype is not None and datatype.isBool():
            return True
        if datatype is not None and datatype.isInteger():
            return datatype.limits()[1]
        return float('inf')
    def cppname(self):
        return 'std::min'
    def call(self, *args):
        return np.minimum(args[0], args[1])

class Pow(CFunctionMixin, BinaryArgsMixin, DenseResultMixin, Operation):
    def cppname(self):
        return 'std::pow'
    def call(self, *args):
        return np.power(args[0], args[1])


class Div(CBinaryOperatorMixin, BinaryArgsMixin, Operation):
    def cppname(self):
        return '/'
    def call(self, *args):
        return args[0] / args[1]
    def sparsityResult(self, spps):
        # 0/x == 0, but x/0 is not zero -- the numerator decides.
        return spps[0]


class Add(CBinaryOperatorMixin, BinaryArgsMixin, CommutativeMonoidMixin, ZeroPreservingMixin, Operation):
    def cppname(self):
        return '+'
    def call(self, *args):
        return args[0] + args[1]
    def neutral(self, datatype=None):
        return 0

class Mul(CBinaryOperatorMixin, BinaryArgsMixin, CommutativeMonoidMixin, RingMixin, Operation):
    def cppname(self):
        return '*'
    def call(self, *args):
        return args[0] * args[1]
    def neutral(self, datatype=None):
        return 1
    def formsRing(self, op):
        return op == Add()
    def sparsityResult(self, spps):
        return _intersectSpp(spps)


class And(CBinaryOperatorMixin, BinaryArgsMixin, CommutativeMonoidMixin, RingMixin, Operation):
    def cppname(self):
        return '&'
    def call(self, *args):
        return args[0] & args[1]
    def neutral(self, datatype=None):
        # bitwise-and needs all-ones, which for intN is -1 rather than 1
        if datatype is not None and datatype.isInteger():
            return -1
        return True
    def formsRing(self, op):
        return op == Or() or op == Xor()
    def sparsityResult(self, spps):
        return _intersectSpp(spps)

class Or(CBinaryOperatorMixin, BinaryArgsMixin, CommutativeMonoidMixin, RingMixin, Operation):
    def cppname(self):
        return '|'
    def call(self, *args):
        return args[0] | args[1]
    def neutral(self, datatype=None):
        return False if datatype is None or datatype.isBool() else 0
    def formsRing(self, op):
        return op == And()
    def sparsityResult(self, spps):
        return _unionSpp(spps)

class Xor(CBinaryOperatorMixin, BinaryArgsMixin, CommutativeMonoidMixin, RingMixin, Operation):
    def cppname(self):
        return '^'
    def call(self, *args):
        return args[0] ^ args[1]
    def neutral(self, datatype=None):
        return False if datatype is None or datatype.isBool() else 0
    def formsRing(self, op):
        return op == And()
    def sparsityResult(self, spps):
        return _unionSpp(spps)

class Not(CUnaryOperatorMixin, UnaryArgsMixin, DenseResultMixin, Operation):
    """Bitwise complement. For BOOL operands, use LogicalNot instead."""
    def cppname(self):
        return '~'
    def call(self, *args):
        return ~args[0]
    def datatypeResult(self, argtypes):
        assert argtypes[0] != Datatype.BOOL, \
            'Bitwise Not on bool is always true; use LogicalNot instead.'
        return argtypes[0]

class LogicalNot(CUnaryOperatorMixin, UnaryArgsMixin, DenseResultMixin, Operation):
    def cppname(self):
        return '!'
    def call(self, *args):
        return np.logical_not(args[0])
    def datatypeResult(self, argtypes):
        return Datatype.BOOL


class _Comparison(CBinaryOperatorMixin, BinaryArgsMixin, DenseResultMixin, Operation):
    def datatypeResult(self, argtypes):
        return Datatype.BOOL

class CmpEq(_Comparison):
    def cppname(self):
        return '=='
    def call(self, *args):
        return args[0] == args[1]
class CmpNe(_Comparison):
    def cppname(self):
        return '!='
    def call(self, *args):
        return args[0] != args[1]
    def sparsityResult(self, spps):
        # a != b can only hold where at least one side is non-zero
        return _unionSpp(spps)
class CmpLt(_Comparison):
    def cppname(self):
        return '<'
    def call(self, *args):
        return args[0] < args[1]
class CmpLe(_Comparison):
    def cppname(self):
        return '<='
    def call(self, *args):
        return args[0] <= args[1]
class CmpGt(_Comparison):
    def cppname(self):
        return '>'
    def call(self, *args):
        return args[0] > args[1]
class CmpGe(_Comparison):
    def cppname(self):
        return '>='
    def call(self, *args):
        return args[0] >= args[1]


# replacement; however it'll execute both code paths, regardless of the result
class Ternary(Operation):
    ARITY = 3

    def callstr(self, *args):
        return f'(({args[2]}) ? ({args[0]}) : ({args[1]}))'
    def call(self, *args):
        return np.where(args[2], args[0], args[1])
    def datatypeResult(self, argtypes):
        assert argtypes[0] == argtypes[1], \
            f'Both branches of a ternary must agree: {argtypes[0]} vs. {argtypes[1]}'
        return argtypes[0]
    def sparsityResult(self, spps):
        # the condition contributes no value; either branch may be taken
        return _unionSpp(spps[:2])


class Typecast(CFunctionMixin, UnaryArgsMixin, ZeroPreservingMixin, Operation):
    def __init__(self, target: Datatype):
        self.target = target
    def cppname(self):
        return f'static_cast<{self.target.ctype()}>'
    def call(self, *args):
        # NOTE: np.astype() is numpy>=2.0 only
        return np.asarray(args[0]).astype(self.target.nptype())
    def datatypeResult(self, argtypes):
        return self.target
    def _key(self):
        # the target type is part of the identity of a cast
        return (type(self).__name__, self.target)
    def __str__(self):
        return f'Cast<{self.target}>'
