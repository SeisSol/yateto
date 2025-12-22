import numpy as np
from .type import Datatype

class Operation:
    #def callstr(self, *args) -> str:
    #    raise NotImplementedError()
    
    def call(self, *args):
        raise NotImplementedError()
    
    def datatypeResult(self, argtypes):
        raise NotImplementedError() # TODO
    
    def __str__(self):
        return type(self).__name__
    
    def __eq__(self, other):
        # we're more or less using "dummy" types here
        return type(self).__name__ == type(other).__name__

class CommutativeMonoidMixin:
    def neutral(self):
        pass

class RingMixin:
    def formsRing(self, op):
        pass

class UnaryArgsMixin:
    pass

class BinaryArgsMixin:
    pass

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

class Sin(Operation, CFunctionMixin, UnaryArgsMixin):
    def cppname(self):
        return 'std::sin'
    def call(self, *args):
        return np.sin(args[0])
    def datatypeResult(self, argtypes):
        return argtypes[0]
class Cos(Operation, CFunctionMixin, UnaryArgsMixin):
    def cppname(self):
        return 'std::cos'
    def call(self, *args):
        return np.cos(args[0])
    def datatypeResult(self, argtypes):
        return argtypes[0]
class Tan(Operation, CFunctionMixin, UnaryArgsMixin):
    def cppname(self):
        return 'std::tan'
    def call(self, *args):
        return np.tan(args[0])
    def datatypeResult(self, argtypes):
        return argtypes[0]
class Asin(Operation, CFunctionMixin, UnaryArgsMixin):
    def cppname(self):
        return 'std::asin'
    def call(self, *args):
        return np.asin(args[0])
    def datatypeResult(self, argtypes):
        return argtypes[0]
class Acos(Operation, CFunctionMixin, UnaryArgsMixin):
    def cppname(self):
        return 'std::acos'
    def call(self, *args):
        return np.acos(args[0])
    def datatypeResult(self, argtypes):
        return argtypes[0]
class Atan(Operation, CFunctionMixin, UnaryArgsMixin):
    def cppname(self):
        return 'std::atan'
    def call(self, *args):
        return np.atan(args[0])
    def datatypeResult(self, argtypes):
        return argtypes[0]

class Sinh(Operation, CFunctionMixin, UnaryArgsMixin):
    def cppname(self):
        return 'std::sinh'
    def call(self, *args):
        return np.sinh(args[0])
    def datatypeResult(self, argtypes):
        return argtypes[0]
class Cosh(Operation, CFunctionMixin, UnaryArgsMixin):
    def cppname(self):
        return 'std::cosh'
    def call(self, *args):
        return np.cosh(args[0])
    def datatypeResult(self, argtypes):
        return argtypes[0]
class Tanh(Operation, CFunctionMixin, UnaryArgsMixin):
    def cppname(self):
        return 'std::tanh'
    def call(self, *args):
        return np.tanh(args[0])
    def datatypeResult(self, argtypes):
        return argtypes[0]
class Asinh(Operation, CFunctionMixin, UnaryArgsMixin):
    def cppname(self):
        return 'std::asinh'
    def call(self, *args):
        return np.asinh(args[0])
    def datatypeResult(self, argtypes):
        return argtypes[0]
class Acosh(Operation, CFunctionMixin, UnaryArgsMixin):
    def cppname(self):
        return 'std::acosh'
    def call(self, *args):
        return np.acosh(args[0])
    def datatypeResult(self, argtypes):
        return argtypes[0]
class Atanh(Operation, CFunctionMixin, UnaryArgsMixin):
    def cppname(self):
        return 'std::atanh'
    def call(self, *args):
        return np.atanh(args[0])
    def datatypeResult(self, argtypes):
        return argtypes[0]

class Log(Operation, CFunctionMixin, UnaryArgsMixin):
    def cppname(self):
        return 'std::log'
    def call(self, *args):
        return np.log(args[0])
    def datatypeResult(self, argtypes):
        return argtypes[0]
class Exp(Operation, CFunctionMixin, UnaryArgsMixin):
    def cppname(self):
        return 'std::exp'
    def call(self, *args):
        return np.exp(args[0])
    def datatypeResult(self, argtypes):
        return argtypes[0]
class Log1p(Operation, CFunctionMixin, UnaryArgsMixin):
    def cppname(self):
        return 'std::log1p'
    def call(self, *args):
        return np.log1p(args[0])
    def datatypeResult(self, argtypes):
        return argtypes[0]
class Expm1(Operation, CFunctionMixin, UnaryArgsMixin):
    def cppname(self):
        return 'std::expm1'
    def call(self, *args):
        return np.expm1(args[0])
    def datatypeResult(self, argtypes):
        return argtypes[0]
class Sqrt(Operation, CFunctionMixin, UnaryArgsMixin):
    def cppname(self):
        return 'std::sqrt'
    def call(self, *args):
        return np.sqrt(args[0])
    def datatypeResult(self, argtypes):
        return argtypes[0]
class Cbrt(Operation, CFunctionMixin, UnaryArgsMixin):
    def cppname(self):
        return 'std::cbrt'
    def call(self, *args):
        return np.cbrt(args[0])
    def datatypeResult(self, argtypes):
        return argtypes[0]

class Abs(Operation, CFunctionMixin, UnaryArgsMixin):
    def cppname(self):
        return 'std::abs'
    def call(self, *args):
        return np.abs(args[0])
    def datatypeResult(self, argtypes):
        return argtypes[0]


class Max(Operation, CFunctionMixin, BinaryArgsMixin, CommutativeMonoidMixin):
    def neutral(self):
        return -float('inf')
    def cppname(self):
        return 'std::max'
    def call(self, *args):
        return max(args[0], args[1])
    def datatypeResult(self, argtypes):
        # assert argtypes[0] == argtypes[1]
        return argtypes[0]
class Min(Operation, CFunctionMixin, BinaryArgsMixin, CommutativeMonoidMixin):
    def neutral(self):
        return float('inf')
    def cppname(self):
        return 'std::min'
    def call(self, *args):
        return min(args[0], args[1])
    def datatypeResult(self, argtypes):
        # assert argtypes[0] == argtypes[1]
        return argtypes[0]
class Pow(Operation, CFunctionMixin, BinaryArgsMixin):
    def cppname(self):
        return 'std::pow'
    def call(self, *args):
        return pow(args[0], args[1])
    def datatypeResult(self, argtypes):
        return argtypes[0]

class Div(Operation, CBinaryOperatorMixin, BinaryArgsMixin):
    def cppname(self, *args):
        return '/'
    def call(self, *args):
        return args[0] / args[1]
    def datatypeResult(self, argtypes):
        # assert argtypes[0] == argtypes[1]
        return argtypes[0]

class Add(Operation, CBinaryOperatorMixin, BinaryArgsMixin, CommutativeMonoidMixin):
    def cppname(self, *args):
        return '+'
    def call(self, *args):
        return args[0] + args[1]
    def neutral(self):
        return 0
    def datatypeResult(self, argtypes):
        # assert argtypes[0] == argtypes[1]
        return argtypes[0]
class Mul(Operation, CBinaryOperatorMixin, BinaryArgsMixin, CommutativeMonoidMixin, RingMixin):
    def cppname(self, *args):
        return '*'
    def call(self, *args):
        return args[0] * args[1]
    def neutral(self):
        return 1
    def formsRing(self, op):
        return op == Add()
    def datatypeResult(self, argtypes):
        # assert argtypes[0] == argtypes[1]
        return argtypes[0]

class And(Operation, CBinaryOperatorMixin, BinaryArgsMixin, CommutativeMonoidMixin, RingMixin):
    def cppname(self, *args):
        return '&'
    def call(self, *args):
        return args[0] & args[1]
    def neutral(self):
        return True
    def formsRing(self, op):
        return op == Or()
    def datatypeResult(self, argtypes):
        # assert argtypes[0] == argtypes[1]
        return argtypes[0]
class Or(Operation, CBinaryOperatorMixin, BinaryArgsMixin, CommutativeMonoidMixin, RingMixin):
    def cppname(self, *args):
        return '|'
    def call(self, *args):
        return args[0] | args[1]
    def neutral(self):
        return False
    def formsRing(self, op):
        return op == And()
    def datatypeResult(self, argtypes):
        # assert argtypes[0] == argtypes[1]
        return argtypes[0]
class Not(Operation, CUnaryOperatorMixin, UnaryArgsMixin):
    def cppname(self, *args):
        return '~'
    def call(self, *args):
        return ~args[0]
    def datatypeResult(self, argtypes):
        # assert argtypes[0] == argtypes[1]
        return argtypes[0]

class CmpEq(Operation, CBinaryOperatorMixin, BinaryArgsMixin):
    def cppname(self, *args):
        return '=='
    def call(self, *args):
        return args[0] == args[1]
    def datatypeResult(self, argtypes):
        return Datatype.BOOL
class CmpNe(Operation, CBinaryOperatorMixin, BinaryArgsMixin):
    def cppname(self, *args):
        return '!='
    def call(self, *args):
        return args[0] != args[1]
    def datatypeResult(self, argtypes):
        return Datatype.BOOL
class CmpLt(Operation, CBinaryOperatorMixin, BinaryArgsMixin):
    def cppname(self, *args):
        return '<'
    def call(self, *args):
        return args[0] < args[1]
    def datatypeResult(self, argtypes):
        return Datatype.BOOL
class CmpLe(Operation, CBinaryOperatorMixin, BinaryArgsMixin):
    def cppname(self, *args):
        return '<='
    def call(self, *args):
        return args[0] <= args[1]
    def datatypeResult(self, argtypes):
        return Datatype.BOOL
class CmpGt(Operation, CBinaryOperatorMixin, BinaryArgsMixin):
    def cppname(self, *args):
        return '>'
    def call(self, *args):
        return args[0] > args[1]
    def datatypeResult(self, argtypes):
        return Datatype.BOOL
class CmpGe(Operation, CBinaryOperatorMixin, BinaryArgsMixin):
    def cppname(self, *args):
        return '>='
    def call(self, *args):
        return args[0] >= args[1]
    def datatypeResult(self, argtypes):
        return Datatype.BOOL

# replacement; however it'll execute both code paths, regardless of the result
class Ternary(Operation):
    def callstr(self, *args):
        return f'(({args[2]}) ? ({args[0]}) : ({args[1]}))'
    def call(self, *args):
        return np.where(args[2], args[0], args[1])
    def datatypeResult(self, argtypes):
        assert argtypes[0] == argtypes[1]
        return argtypes[0]
class Typecast(Operation, CFunctionMixin, UnaryArgsMixin):
    def __init__(self, target: Datatype):
        self.target = target
    def cppname(self, *args):
        return f'static_cast<{self.target.ctype()}>'
    def call(self, *args):
        return np.astype(args[0], self.target.nptype())
    def datatypeResult(self, argtypes):
        return self.target
    def __str__(self):
        return f'Cast<{self.target}>'
