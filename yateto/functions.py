from . import ops
from .ast import node
from .type import Datatype

def sin(x): return node.Elementwise(ops.Sin(), x)
def cos(x): return node.Elementwise(ops.Cos(), x)
def tan(x): return node.Elementwise(ops.Tan(), x)
def asin(x): return node.Elementwise(ops.Asin(), x)
def acos(x): return node.Elementwise(ops.Acos(), x)
def atan(x): return node.Elementwise(ops.Atan(), x)

def sinh(x): return node.Elementwise(ops.Sinh(), x)
def cosh(x): return node.Elementwise(ops.Cosh(), x)
def tanh(x): return node.Elementwise(ops.Tanh(), x)
def asinh(x): return node.Elementwise(ops.Asinh(), x)
def acosh(x): return node.Elementwise(ops.Acosh(), x)
def atanh(x): return node.Elementwise(ops.Atanh(), x)

def log(x): return node.Elementwise(ops.Log(), x)
def log1p(x): return node.Elementwise(ops.Log1p(), x)
def exp(x): return node.Elementwise(ops.Exp(), x)
def expm1(x): return node.Elementwise(ops.Expm1(), x)
def sqrt(x): return node.Elementwise(ops.Sqrt(), x)
def cbrt(x): return node.Elementwise(ops.Cbrt(), x)

def abs(x): return node.Elementwise(ops.Abs(), x)

def max(x, y): return node.Elementwise(ops.Max(), x, y)
def min(x, y): return node.Elementwise(ops.Min(), x, y)
def pow(x, y): return node.Elementwise(ops.Pow(), x, y)

def assign(lhs, rhs): return node.Assign(lhs, rhs)
def assignIf(condition, lhs, rhs): return node.Assign(lhs, rhs, condition)

# def where(condition, yes, no): return node.IfThenElse(condition, yes, no)
def where(condition, yes, no): return node.Elementwise(ops.Ternary(), yes, no, condition)

def equal(x, y): return node.Elementwise(ops.CmpEq(), x, y)
def not_equal(x, y): return node.Elementwise(ops.CmpNe(), x, y)
def less(x, y): return node.Elementwise(ops.CmpLt(), x, y)
def less_equal(x, y): return node.Elementwise(ops.CmpLe(), x, y)
def greater(x, y): return node.Elementwise(ops.CmpGt(), x, y)
def greater_equal(x, y): return node.Elementwise(ops.CmpGe(), x, y)

# extra reduction functions; e.g. for input to `where`
def reduction(op, term, indices):
    if len(indices) == 0:
        return term
    else:
        reduction(op, node.Reduction(op, term, indices[0]), indices[1:])

def sum(term, indices): return reduction(ops.Add(), term, indices)
def product(term, indices): return reduction(ops.Mul(), term, indices)
def all(term, indices): return reduction(ops.And(), term, indices)
def any(term, indices): return reduction(ops.Or(), term, indices)

def cast(x, dtype): return node.Elementwise(ops.Typecast(dtype), x)
