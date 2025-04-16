from . import ops
from .ast import node

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

# extra reduction functions; e.g. for input to `where`
def reductionSum(term, indices): return node.Reduction(ops.Add(), term, indices)
def reductionMul(term, indices): return node.Reduction(ops.Mul(), term, indices)ass
def reductionAnd(term, indices): return node.Reduction(ops.And(), term, indices)
def reductionOr(term, indices): return node.Reduction(ops.Or(), term, indices)
