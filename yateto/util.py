from . import Tensor
from .ast.transformer import DeduceIndices
from .ast.visitor import ComputeConstantExpression
from numpy import float64, dtype
from .ast.indices import Indices

def tensor_from_constant_expression(name: str, expression, targetIndices: Indices = None, dtype: dtype = float64, tensorArgs: dict = dict()):
  """
  Computes the result of an expression and returns
  an appropriately sized tensor. Works only for expressions
  where are involved tensors and scalars are constant.

  Example:
    tensor = tensor_from_constant_expression('C', A['ik'] * B['kj'])

  Args:
    name: Tensor name
    expression: A valid expression which involves only constant tensors
    targetIndices: The index permutation of the resulting tensor
    dtype: Precision used in computation
    tensorArgs: Additional arguments to Tensor constructor

  Returns:
    A new Tensor object
  """
  expression = DeduceIndices(targetIndices).visit(expression)
  values = ComputeConstantExpression(dtype).visit(expression)
  return Tensor(name, values.shape, spp=values, **tensorArgs)
