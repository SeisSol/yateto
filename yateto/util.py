from . import Tensor, Collection
from .ast.transformer import DeduceIndices
from .ast.visitor import ComputeConstantExpression
from numpy import float64, dtype
from .ast.indices import Indices

import numpy as np

def create_collection(matrices):
  maxIndex = dict()
  collection = Collection()
  for name, matrix in matrices.items():
    if not Tensor.isValidName(name):
      raise ValueError('Illegal matrix name', name)
    baseName = Tensor.getBaseName(name)
    group = Collection.group(name)
    if group is tuple():
      collection[baseName] = matrix
    else:
      if baseName in collection:
        collection[baseName][group] = matrix
      else:
        collection[baseName] = {group: matrix}

  return collection

def tensor_from_constant_expression(name: str,
                                    expression,
                                    target_indices: Indices = None,
                                    dtype: dtype = np.float128,
                                    tensor_args: dict = dict()):
  """
  Computes the result of an expression and returns
  an appropriately sized tensor. Works only for expressions
  where all involved tensors and scalars are constant.

  Example:
    tensor = tensor_from_constant_expression('C', A['ik'] * B['kj'])

  Args:
    name: Tensor name
    expression: A valid expression which involves only constant tensors
    target_indices: The index permutation of the resulting tensor
    dtype: Precision used in computation
    tensor_args: Additional arguments to Tensor constructor

  Returns:
    A new Tensor object
  """
  expression = DeduceIndices(target_indices).visit(expression)
  values = ComputeConstantExpression(dtype).visit(expression)
  return Tensor(name, values.shape, spp=values, **tensor_args)

def tensor_collection_from_constant_expression(base_name: str,
                                               expressions,
                                               group_indices,
                                               target_indices: Indices = None,
                                               dtype=np.float128,
                                               tensor_args: dict = {}):
  """
  Computes the result of an expression group and returns
  a group of appropriately sized tensors. Works only for expressions
  where all involved tensors and scalars are constant.

  Example:
    tensor = tensor_collection_from_constant_expression('C', lambda a: A[a]['ik'] * B['kj'], simpleParameterSpace(2))

  Args:
    base_name: Tensor base name
    expressions: A lambda that takes a group index and returns an expression consisting in constant tensors
    group_indices: All valid indices for which the groups are defined. Supposed to be an iterator which returns tuples (such as a YATeTo parameter space)
    target_indices: The index permutation of the resulting tensor
    dtype: Precision used in computation
    tensor_args: Additional arguments to Tensor constructor

  Returns:
    A new Tensor collection
  """
  tensors = {}
  for idx in group_indices:
    expression = expressions(*idx)
    name = "{}({})".format(base_name, ','.join(str(i) for i in idx))
    tensor = tensor_from_constant_expression(name=name,
                                             expression=expression,
                                             target_indices=target_indices,
                                             dtype=dtype,
                                             tensor_args=tensor_args)
    tensors[name] = tensor

  return create_collection(tensors)
