from .node import Op, Contract, Assign, Add, Indices, IndexedTensor
from functools import singledispatch
from numpy import zeros, einsum

def addIndent(string, indent):
  return '\n'.join([indent + line for line in string.splitlines()])

@singledispatch
def pprint(node, indent=''):
  print(indent + str(node))

@pprint.register(Op)
def _(node, indent=''):
  print(indent + str(node))
  for child in node:
    pprint(child, indent + '  ')

@singledispatch
def simplify(node):
  pass

@simplify.register(Op)
def _(node):
  newChildren = []
  for child in node:
    simplify(child)
    if isinstance(child, type(node)):
      newChildren.extend(list(child))
    else:
      newChildren.append(child)
  node.setChildren(newChildren)

@singledispatch
def evaluate(node):
  pass
  
@evaluate.register(Op)
def _(node):
  raise NotImplementedError()

@evaluate.register(Contract)
def _(node):
  for child in node:
    evaluate(child)
  
  g = Indices()
  contractions = set()
  for child in node:
    overlap = g & child.indices
    if any([g.size()[index] != child.size()[index] for index in overlap]):
      pprint(node)
      raise ValueError('Contract: Index dimensions do not match: ', g, child.indices, str(child))
    g = g.merged(child.indices - overlap)
    contractions.update(overlap)

  deduced = g - contractions
  if node.indices == None:
    node.indices = deduced.sorted()
  elif not node.indices <= deduced:
    raise ValueError('Contract: Indices are not contained in deduced indices or sizes do not match. [{} not contained in {}]'.format(node.indices.__repr__(), deduced.__repr__()))

@evaluate.register(Add)
def _(node):
  for child in node:
    if isinstance(child, Op):
      child.indices = node.indices
    evaluate(child)

  ok = all([node[0].indices == child.indices for child in node])
  if not ok:
    raise ValueError('Add: Indices do not match: ', *[child.indices for child in node])

  if node.indices == None:
    node.indices = node[0].indices
  elif node.indices != node[0].indices:
    raise ValueError('Add: {} is not a equal to {}'.format(node.indices.__repr__(), node[0].indices.__repr__()))

@evaluate.register(Assign)
def _(node):
  lhs = node[0]
  rhs = node[1]
  
  if not isinstance(lhs, IndexedTensor):
    raise ValueError('Assign: Left-hand side must be of type IndexedTensor')

  node.indices = lhs.indices

  if isinstance(rhs, Op):
    rhs.indices = node.indices
  elif rhs.indices != node.indices:
    raise ValueError('Assign: Index dimensions do not match: {} != {}'.format(node.indices.__repr__(), rhs.indices.__repr__()))
  evaluate(rhs)

@singledispatch
def equivalentSparsityPattern(node):
  pass

@equivalentSparsityPattern.register(IndexedTensor)  
def _(node):
  node.setEqspp(node.spp().copy())

@equivalentSparsityPattern.register(Contract)
def _(node):
  for child in node:
    equivalentSparsityPattern(child)

  spps = [child.eqspp() for child in node]
  indices = ','.join([child.indices.tostring() for child in node])
  node.setEqspp( einsum('{}->{}'.format(indices, node.indices.tostring()), *spps, optimize=True) )
  
  for child in node:
    child.setEqspp( einsum('{}->{}'.format(indices, child.indices.tostring()), *spps, optimize=True) )
  
  # TODO: Backtracking of equivalent sparsity pattern to children?
    

@equivalentSparsityPattern.register(Add)
def _(node):
  for child in node:
    equivalentSparsityPattern(child)
  
  spp = zeros(node.indices.shape(), dtype=bool)
  for child in node:
    spp += child.eqspp()
  node.setEqspp(spp)

@equivalentSparsityPattern.register(Assign)
def _(node):
  for child in node:
    equivalentSparsityPattern(child)
  node.setEqspp(child[1].eqspp())
