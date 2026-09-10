import collections
from .. import aspp
from .. import ops
from ..ast.visitor import Visitor
from ..type import AddressingMode, Tensor, DerivedScalar
from .graph import *
from .transformer import liveness
from ..memory import DenseMemoryLayout
from ..ast.node import Permute, Node, Broadcast

class AST2ControlFlow(Visitor):
  TEMPORARY_RESULT = '_tmp'

  def __init__(self, simpleMemoryLayout=False):
    self._tmp = 0
    self._cfg = []
    self._writable = set()
    self._simpleMemoryLayout = simpleMemoryLayout
    self._guard = [Guard.always()]
    # a condition tensor may be rewritten inside the kernel, so every write
    # starts a new version and guards refer to (variable, version)
    self._version = collections.defaultdict(int)
    # the guard a given version was produced under; reading it is only
    # meaningful where that guard held, so it is conjoined at every use
    self._definitionGuard = dict()
    # name -> (tensor, datatype), so a name collision is reported where it happens
    self._bound = dict()

  def cfg(self):
    return list(self._cfg)

  def _ml(self, node):
    return DenseMemoryLayout(node.shape()) if self._simpleMemoryLayout else node.memoryLayout()

  def _addTransformOp(self, permute, variable):
    if not self._simpleMemoryLayout:
      permute.setEqspp( permute.computeSparsityPattern() )
      permute.computeMemoryLayout()
    permute.datatype = permute[0].datatype
    result = self._nextTemporary(permute)
    action = ProgramAction(result, Expression(permute, self._ml(permute), [variable]), False, condition=self._guard[-1])
    self._addAction(action)
    return result

  def _addPermuteIfRequired(self, indices, term, variable):
    result = variable
    if indices != term.indices:
      # always assume that we write into the _bigger_ output
      # (otherwise, there'd need to be a reduction/IndexSum first)
      assert term.indices <= indices

      order = [idx for idx in indices if idx in term.indices]
      termOrder = [idx for idx in term.indices]

      intermediate = variable
      inode = term
      if order != termOrder:
        # permute needed, run before broadcast
        inode = Permute.subPermute(term, indices)
        intermediate = self._addTransformOp(inode, variable)

      result = intermediate
      if len(term.indices) != len(indices):
        # broadcast needed, more output than input indices
        result = self._addTransformOp(Broadcast(inode, indices), intermediate)

    return result

  def generic_visit(self, node):
    variables = [self.visit(child) for child in node]

    result = self._nextTemporary(node)
    action = ProgramAction(result, Expression(node, self._ml(node), variables), False, condition=self._guard[-1])
    self._addAction(action)

    return result

  def visit_SliceView(self, node):
    var = self.visit(node.term())
    ml = node.getMemoryLayout(var.memoryLayout())
    return VariableView(var, ml, node.eqspp())

  def visit_Accumulate(self, node):
    variables = [self.visit(child) for child in node]
    assert len(variables) >= 1

    assert node.optype == ops.Add(), \
      f'{node} should have been folded into element-wise steps by FoldAccumulate.'

    # A sum becomes a chain of accumulating stores rather than one n-ary
    # operation: that is what lets a GEMM write into the result with beta = 1
    # instead of into a temporary.
    variables.sort(key=lambda var: int(not var.writable) + int(not var.isGlobal()))

    tmp = self._nextTemporary(node)
    add = False
    for i,var in enumerate(variables):
      rhs = self._addPermuteIfRequired(node.indices, node[i], var)
      action = ProgramAction(tmp, rhs, add, condition=self._guard[-1])
      self._addAction(action)
      add = True

    return tmp

  def visit_Elementwise(self, node):
    scaling = node.scalingOperands()
    if scaling is None:
      return self.generic_visit(node)

    # A multiplication by a by-value rank-0 quantity becomes the scale factor of
    # a single action rather than a loop of its own; that is what lets it fold
    # into a GEMM's alpha instead of running as a separate pass.
    scalar, term = scaling
    variable = self.visit(term)

    result = self._nextTemporary(node)
    action = ProgramAction(result, variable, False, scalar, condition=self._guard[-1])
    self._addAction(action)

    return result

  def visit_Assign(self, node):
    outerGuard = self._guard[-1]

    # The condition is evaluated to decide the branch, so it is computed
    # outside the new guard -- before it is pushed.
    if isinstance(node.condition(), Node):
      conditionVar = self.visit(node[2])
      version = self._version[conditionVar.name]
      myGuard = Guard.literal(conditionVar, version) \
                & self._definitionGuard.get((conditionVar.name, version), Guard.always())
    else:
      myGuard = Guard.coerce(node.condition())

    self.updateWritable(node[0].name())

    guard = outerGuard & myGuard

    # The whole right-hand side runs under the guard, not just the final store.
    self._guard.append(guard)
    try:
      rVar = self.visit(node[1])
      rhs = self._addPermuteIfRequired(node.indices, node.rightTerm(), rVar)

      lVar = self.visit(node[0])
      self._addAction(ProgramAction(lVar, rhs, False, condition=guard))
    finally:
      self._guard.pop()

    name = node[0].name()
    self._version[name] += 1
    self._definitionGuard[(name, self._version[name])] = guard

    return lVar

  def visit_IndexedTensor(self, node):
    self._bindName(node.name(), node.tensor, node.datatype)
    return Variable(node.name(), node.name() in self._writable, self._ml(node), node.eqspp(), node.tensor, datatype=node.datatype, is_temporary=node.tensor.temporary)

  def _bindName(self, name, tensor, datatype):
    """One name, one tensor: a name yields one declaration in the signature.

    Checked here because this is where a name is first bound; further down the
    variables are deduplicated by name and the second tensor is no longer
    visible.
    """
    bound, boundType = self._bound.setdefault(name, (tensor, datatype))
    if bound is tensor:
      return
    # the datatype comes from the node: by this point SetDatatype has resolved
    # the ones that were left to the architecture
    for what, mine, theirs in (('shape', tensor.shape(), bound.shape()),
                               ('addressing', tensor.addressing, bound.addressing),
                               ('datatype', datatype, boundType),
                               ('memory layout', tensor.memoryLayout(), bound.memoryLayout())):
      if mine != theirs:
        raise ValueError(
          f'"{name}" is used with two different {what}s ({mine} vs. {theirs}); '
          f'one name yields one declaration.')
    if not aspp.array_equal(tensor.spp(), bound.spp()):
      raise ValueError(f'"{name}" is used with two different sparsity patterns; '
                       f'one name yields one declaration.')

  def visit_IfThenElse(self, node):
    raise NotImplementedError(
      'IfThenElse is not lowered yet; use yateto.functions.where (Elementwise(Ternary)).')

  def _addAction(self, action):
    self._cfg.append(action)

  def _nextTemporary(self, node):
    name = f'{self.TEMPORARY_RESULT}{self._tmp}'
    self._tmp += 1
    return Variable(name, True, self._ml(node), node.eqspp(), is_temporary=True, datatype=node.datatype)

  def updateWritable(self, name):
    self._writable = self._writable | {name}
    # Set variables writable that were added beforehand
    for action in self._cfg:
      action.setVariablesWritable(name)

class SortedGlobalsList(object):
  def visit(self, cfg):
    V = set()
    for action in cfg:
      V = V | action.result.variables() | action.allVariables()
    return sorted([var for var in V if var.isGlobal()], key=lambda x: str(x))

def _scalarsOf(cfg):
  S = set()
  for action in cfg:
    S = S | {scalar for scalar in [action.scalar] if isinstance(scalar, Tensor)}
  return S

class ScalarsSet(object):
  """The scalars the caller sets, i.e. the ones in the kernel signature."""

  def visit(self, cfg):
    scalars = _scalarsOf(cfg)
    # a derived scalar is computed in the prologue, so it also pulls in the
    # named scalars its expression reads
    for derived in [s for s in scalars if isinstance(s, DerivedScalar)]:
      scalars = scalars | derived.dependencies()
    return {scalar for scalar in scalars if not scalar.temporary}

class PrettyPrinter(object):
  def __init__(self, printPPState = False):
    self._printPPState = printPPState

  def visit(self, cfg):
    live = liveness(cfg) if self._printPPState else None
    for position, action in enumerate(cfg):
      if live is not None:
        print('L =', live[position])
      actionRepr = str(action.term)
      if action.scalar is not None:
        actionRepr = str(action.scalar) + ' * ' + actionRepr
      print( '  {} {} {}'.format(action.result, '+=' if action.add else '=', actionRepr) )
