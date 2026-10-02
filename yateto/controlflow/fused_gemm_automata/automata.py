from ...ast.node import LoopOverGEMM
from ..graph import FusedActions, FusedProgramPoint, VariableView
from typing import List, Set, Union


def readsAfter(cfg) -> List[Set[str]]:
  """Per program point, the names of the variables the ones after it read."""
  reads = [set() for _ in cfg]
  seen = set()
  for i in reversed(range(len(cfg))):
    reads[i] = set(seen)
    action = cfg[i].action
    if action:
      seen |= {var.viewed().name for var in action.allVariables()}
  return reads


class Context:
  """Groups consecutive GEMMs into chains, each generated as one kernel.

  What a chain may be is what the fused-GEMM generator can run as one kernel.
  chainforge keeps the temporaries of a chain in the kernel's own memory, and
  only there, and hands the kernel the scale factors of its last GEMM alone:

  * a temporary a chain reads is one it wrote -- one of an earlier statement
    is not in the kernel's memory -- and one it writes is one nothing after
    the chain reads, since it never leaves that memory. Nor does a GEMM of
    the chain accumulate into a temporary: the generator makes a new one for
    every GEMM that writes one.
  * only its last GEMM is scaled by a factor the kernel is handed at run time.
  * no operand is a window into a tensor: the generator would take it for a
    matrix of its own, and no longer see that it is the tensor the chain
    writes or reads elsewhere.

  A GEMM that cannot be in a chain is a GEMM of its own. Where the temporaries
  a chain writes are read after it, the chain loses GEMMs from its end until
  none are.
  """

  def __init__(self, reads: Union[List[Set[str]], None] = None):
    self._reads = reads
    self._position = 0
    self._chain = []
    self._cfg = []

  def process(self, program_point):
    position = self._position
    self._position += 1
    if not self._isGemm(program_point):
      self._close()
      self._cfg.append(program_point)
      return
    action = program_point.action
    if self._chain and not self._mayJoin(action):
      self._close()
    if not self._chain and not self._mayStart(action):
      self._cfg.append(program_point)
      return
    self._chain.append((position, program_point))

  def get_cfg(self):
    self._close()
    return self._cfg

  @classmethod
  def get_finite_automata(cls, reads=None):
    return Context(reads)

  @staticmethod
  def _isGemm(program_point) -> bool:
    action = program_point.action
    if not (action and action.isRHSExpression()):
      return False
    node = action.term.node
    if not (isinstance(node, LoopOverGEMM) and node.is_pure_gemm()):
      return False
    variables = [action.result] + list(action.term.variableList())
    return not any(isinstance(var, VariableView) for var in variables)

  @staticmethod
  def _temporariesRead(action):
    return {var.name for var in action.variables() if var.is_temporary}

  @staticmethod
  def _accumulatesIntoTemporary(action):
    return action.add and action.result.is_temporary

  def _mayStart(self, action):
    return not self._temporariesRead(action) and not self._accumulatesIntoTemporary(action)

  def _mayJoin(self, action):
    last = self._chain[-1][1].action
    if last.scalar is not None and not isinstance(last.scalar, (int, float)):
      return False
    if self._accumulatesIntoTemporary(action):
      return False
    written = {pp.action.result.viewed().name for _, pp in self._chain}
    return self._temporariesRead(action) <= written

  def _escapes(self, chain):
    if self._reads is None:
      return False
    position = chain[-1][0]
    written = {pp.action.result.viewed().name for _, pp in chain
               if pp.action.result.is_temporary}
    return bool(written & self._reads[position])

  def _close(self):
    chain, self._chain = self._chain, []
    alone = []
    while chain and self._escapes(chain):
      alone.insert(0, chain.pop())
    if chain:
      fused = FusedActions()
      for _, program_point in chain:
        fused.add(program_point.action)
      self._cfg.append(FusedProgramPoint(fused))
    self._cfg.extend(program_point for _, program_point in alone)
