from .graph import *
from collections import deque
from ..ast.node import Elementwise, FusedElementwise, LoopOverGEMM, Reduction
from ..ast.indices import BoundingBox, Indices
import string
from .fused_gemm_automata import Context as FusedGemmsContext


class MergeScalarMultiplications(object):
  def visit(self, cfg):
    n = len(cfg)-1
    i = 1
    while i < n:
      ua = cfg[i].action
      if ua.isRHSVariable() and not ua.isCompound() and ua.scalar is not None:
        va = cfg[i-1].action
        if va.isRHSExpression() and not va.isCompound() and ua.term == va.result:
          va.scalar = ua.scalar
          va.result = ua.result
          # the merged action now performs ua's store, so it inherits ua's guard
          va.condition = va.getGuard() & ua.getGuard()
          del cfg[i]
          i -= 1
          n -= 1
      i += 1
    return cfg

class LivenessAnalysis(object):
  def visit(self, cfg):
    cfg[-1].live = LiveSet({})
    for i in reversed(range(len(cfg)-1)):
      action = cfg[i].action
      guard = action.getGuard()
      live = cfg[i+1].live - {action.result: guard}
      live = live | {var: guard for var in action.variables()}
      # the guard has to be read to decide the branch, so its variables are
      # live regardless of the outcome
      live = live | {var: Guard.always() for var in action.guardVariables()}
      cfg[i].live = live
    return cfg

def _guardsCompatible(cfg, rng, definition):
  """Every touched action must run at least as restrictively as `definition`.

  Otherwise the substituted variable may be read where it was never written.
  """
  return all(cfg[j].action.getGuard().implies(definition) for j in rng)

class SubstituteForward(object):
  def visit(self, cfg):
    n = len(cfg)-1
    for i in range(n):
      ua = cfg[i].action
      v = cfg[i+1]

      if not ua.isCompound() \
          and ua.isRHSVariable() \
          and ua.term.writable \
          and ua.result.isLocal() \
          and (ua.term, ua.getGuard()) not in v.live \
          and (ua.hasTrivialScalar() or ua.term.isLocal()):

        when = ua.result
        by = ua.term
        maySubs = all([cfg[j].action.maySubstitute(when, by) for j in range(i, n)]) \
                  and _guardsCompatible(cfg, range(i, n), ua.getGuard())
        if maySubs:
          for j in range(i, n):
            # a read substitution; the downstream guards stay as they are
            cfg[j].action = cfg[j].action.substituted(when, by)
          cfg = LivenessAnalysis().visit(cfg)

    return cfg

class SubstituteBackward(object):
  def visit(self, cfg):
    n = len(cfg)-1
    for i in reversed(range(n)):
      va = cfg[i].action
      if not va.isCompound() and va.isRHSVariable() and va.term.isLocal():
        by = va.result
        found = -1
        for j in range(i):
          u = cfg[j]
          if (by, va.getGuard()) not in u.live and not u.action.isCompound() and u.action.result == va.term:
            found = j
            break
        if found >= 0:
          when = cfg[found].action.result
          maySubs = cfg[found].action.maySubstitute(when, by, term=False) \
                    and all([cfg[j].action.maySubstitute(when, by) for j in range(found+1,i+1)]) \
                    and _guardsCompatible(cfg, range(found, i+1), va.getGuard())
          if maySubs:
            # only the producing action changes its write target and hence
            # inherits va's guard; the remaining ones merely read `by`
            cfg[found].action = cfg[found].action.substituted(when, by, va.getGuard(), term=False)
            for j in range(found+1,i+1):
              cfg[j].action = cfg[j].action.substituted(when, by)
            cfg = LivenessAnalysis().visit(cfg)
    return cfg

class RemoveEmptyStatements(object):
  def visit(self, cfg):
    n = len(cfg)-1
    i = 0
    while i < n:
      ua = cfg[i].action
      if not ua.isCompound() and ua.isRHSVariable() and ua.result == ua.term and ua.hasTrivialScalar():
        del cfg[i]
        n -= 1
      else:
        i += 1
    return cfg

class MergeActions(object):
  def visit(self, cfg):
    n = len(cfg)-1
    i = 0
    while i < n:
      ua = cfg[i].action
      if not ua.isCompound():
        found = -1
        V = ua.allVariables()
        for j in range(i+1,n):
          va = cfg[j].action
          if va.isRHSVariable() \
              and ua.result == va.term \
              and va.result not in V \
              and (ua.hasTrivialScalar() or va.hasTrivialScalar()) \
              and ua.result.isLocal():
            found = j
            break
          elif ua.result in va.allVariables() or ua.result == va.result:
            break
          else:
            V = V | va.allVariables() | {va.result}
        if found >= 0:
          va = cfg[found].action
          if ua.maySubstitute(ua.result, va.result, term=False):
            # this action's write target becomes va's, so it inherits va's guard
            cfg[i].action = ua.substituted(ua.result, va.result, va.getGuard(), term=False)
            cfg[i].action.add = va.add
            if not va.hasTrivialScalar():
              cfg[i].action.scalar = va.scalar
            del cfg[found]
            n -= 1
      i += 1
    return LivenessAnalysis().visit(cfg)

class DetermineLocalInitialization(object):
  def visit(self, cfg):
    numBuffers = 0
    usedBuffers = dict()
    freeBuffers = deque()
    bufferSize = dict()

    for pp in cfg:
      pp.initBuffer = dict()
      pp.bufferMap = dict()

    n = len(cfg)
    for i in range(n-1):
      ua = cfg[i].action
      # assign buffer
      if ua and not ua.isCompound() and not ua.result.isGlobal():
        if ua.result in usedBuffers:
          buf = usedBuffers[ua.result]
        elif len(freeBuffers) > 0:
          buf = freeBuffers.pop()
        else:
          buf = numBuffers
          numBuffers += 1
        cfg[i].bufferMap[ua.result] = buf
        usedBuffers[ua.result] = buf

        # size in bytes
        datatype = ua.result.datatype
        assert datatype is not None, \
          f'No datatype deduced for {ua.result}; run SetDatatype before the code generator.'
        size = ua.result.viewed().memoryLayout().storage().requiredReals() * datatype.size()
        if buf in bufferSize:
          bufferSize[buf] = max(bufferSize[buf], size)
        else:
          bufferSize[buf] = size

      # free buffers
      free = cfg[i].live.variables() - cfg[i+1].live.variables()
      for local in free:
        # warning: local.isLocal() check is suboptimal (but currently good enough)
        # refactor liveness for better results
        if local.isLocal():
          if local in usedBuffers:
            freeBuffers.appendleft(usedBuffers.pop(local))

    if len(cfg) > 0:
      cfg[0].initBuffer = bufferSize
    return cfg


class FindFusedElementwise(object):
  """Puts adjacent element-wise steps into one loop nest.

  Each of them walks the same index space, and today each walks it on its own
  and leaves its result in a buffer for the next to read. Fusing them turns
  those buffers into locals: measured on a plasticity-shaped chain of nine
  steps over 56x6, 1150 ns becomes 355 on both gcc and clang.

  A scaling counts as a step. It is an element-wise multiplication seen from
  the control-flow graph, where it is an action's factor rather than a node,
  and leaving it out breaks a group in two wherever a kernel scales something
  in the middle of a chain -- which is most of them.

  Deliberately strict. A group is only formed where every step writes the same
  index space under the same guard, and where each intermediate is local,
  written once and read only inside the group.
  """

  @staticmethod
  def _box(action):
    """The index space the action writes, or None if it is not a candidate."""
    if action is None:
      return None
    result = action.result
    if result.memoryLayout().isSparse() or result.eqspp() is None:
      return None
    box = str(BoundingBox.fromSpp(result.eqspp()))
    if action.isRHSExpression():
      node = action.term.node
      if node.eqspp() is None:
        return None
      if isinstance(node, Reduction):
        # its operand walks the nest's space and one axis more, which the nest
        # opens for it; the box check below is about the nest's axes, so it is
        # made against the operand seen through the result's indices
        if node.term().memoryLayout().isSparse():
          return None
        return (str(node.indices), box)
      if not isinstance(node, Elementwise):
        return None
      operands = action.term.variableList()
      # the loop variables come from these names, so two steps that spell the
      # same space differently cannot share a nest
      names = str(node.indices)
    else:
      operands = [action.term]
      # a copy takes the loop variables the code generator hands it, so it fits
      # whatever the group is already spelling
      names = None
    for term in operands:
      if term.memoryLayout().isSparse() or term.eqspp() is None:
        return None
      # Every operand is read over the nest's range, so it has to be the range
      # the operand has. An operand that is narrower would be read outside what
      # it stores, and one that is broadcast over an axis the result has is not
      # addressed the same way; the unfused backend asserts as much per action,
      # and here there is no per-action range left to assert against.
      if str(BoundingBox.fromSpp(term.eqspp())) != box:
        return None
    return (names, box)

  @staticmethod
  def _agree(a, b):
    return a is not None and b is not None and a[1] == b[1] and (
        a[0] is None or b[0] is None or a[0] == b[0])

  def _reaches(self, cfg, first, next):
    """Whether the action at `next` could share a nest with the one at `first`.

    It does not have to read anything the group produced. Every step touches
    one element of the index space per iteration and no other, so putting two
    of them in one nest keeps the order they had for every element -- whether
    or not they have anything to do with each other.
    """
    return (self._agree(self._box(cfg[first].action), self._box(cfg[next].action))
            and cfg[first].action.getGuard() == cfg[next].action.getGuard()
            and not cfg[next - 1].action.add
            # NOTE: not isLocal(), which is true only of an anonymous
            #       temporary. A named one -- SeisSol's `Iprev`, `IAcc` -- is
            #       neither local nor global by that pair of predicates, and it
            #       is a buffer like any other: nobody outside the kernel sees
            #       it, so it may become a variable of the loop body.
            and not cfg[next - 1].action.result.isGlobal())

  @staticmethod
  def _contained(cfg, first, last):
    """Whether everything the group writes before its last step stays inside.

    Only then does a buffer become a variable of the loop body. Asked of the
    whole group rather than of each extension: the first thing a plasticity
    kernel computes is often read by the last thing it does, so a group that
    stopped at the first result with a later reader would never grow past it.
    """
    return all((cfg[k].action.result, cfg[k].action.getGuard()) not in cfg[last + 1].live
               for k in range(first, last))

  @staticmethod
  def _names(actions):
    """The index names the nest runs over.

    Taken from whichever step spells them; a group of copies alone gets the
    names the code generator would have given them.
    """
    for action in actions:
      box = FindFusedElementwise._box(action)
      if box[0] is not None:
        return box[0]
    return string.ascii_lowercase[:len(actions[0].result.memoryLayout().shape())]

  def visit(self, cfg):
    i = 0
    while i + 1 < len(cfg) - 1:
      # as far as the index space and the guard reach ...
      m = i
      while m + 1 < len(cfg) - 1 and self._reaches(cfg, i, m + 1):
        m += 1
      # ... and then back until nothing the group writes is read outside it
      j = m
      while j > i and not self._contained(cfg, i, j):
        j -= 1
      members = [cfg[k].action for k in range(i, j + 1)]
      if len(members) > 1:
        last = members[-1]
        shape = last.result.memoryLayout().shape()
        names = self._names(members)
        steps = []
        outside = []
        produced = {}
        for k, action in enumerate(members):
          if action.isRHSExpression() and isinstance(action.term.node, Reduction):
            step = FusedElementwise.Step.fromReduction(action.term.node)
            operands = action.term.variableList()
          elif action.isRHSExpression():
            step = FusedElementwise.Step.fromElementwise(action.term.node)
            operands = action.term.variableList()
          else:
            step = FusedElementwise.Step.scaling(action.scalar, action.add)
            operands = [action.term]
          # per operand: the step that wrote it, or None if it comes from
          # outside the group and the code generator reads it as usual
          step.sources = [produced.get(variable) for variable in operands]
          outside.extend(variable for variable, source in zip(operands, step.sources)
                         if source is None)
          produced[action.result] = k
          steps.append(step)
        node = FusedElementwise(steps, Indices(names, shape))
        node.setEqspp(last.result.eqspp())
        node.setMemoryLayout(last.result.memoryLayout())
        node.datatype = last.result.datatype
        cfg[i].action = ProgramAction(
          last.result,
          Expression(node, last.result.memoryLayout(), outside),
          last.add, last.scalar if last.isRHSExpression() else None, last.condition)
        del cfg[i+1:j+1]
      i += 1
    return LivenessAnalysis().visit(cfg)


class FindFusedGemms(object):
  def visit(self, cfg):
    context = FusedGemmsContext.get_finite_automata()
    try:
      for pp in cfg:
        context.process(pp)
      cfg = context.get_cfg()
    except Exception as err:
      print(f'Warning: {err}')
    return cfg
