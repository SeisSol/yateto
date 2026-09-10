from .graph import *
from collections import deque
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
