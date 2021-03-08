from .graph import *
from collections import deque

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
          del cfg[i]
          i -= 1
          n -= 1
      i += 1
    return cfg

class LivenessAnalysis(object):
  def visit(self, cfg):
    cfg[-1].live = set()
    for i in reversed(range(len(cfg)-1)):
      cfg[i].live = (cfg[i+1].live - {cfg[i].action.result}) | cfg[i].action.variables()
    return cfg

class SubstituteForward(object):
  def visit(self, cfg):
    n = len(cfg)-1
    for i in range(n):
      ua = cfg[i].action
      v = cfg[i+1]
      if not ua.isCompound() and ua.isRHSVariable() and ua.term.writable and ua.result.isLocal() and ua.term not in v.live:
        when = ua.result
        by = ua.term
        maySubs = all([cfg[j].action.maySubstitute(when, by) for j in range(i, n)])
        if maySubs:
          for j in range(i, n):
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
          if by not in u.live and not u.action.isCompound() and u.action.result == va.term:
            found = j
            break
        if found >= 0:
          when = u.action.result
          maySubs = cfg[found].action.maySubstitute(when, by, term=False) and all([cfg[j].action.maySubstitute(when, by) for j in range(found+1,i+1)])
          if maySubs:
            cfg[found].action = cfg[found].action.substituted(when, by, term=False)
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
        V = ua.variables()
        for j in range(i+1,n):
          va = cfg[j].action
          if va.isRHSVariable() and ua.result == va.term and va.result not in V and (ua.hasTrivialScalar() or va.hasTrivialScalar()):
            found = j
            break
          elif ua.result in va.variables() or ua.result == va.result:
            break
          else:
            V = V | va.variables() | {va.result}
        if found >= 0:
          va = cfg[found].action
          if ua.maySubstitute(ua.result, va.result, term=False):
            cfg[i].action = ua.substituted(ua.result, va.result, term=False)
            cfg[i].action.add = va.add
            if not va.hasTrivialScalar():
              cfg[i].action.scalar = va.scalar
            del cfg[found]
            n -= 1
      i += 1
    return LivenessAnalysis().visit(cfg)

class DetermineLocalInitialization(object):
  def visit(self, cfg):
    lcls = dict()
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
      if ua and not ua.isCompound() and ua.result.isLocal():
        if ua.result in usedBuffers:
            buf = usedBuffers[ua.result]
        elif len(freeBuffers) > 0:
          buf = freeBuffers.pop()
        else:
          buf = numBuffers
          numBuffers += 1
        cfg[i].bufferMap[ua.result] = buf
        usedBuffers[ua.result] = buf

        size = ua.result.memoryLayout().requiredReals()
        if buf in bufferSize:
          bufferSize[buf] = max(bufferSize[buf], size)
        else:
          bufferSize[buf] = size

      # free buffers
      free = cfg[i].live - cfg[i+1].live
      for local in free:
        if local in usedBuffers:
          freeBuffers.appendleft(usedBuffers.pop(local))

    if len(cfg) > 0:
      cfg[0].initBuffer = bufferSize
    return cfg
