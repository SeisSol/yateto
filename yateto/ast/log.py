import sys
import functools

@functools.total_ordering
class Cost(object):    
  def __init__(self, transpose = sys.maxsize, fusedIndices = 0):
    self._transpose = transpose
    self._fusedIndices = fusedIndices
  
  def __lt__(self, other):
    return self._transpose < other._transpose or (self._transpose == other._transpose and self._fusedIndices > other._fusedIndices)

  def __eq__(self, other):
    return self._transpose == other._transpose and self._fusedIndices == other._fusedIndices
  
  def __add__(self, other):
    return Cost(self._transpose + self._transpose, self._fusedIndices + self._fusedIndices)

def allSubstrings(s):
  L = len(s)
  return [s[i:j+1] for i in range(L) for j in range(i,L)]

def splitByDistance(p):
  L = len(p)
  splits = [i+1 for x,y,i in zip(p[:-1], p[1:], range(L)) if y-x != 1]
  return [p[i:j] for i,j in zip([0] + splits, splits + [L])]

def fusedVariants(I, P, M, prune = False):
  D = list()
  indices = sorted([P[p] for p in I])
  groups = splitByDistance(indices)
  groupStrings = [''.join([M[p] for p in sorted(g)]) for g in groups]
  D = set([s for g in groupStrings for s in allSubstrings(g)])
  if prune:
    D = set([d for d in D if d[0] == M[0]])
  return D

def indexString(s, M):
  return ''.join([m if m in s else ':' for m in M])

def LoG(A, B, C, ATFree = False, BTFree = False):
  candidates = list()
  if set(C) != (set(A) | set(B)) - (set(A) & set(B)):
    return sys.maxsize
  requiredIndices = set([A[0], B[0], C[0]])
  if C[0] in set(B):
    B, A = A, B
    BTFree, ATFree = ATFree, BTFree
  Im = set(A) & set(C)
  In = set(B) & set(C)
  Ik = set(A) & set(B)
  
  PA = {idx: pos for pos, idx in enumerate(A)}
  PB = {idx: pos for pos, idx in enumerate(B)}
  PC = {idx: pos for pos, idx in enumerate(C)}

  CM = fusedVariants(Im, PC, C, True)
  CN = fusedVariants(In, PC, C)
  AM = fusedVariants(Im, PA, A)
  AK = fusedVariants(Ik, PA, A)
  BK = fusedVariants(Ik, PB, B)
  BN = fusedVariants(In, PB, B)
  
  MC = CM & AM
  NC = CN & BN
  KC = AK & BK
  
  minCost = Cost()
  for m in MC:
    for n in NC:
      for k in KC:
        transpose = 0
        if A.find(m[0]) < A.find(k[0]):
          if A.find(m[0]) != 0:
            continue
        else:
          if not ATFree:
            transpose = transpose + 1
          if A.find(k[0]) != 0:
            continue
        if B.find(k[0]) < B.find(n[0]):
          if B.find(k[0]) != 0:
            continue
        else:
          if not BTFree:
            transpose = transpose + 1
          if B.find(n[0]) != 0:
            continue
        cost = Cost(transpose, len(m) + len(n) + len(k))
        minCost = min(minCost, cost)
  return minCost
