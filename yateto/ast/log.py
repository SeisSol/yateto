import sys
from .node import LoopOverGEMM
from .indices import LoGCost

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
  
  minCost = LoGCost()
  for m in MC:
    for n in NC:
      for k in KC:
        Cstr = 'C_{}'.format(indexString(m+n, C))
        transpose = 0
        if A.find(m[0]) < A.find(k[0]):
          Astr = 'A_{}'.format(indexString(m+k, A))
          if A.find(m[0]) != 0:
            continue
        else:
          Astr = '(A_{})\''.format(indexString(m+k, A))
          if not ATFree:
            transpose = transpose + 1
          if A.find(k[0]) != 0:
            continue
        if B.find(k[0]) < B.find(n[0]):
          Bstr = 'B_{}'.format(indexString(k+n, B))
          if B.find(k[0]) != 0:
            continue
        else:
          Bstr = '(B_{})\''.format(indexString(k+n, B))
          if not BTFree:
            transpose = transpose + 1
          if B.find(n[0]) != 0:
            continue
        cost = LoGCost(transpose, len(m) + len(n) + len(k))
        #~ print(m, n, k, len(m) + len(n) + len(k), '{} = {} {}'.format(Cstr, Astr, Bstr))
        minCost = min(minCost, cost)
  return minCost

def LoGbla(contraction):
  L = contraction.leftTerm()
  R = contraction.rightTerm()

  A = L.indices.tostring()
  B = R.indices.tostring()
  C = contraction.indices.tostring()

  candidates = list()
  if set(C) != (set(A) | set(B)) - (set(A) & set(B)):
    return sys.maxsize
  requiredIndices = set([A[0], B[0], C[0]])
  if C[0] in set(B):
    B, A = A, B
    R, L = L, R
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
  
  minCost = LoGCost()
  minLog = None
  for m in MC:
    for n in NC:
      for k in KC:
        log = LoopOverGEMM(contraction.indices, L, R, m, n, k)
        cost = log.cost()
        if cost < minCost:
          minCost = cost
          minLog = log
  return minLog
