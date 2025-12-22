from ...ast.indices import Indices
from ..common import *
from .. import gemm
from ...memory import DenseMemoryLayout

class Generic(object):
  def __init__(self, arch, descr, target):
    self._arch = arch
    self._descr = descr
    self._target = target
  
  def _pointer(self, cpp, targetName, baseName, term, loopIndices, fixed, const=True):
    indices = term.indices & loopIndices
    addressStr = term.memoryLayout.addressString(term.indices, indices, fixed) if len(indices) > 0 else ''
    if len(addressStr) > 0:
      addressStr = ' + ' + addressStr
    cpp('{} {}* {} = {}{};'.format(self._arch.typename, 'const' if const else '', targetName, baseName, addressStr))

  def _alignedStart(self, term, loopIndices, fixed):
    return term.memoryLayout.isAlignedAddressString(term.indices, term.indices & loopIndices, fixed)
    
  def _memLayout(self, term, I, J, fixed):
    if len(I) == 0 and len(J) == 0:
      return DenseMemoryLayout((1,1))
    elif len(I) == 0:
      ml = term.memoryLayout.vec(term.indices, J, fixed)
      return ml.withDummyDimension()
    elif len(J) == 0:
      ml = term.memoryLayout.vec(term.indices, I, fixed)
      return ml.withDummyDimension()
    elif len(term.indices) == 2:
      return term.memoryLayout
    return term.memoryLayout.unfold(term.indices, I, J, fixed)
  
  def _unroll(self, term, I):
    if not term.memoryLayout.isSparse():
      return set()

    return I

  def _reduce(self, term, subset, memLayout, fixed):
    return reduceSpp(term.eqspp, term.indices, subset, fixed).reshape(memLayout.shape())
  
  def _defuse(self, fusedRange, term, I):
    if len(I) == 1:
      return  {next(iter(I)): fusedRange}
    return term.memoryLayout.defuse(fusedRange, term.indices, I)

  def _generateSingle(self, cpp, routineCache, gemm_cfg, fixed = {}):
    d = self._descr
    
    A = d.leftTerm.indices - d.loopIndices
    B = d.rightTerm.indices - d.loopIndices
    C = d.result.indices - d.loopIndices
    Im = set(A) & set(C)
    In = set(B) & set(C)
    Ik = set(A) & set(B)
    
    hasOuterLoops = len(d.outerLoopIndices) > 0

    if hasOuterLoops and self._target == 'gpu':
      raise RuntimeError("Loop over GEMM with the outer loop hasn't been implemented yet "
                         "for the GPU-like architectures")

    outerAname = '_A' if hasOuterLoops else d.leftTerm.name
    outerBname = '_B' if hasOuterLoops else d.rightTerm.name
    outerCname = '_C' if hasOuterLoops else d.result.name
    outerPrefetchName = '_Cprefetch' if hasOuterLoops and d.prefetchName is not None else d.prefetchName
    
    hasInnerLoops = len(d.innerLoopIndices) > 0
    innerAname = '_Ain' if hasInnerLoops else outerAname
    innerBname = '_Bin' if hasInnerLoops else outerBname
    innerCname = '_Cin' if hasInnerLoops else outerCname
    innerPrefetchName = '_Cprefetchin' if hasInnerLoops and outerPrefetchName is not None else outerPrefetchName
    
    AmemLayout = self._memLayout(d.leftTerm, Im, Ik, fixed)
    BmemLayout = self._memLayout(d.rightTerm, Ik, In, fixed)
    CmemLayout = self._memLayout(d.result, Im, In, fixed)

    Aeqspp = self._reduce(d.leftTerm, A, AmemLayout, fixed)
    Beqspp = self._reduce(d.rightTerm, B, BmemLayout, fixed)
    Ceqspp = self._reduce(d.result, C, CmemLayout, fixed)

    gemmDescr = gemm.Description(
      leftTerm = TensorDescription(innerAname, AmemLayout, Aeqspp, d.leftTerm.is_compute_constant, d.leftTerm.is_temporary),
      rightTerm = TensorDescription(innerBname, BmemLayout, Beqspp, d.rightTerm.is_compute_constant, d.rightTerm.is_temporary),
      result = TensorDescription(innerCname, CmemLayout, Ceqspp, d.result.is_compute_constant, d.result.is_temporary),
      transA = d.transA,
      transB = d.transB,
      alpha = d.alpha,
      beta = 1.0 if d.add else 0.0,
      arch = self._arch,
      alignedStartA = self._alignedStart(d.leftTerm, d.outerLoopIndices, fixed) and self._alignedStart(d.leftTerm, d.innerLoopIndices, fixed),
      alignedStartC = self._alignedStart(d.result, d.outerLoopIndices, fixed) and self._alignedStart(d.result, d.innerLoopIndices, fixed),
      prefetchName = innerPrefetchName
    )
    
    if not d.add:
      lr = dict()
      m, n, k = gemmDescr.mnk()
      lr.update(d.loopRanges)
      lr.update( self._defuse(m, d.leftTerm, Im) )
      lr.update( self._defuse(n, d.rightTerm, In) )
      writeBB = boundingBoxFromLoopRanges(d.result.indices, lr)
      initializeWithZero(cpp, self._arch, d.result, writeBB)
    
    class LoGBody(object):
      def __call__(s):
        if hasInnerLoops:
          self._pointer(cpp, innerAname, outerAname, d.leftTerm, d.innerLoopIndices, fixed)
          self._pointer(cpp, innerBname, outerBname, d.rightTerm, d.innerLoopIndices, fixed)
          self._pointer(cpp, innerCname, outerCname, d.result, d.innerLoopIndices, fixed, const=False)
          if outerPrefetchName is not None:
            self._pointer(cpp, innerPrefetchName, outerPrefetchName, d.result, d.innerLoopIndices, fixed)
        generator = gemm.generator(self._arch, gemmDescr, gemm_cfg, self._target)
        return generator.generate(cpp, routineCache)

    class InnerLoopBody(object):
      def __call__(s):
        flops = 0
        if hasOuterLoops:
          self._pointer(cpp, outerAname, d.leftTerm.name, d.leftTerm, d.outerLoopIndices, fixed)
          self._pointer(cpp, outerBname, d.rightTerm.name, d.rightTerm, d.outerLoopIndices, fixed)
          self._pointer(cpp, outerCname, d.result.name, d.result, d.outerLoopIndices, fixed, const=False)
          if d.prefetchName is not None:
            self._pointer(cpp, outerPrefetchName, d.prefetchName, d.result, d.outerLoopIndices, fixed)

        if d.assignLoopRanges is not None:
          gemmDescr.setBeta(0.0)
          flops += forLoops(cpp, d.innerLoopIndices, d.assignLoopRanges, LoGBody(), pragmaSimd=False, fixed=fixed)
        if d.addLoopRanges is not None:
          gemmDescr.setBeta(1.0)
          flops += forLoops(cpp, d.innerLoopIndices, d.addLoopRanges, LoGBody(), pragmaSimd=False, fixed=fixed)
        return flops

    return forLoops(cpp, d.outerLoopIndices, d.loopRanges, InnerLoopBody(), pragmaSimd=False, fixed=fixed)

  def _generateUnroll(self, cpp, routineCache, gemm_cfg, fixed, unroll):
    d = self._descr

    if len(unroll) == 0:
      return self._generateSingle(cpp, routineCache, gemm_cfg, fixed)
    
    unrollNow = unroll[0]

    rngNow = d.loopRanges[unrollNow]

    result = 0
    for i in range(rngNow.start, rngNow.stop):
      fixedNow = dict(fixed)
      fixedNow[unrollNow] = i
      result += self._generateUnroll(cpp, routineCache, gemm_cfg, fixedNow, unroll[1:])
    
    return result

  def generate(self, cpp, routineCache, gemm_cfg):
    d = self._descr
    
    A = d.leftTerm.indices - d.loopIndices
    B = d.rightTerm.indices - d.loopIndices
    C = d.result.indices - d.loopIndices
    Im = set(A) & set(C)
    In = set(B) & set(C)
    Ik = set(A) & set(B)

    toBeUnrolled = d.loopRanges.keys()

    unrollNeeded = set()
    if d.leftTerm.memoryLayout.isSparse():
      unrollNeeded |= set(d.leftTerm.indices)
    if d.rightTerm.memoryLayout.isSparse():
      unrollNeeded |= set(d.rightTerm.indices)
    if d.result.memoryLayout.isSparse():
      unrollNeeded |= set(d.result.indices)
    
    toBeUnrolled &= unrollNeeded

    return self._generateUnroll(cpp, routineCache, gemm_cfg, {}, list(toBeUnrolled))
