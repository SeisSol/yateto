from ...ast.indices import Indices
from ..common import *
from .. import gemm

class Generic(object):
  def __init__(self, arch, descr):
    self._arch = arch
    self._descr = descr
  
  def _pointer(self, cpp, name, term, loopIndices):
    addressStr = term.memoryLayout.addressString(term.indices, term.indices & loopIndices)
    if len(addressStr) > 0:
      addressStr = ' + ' + addressStr
    cpp('{}* {} = {}{};'.format(self._arch.typename, name, term.name, addressStr))
    
  def _memLayout(self, term, I, J):
    if len(term.indices) > 2:
      return term.memoryLayout.fusedSlice(term.indices, I, J)
    return term.memoryLayout

  def _reduce(self, term, subset, memLayout):
    if len(term.indices) > 2:
      return reduceSpp(term.eqspp, term.indices, subset).reshape(memLayout.shape(), order='F')
    return term.eqspp
  
  def _defuse(self, fusedRange, term, I):
    if len(I) == 1:
      return  {next(iter(I)): fusedRange}
    return term.memoryLayout.defuse(fusedRange, term.indices, I)

  def generate(self, cpp):
    d = self._descr
    
    A = d.leftTerm.indices - d.loopIndices
    B = d.rightTerm.indices - d.loopIndices
    C = d.result.indices - d.loopIndices
    Im = set(A) & set(C)
    In = set(B) & set(C)
    Ik = set(A) & set(B)
    
    hasLoops = len(d.loopIndices) > 0
    Aname = 'A' if hasLoops else d.leftTerm.name
    Bname = 'B' if hasLoops else d.rightTerm.name
    Cname = 'C' if hasLoops else d.result.name
    
    AmemLayout = self._memLayout(d.leftTerm, Im, Ik)
    BmemLayout = self._memLayout(d.rightTerm, Ik, In)
    CmemLayout = self._memLayout(d.result, Im, In)

    Aeqspp = self._reduce(d.leftTerm, A, AmemLayout)
    Beqspp = self._reduce(d.rightTerm, B, BmemLayout)
    Ceqspp = self._reduce(d.result, C, CmemLayout)

    gemmDescr = gemm.Description(
      leftTerm = TensorDescription(Aname, AmemLayout, Aeqspp),
      rightTerm = TensorDescription(Bname, BmemLayout, Beqspp),
      result = TensorDescription(Cname, CmemLayout, Ceqspp),
      transA = d.transA,
      transB = d.transB,
      alpha = 1.0,
      beta = 1.0 if d.add else 0.0,
      arch = self._arch
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
        if hasLoops:
          self._pointer(cpp, Aname, d.leftTerm, d.loopIndices)
          self._pointer(cpp, Bname, d.rightTerm, d.loopIndices)
          self._pointer(cpp, Cname, d.result, d.loopIndices)
        generator = gemm.generator(self._arch, gemmDescr)
        generator.generate(cpp)
    
    forLoops(cpp, d.loopIndices, d.loopRanges, LoGBody())
