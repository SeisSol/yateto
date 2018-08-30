from ...ast.indices import Indices
from ..common import forLoops, reduceSpp, TensorDescription
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

  def generate(self, cpp):
    d = self._descr
    class LoGBody(object):
      def __call__(s):
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
        if hasLoops:
          self._pointer(cpp, Aname, d.leftTerm, d.loopIndices)
          self._pointer(cpp, Bname, d.rightTerm, d.loopIndices)
          self._pointer(cpp, Cname, d.result, d.loopIndices)
        
        AmemLayout = d.leftTerm.memoryLayout.slice(d.leftTerm.indices, Im, Ik)
        BmemLayout = d.rightTerm.memoryLayout.slice(d.rightTerm.indices, Ik, In)
        CmemLayout = d.result.memoryLayout.slice(d.result.indices, Im, In)

        Aeqspp = reduceSpp(d.leftTerm.eqspp, d.leftTerm.indices, A).reshape(AmemLayout.shape())
        Beqspp = reduceSpp(d.rightTerm.eqspp, d.rightTerm.indices, B).reshape(BmemLayout.shape())
        Ceqspp = reduceSpp(d.result.eqspp, d.result.indices, C).reshape(CmemLayout.shape())

        gemmDescr = gemm.Description(
          leftTerm = TensorDescription(Aname, AmemLayout, Aeqspp),
          rightTerm = TensorDescription(Bname, BmemLayout, Beqspp),
          result = TensorDescription(Cname, CmemLayout, Ceqspp),
          transA = d.transA,
          transB = d.transB,
          alpha = 1.0,
          beta = 1.0 if d.add else 0.0
        )
        generator = gemm.generator(self._arch, gemmDescr)
        generator.generate(cpp)
    forLoops(cpp, d.loopIndices, len(d.loopIndices)-1, LoGBody())
