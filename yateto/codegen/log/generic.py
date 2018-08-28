from ...ast.indices import Indices
from ..common import forLoops, TensorDescription
from .. import gemm

class Generic(object):
  def __init__(self, arch, descr):
    self._arch = arch
    self._descr = descr
  
  def _pointer(self, cpp, name, term, loopIndices):
    cpp('{}* {} = {} + {};'.format(self._arch.typename, name, term.name, term.memoryLayout.addressString(term.indices, term.indices & loopIndices)))

  def generate(self, cpp):
    d = self._descr
    class LoGBody(object):
      def __call__(s):
        A = set(d.leftTerm.indices - d.loopIndices)
        B = set(d.rightTerm.indices - d.loopIndices)
        C = set(d.result.indices - d.loopIndices)
        Im = A & C
        In = B & C
        Ik = A & B
        
        hasLoops = len(d.loopIndices) > 0
        Aname = 'A' if hasLoops else d.leftTerm.name
        Bname = 'B' if hasLoops else d.rightTerm.name
        Cname = 'C' if hasLoops else d.result.name
        if hasLoops:
          self._pointer(cpp, Aname, d.leftTerm, d.loopIndices)
          self._pointer(cpp, Bname, d.rightTerm, d.loopIndices)
          self._pointer(cpp, Cname, d.result, d.loopIndices)
        gemmDescr = gemm.Description(
          result = TensorDescription(Cname, d.result.memoryLayout.slice(d.result.indices, Im, In)),
          leftTerm = TensorDescription(Aname, d.leftTerm.memoryLayout.slice(d.leftTerm.indices, Im, Ik)),
          rightTerm = TensorDescription(Bname, d.rightTerm.memoryLayout.slice(d.rightTerm.indices, Ik, In)),
          transA = d.transA,
          transB = d.transB,
          alpha = 1.0,
          beta = 1.0 if d.add else 0.0
        )
        generator = gemm.generator(self._arch, gemmDescr)
        generator.generate(cpp)
    forLoops(cpp, d.loopIndices, len(d.loopIndices)-1, LoGBody())
