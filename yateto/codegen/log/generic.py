import copy

from ... import ir
from ..common import *
from .. import gemm
from ...memory import DenseMemoryLayout


class Generic(object):
  """A contraction run as a loop over matrix products.

  The loops and the pointers are stated here; the product itself is a call,
  because which kernel takes it is the gemm configuration's decision and not
  this backend's.
  """

  def __init__(self, arch, descr, target, attrs=None):
    self._arch = arch
    self._descr = descr
    self._target = target
    # passed on to the gemm generator, which is where a call into an external
    # kernel is emitted and the batch flags have to be named or not
    self._attrs = attrs

  def _alignedStart(self, term, loopIndices, fixed):
    return term.memoryLayout.isAlignedAddressString(term.indices, term.indices & loopIndices, fixed)

  def _memLayout(self, term, I, J, fixed, isResult=False):
    if len(I) == 0 and len(J) == 0:
      return DenseMemoryLayout((1,1))
    elif len(I) == 0:
      ml = term.memoryLayout.vec(term.indices, J, fixed)
      # A degenerate m dimension makes this a 1 x N GEMM. For the operands that
      # is absorbed by transA/transB (see LoopOverGEMM in ast/node.py), so the
      # dummy goes last. The result has no transC, so it needs the dummy in
      # front -- otherwise the layout comes out as N x 1 and the bounding boxes
      # disagree with (m, n).
      return ml.withDummyDimension(front=isResult)
    elif len(J) == 0:
      ml = term.memoryLayout.vec(term.indices, I, fixed)
      return ml.withDummyDimension()
    elif len(term.indices) == 2:
      return term.memoryLayout
    return term.memoryLayout.unfold(term.indices, I, J, fixed)

  def _reduce(self, term, subset, memLayout, fixed):
    return reduceSpp(term.eqspp, term.indices, subset, fixed).reshape(memLayout.shape())

  def _defuse(self, fusedRange, term, I):
    if len(I) == 1:
      return  {next(iter(I)): fusedRange}
    return term.memoryLayout.defuse(fusedRange, term.indices, I)

  def _pointer(self, builder, name, buffer, term, loopIndices, indices, const=True):
    """A pointer that moves along `loopIndices` and stays put on the rest."""
    moving = term.indices & loopIndices
    axes = {position for position, index in enumerate(term.indices)
            if index in moving}
    # an axis the pointer stays put on contributes nothing, so what stands in
    # its coordinate is never read
    coords = [indices.get(index, 0) for index in term.indices]
    return builder.add(ir.Pointer(buffer, coords, axes, name=name, const=const))

  def _buffer(self, term, name=None):
    """The tensor's storage, or a pointer's name for it.

    A pointer names where a buffer starts; it is not storage of its own, so
    it is nobody's temporary however temporary what it points into is.
    """
    if name is None:
      return ir.Buffer(term.name, term.datatype, term.memoryLayout, term.eqspp,
                       term.is_temporary)
    return ir.Buffer(name, term.datatype, term.memoryLayout, term.eqspp)

  def _nest(self, builder, loopIndices, ranges, fixed, indices):
    """The nest over the indices that are not pinned, or None where it is empty.

    A pinned index that misses the range this nest runs over means the nest
    never runs, and nothing at all is stated for it.
    """
    names = [str(index) for index in loopIndices]
    for name in names:
      if name in fixed and not (ranges[name].start <= fixed[name] < ranges[name].stop):
        return None
    free = [indices[name] for name in names if name not in fixed]
    if names and not free:
      # every index of this nest is pinned, so there is no nest and nothing to
      # separate from what surrounds it
      return builder
    return ir.loopNest(builder, free, ranges, simd=False)

  def lower(self, gemm_cfg):
    d = self._descr

    unrollNeeded = set()
    for term in (d.leftTerm, d.rightTerm, d.result):
      if term.memoryLayout.isSparse():
        unrollNeeded |= set(term.indices)

    # NOTE: the unrolled indices are nested scopes in the emitted code, so their
    #       order is part of the output. Filtering the loop ranges keeps that
    #       order; intersecting a key view with a set hands back a set, which
    #       enumerates in an order PYTHONHASHSEED varies between runs.
    toBeUnrolled = [index for index in d.loopRanges if index in unrollNeeded]

    region = ir.Region()
    self._pin(ir.Builder(region), {}, toBeUnrolled, gemm_cfg)
    return region

  def _pin(self, builder, fixed, remaining, gemm_cfg):
    """One instantiation per value of the indices that have to be pinned.

    A sparse operand has an address for a known entry and none for an index, so
    the indices that reach it are numbers before anything is stated. Each
    instantiation gets a scope of its own, since each names its own pointers.
    """
    if not remaining:
      self._single(builder, fixed, gemm_cfg)
      return

    index, rest = remaining[0], remaining[1:]
    rng = self._descr.loopRanges[index]
    for value in range(rng.start, rng.stop):
      scope = builder.add(ir.Scope())
      self._pin(ir.Builder(scope.region), {**fixed, index: value}, rest, gemm_cfg)

  def _single(self, builder, fixed, gemm_cfg):
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
    hasInnerLoops = len(d.innerLoopIndices) > 0

    AmemLayout = self._memLayout(d.leftTerm, Im, Ik, fixed)
    BmemLayout = self._memLayout(d.rightTerm, Ik, In, fixed)
    CmemLayout = self._memLayout(d.result, Im, In, fixed, isResult=True)

    gemmDescr = gemm.Description(
      leftTerm = TensorDescription('_Ain' if hasInnerLoops else ('_A' if hasOuterLoops else d.leftTerm.name),
                                   AmemLayout, self._reduce(d.leftTerm, A, AmemLayout, fixed),
                                   d.leftTerm.is_compute_constant, d.leftTerm.is_temporary,
                                   datatype=d.leftTerm.datatype),
      rightTerm = TensorDescription('_Bin' if hasInnerLoops else ('_B' if hasOuterLoops else d.rightTerm.name),
                                    BmemLayout, self._reduce(d.rightTerm, B, BmemLayout, fixed),
                                    d.rightTerm.is_compute_constant, d.rightTerm.is_temporary,
                                    datatype=d.rightTerm.datatype),
      result = TensorDescription('_Cin' if hasInnerLoops else ('_C' if hasOuterLoops else d.result.name),
                                 CmemLayout, self._reduce(d.result, C, CmemLayout, fixed),
                                 d.result.is_compute_constant, d.result.is_temporary,
                                 datatype=d.result.datatype),
      transA = d.transA,
      transB = d.transB,
      alpha = d.alpha,
      beta = 1.0 if d.add else 0.0,
      arch = self._arch,
      alignedStartA = self._alignedStart(d.leftTerm, d.outerLoopIndices, fixed) and self._alignedStart(d.leftTerm, d.innerLoopIndices, fixed),
      alignedStartC = self._alignedStart(d.result, d.outerLoopIndices, fixed) and self._alignedStart(d.result, d.innerLoopIndices, fixed),
      prefetchName = None
    )

    if not d.add:
      lr = dict()
      m, n, k = gemmDescr.mnk()
      lr.update(d.loopRanges)
      lr.update(self._defuse(m, d.leftTerm, Im))
      lr.update(self._defuse(n, d.rightTerm, In))
      writeBB = boundingBoxFromLoopRanges(d.result.indices, lr)
      ir.zero(builder, self._buffer(d.result), writeBB)

    # a pinned index is a number wherever it is read, which is what turns a
    # sparse operand's address into one
    indices = {name: ir.Index(name) for name in d.loopRanges}
    indices.update(fixed)

    outer = self._nest(builder, d.outerLoopIndices, d.loopRanges, fixed, indices)
    if outer is None:
      return

    names = dict(A=d.leftTerm.name, B=d.rightTerm.name, C=d.result.name,
                 prefetch=d.prefetchName)
    outerPointers = []
    if hasOuterLoops:
      outerPointers = [
        self._pointer(outer, '_A', self._buffer(d.leftTerm), d.leftTerm,
                      d.outerLoopIndices, indices),
        self._pointer(outer, '_B', self._buffer(d.rightTerm), d.rightTerm,
                      d.outerLoopIndices, indices),
        self._pointer(outer, '_C', self._buffer(d.result), d.result,
                      d.outerLoopIndices, indices, const=False)]
      names.update(A='_A', B='_B', C='_C')
      if d.prefetchName is not None:
        outerPointers.append(
          self._pointer(outer, '_Cprefetch', self._buffer(d.result, d.prefetchName),
                        d.result, d.outerLoopIndices, indices))
        names['prefetch'] = '_Cprefetch'

    if d.assignLoopRanges is not None:
      self._boxes(outer, [d.assignLoopRanges], 0.0, gemmDescr, names, fixed,
                  indices, hasInnerLoops, gemm_cfg, outerPointers)
    if d.addLoopRanges is not None:
      self._boxes(outer, d.addLoopRanges, 1.0, gemmDescr, names, fixed,
                  indices, hasInnerLoops, gemm_cfg, outerPointers)

  def _boxes(self, builder, boxes, beta, gemmDescr, names, fixed, indices,
             hasInnerLoops, gemm_cfg, outerPointers):
    d = self._descr
    for ranges in boxes:
      inner = self._nest(builder, d.innerLoopIndices, ranges, fixed, indices)
      if inner is None:
        continue
      call = copy.copy(gemmDescr)
      call.setBeta(beta)
      handed = list(outerPointers)
      if hasInnerLoops:
        handed.append(self._pointer(inner, '_Ain', self._buffer(d.leftTerm, names['A']),
                                    d.leftTerm, d.innerLoopIndices, indices))
        handed.append(self._pointer(inner, '_Bin', self._buffer(d.rightTerm, names['B']),
                                    d.rightTerm, d.innerLoopIndices, indices))
        handed.append(self._pointer(inner, '_Cin', self._buffer(d.result, names['C']),
                                    d.result, d.innerLoopIndices, indices, const=False))
        if names['prefetch'] is not None:
          handed.append(self._pointer(inner, '_Cprefetchin',
                                      self._buffer(d.result, names['prefetch']), d.result,
                                      d.innerLoopIndices, indices))
          call.prefetchName = '_Cprefetchin'
      elif names['prefetch'] is not None:
        call.prefetchName = names['prefetch']
      generator = gemm.generator(self._arch, call, gemm_cfg, self._target,
                                 self._attrs)
      # the product reads the two operands and writes the result, whichever
      # kernel ends up performing it
      inner.add(ir.Call(generator.generate,
                        reads=[self._buffer(d.leftTerm), self._buffer(d.rightTerm),
                               self._buffer(d.result)],
                        writes=[self._buffer(d.result)],
                        operands=handed))
