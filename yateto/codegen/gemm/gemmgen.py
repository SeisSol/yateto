import hashlib
import subprocess
import numpy
import tempfile
from ..cache import RoutineGenerator

LIBXSMM_GENERATOR = 'libxsmm_gemm_generator'
SPARSEMMGEN_GENERATOR = 'sparsemmgen.py'

class GemmGen(object):
  def __init__(self, arch, descr, mode):
    self._arch = arch
    self._descr = descr
    self._mode = mode
  
  def generateRoutineName(self, gemm, spp):
    name = self._mode
    if spp is not None:
      sha = hashlib.md5()
      sha.update(str(spp).encode())
      name += 'sparse_' + sha.hexdigest()
    alpha = '1' if gemm['alpha'] == 1 else '_1'
    return '{name}_m{M}_n{N}_k{K}_ldA{LDA}_ldB{LDB}_ldC{LDC}_alpha{alphaSubs}_beta{beta}_alignedA{alignedA}_alignedC{alignedC}_{prefetch}'.format(name=name, alphaSubs=alpha, **gemm)
  
  def _pointer(self, term, offset2):
    o = term.memoryLayout.subtensorOffset(offset2)
    if o > 0:
      return '{} + {}'.format(term.name, o)
    return term.name
    
  def generate(self, cpp, routineCache):
    d = self._descr
    m, n, k = d.mnk()
    ldA = 0 if d.isACsc else d.leftTerm.memoryLayout.stridei(1)
    ldB = 0 if d.isBCsc else d.rightTerm.memoryLayout.stridei(1)
    ldC = d.result.memoryLayout.stridei(1)
    
    assert (m,k) in d.leftTerm.memoryLayout
    assert (k,n) in d.rightTerm.memoryLayout
    assert (m,n) in d.result.memoryLayout
    
    gemm = {
      'M':            m.size(),
      'N':            n.size(),
      'K':            k.size(),
      'LDA':          ldA,
      'LDB':          ldB,
      'LDC':          ldC,
      'alpha':        int(d.alpha),
      'beta':         int(d.beta),
      'alignedA':     int(d.alignedA),
      'alignedC':     int(d.alignedC),
      'prefetch':     'BL2viaC' if d.prefetchName is not None else 'pfsigonly'
    }

    spp = None
    sppRows = None
    flops = 0
    if d.isACsc:
      spp = d.leftTerm.memoryLayout.entries(m, k)
      sppRows = d.leftTerm.memoryLayout.shape()[0]
      flops = 2 * len(spp) * n.size()
    elif d.isBCsc:
      spp = d.rightTerm.memoryLayout.entries(k, n)
      sppRows = d.rightTerm.memoryLayout.shape()[0]
      flops = 2 * m.size() * len(spp)
    else:
      flops = 2 * m.size() * n.size() * k.size()
    
    routineName = self.generateRoutineName(gemm, spp)
    
    cpp( '{}({}, {}, {}, nullptr, {}, nullptr);'.format(
      routineName,
      self._pointer(d.leftTerm, (m.start, k.start)),
      self._pointer(d.rightTerm, (k.start, n.start)),
      self._pointer(d.result, (m.start, n.start)),
      d.prefetchName if d.prefetchName is not None else 'nullptr'
    ))
    
    routineCache.addRoutine(routineName, ExecuteGemmGen(self._arch, gemm, spp, sppRows, self._mode))
    
    return flops

class ExecuteGemmGen(RoutineGenerator):  
  def __init__(self, arch, gemmDescr, spp, sppRows, mode):
    self._arch = arch
    self._gemmDescr = gemmDescr
    self._spp = spp
    self._sppRows = sppRows
    self._mode = mode
  
  def __eq__(self, other):
    return self._arch == other._arch and self._gemmDescr == other._gemmDescr and numpy.array_equal(self._spp, other._spp) and self._mode == other._mode
  
  def header(self, cpp):
    with cpp.PPIfndef('NDEBUG'):
      cpp('extern long long libxsmm_num_total_flops;')
      cpp('extern long long sparsemmgen_num_total_flops;')
    with cpp.PPIf('defined( __SSE3__) || defined(__MIC__)'):
      cpp.includeSys('immintrin.h')

  def _callGenerator(self, argList):
    try:
      subprocess.call([str(arg) for arg in argList])
    except OSError:
      raise RuntimeError('{} executable "{}" not found. (Make sure to add the folder containing the executable to your PATH.)'.format(mode, LIBXSMM_GENERATOR if mode == 'libxsmm' else SPARSEMMGEN_GENERATOR))
  
  def __call__(self, routineName, fileName):
    if self._mode == 'libxsmm':
      argList = [
        LIBXSMM_GENERATOR,
        'dense',
        fileName,
        routineName,
        self._gemmDescr['M'],
        self._gemmDescr['N'],
        self._gemmDescr['K'],
        self._gemmDescr['LDA'],
        self._gemmDescr['LDB'],
        self._gemmDescr['LDC'],
        self._gemmDescr['alpha'],
        self._gemmDescr['beta'],
        self._gemmDescr['alignedA'],
        self._gemmDescr['alignedC'],
        self._arch.name,
        self._gemmDescr['prefetch'],
        self._arch.precision + 'P'
      ]
    else:
      argList = [
        SPARSEMMGEN_GENERATOR,
        self._gemmDescr['M'],
        self._gemmDescr['N'],
        self._gemmDescr['K'],
        self._gemmDescr['LDA'],
        self._gemmDescr['LDB'],
        self._gemmDescr['LDC'],
        self._gemmDescr['beta'],
        '--arch',
        self._arch.name,
        '--prefetching',
        self._gemmDescr['prefetch'],
        '--output_funcname',
        routineName,
        '--output_filename',
        fileName,
      ]
    if self._spp is not None:
      cols = self._gemmDescr['K'] if self._gemmDescr['LDA'] == 0 else self._gemmDescr['N']
      rows = self._gemmDescr['M'] if self._gemmDescr['LDA'] == 0 else self._gemmDescr['K']
      if self._mode == 'sparsemmgen':
        rows = self._sppRows
      shape = (rows, cols)      
      with tempfile.NamedTemporaryFile() as temp:
        temp.write('%%MatrixMarket matrix coordinate real general\n'.encode())
        temp.write('%\n'.encode())
        temp.write('{} {} {}\n'.format(shape[0], shape[1], len(self._spp)).encode())
        for r,c in self._spp:
          temp.write('{} {} 1.0\n'.format(r+1,c+1).encode())
        temp.flush()
        if self._mode == 'libxsmm':
          argList[1] = 'sparse'
        else:
          argList.append('--mtx_filename')
        argList.append(temp.name)
        self._callGenerator(argList)
    else:
      self._callGenerator(argList)

    return 'void {name}(const {type}* A, const {type}* B, {type}* C, const {type}* A_prefetch, const {type}* B_prefetch, const {type}* C_prefetch);'.format(name=routineName, type=self._arch.typename)
  
