import hashlib
import subprocess
import tempfile
from abc import ABC
import numpy as np

from ..cache import RoutineGenerator, GpuRoutineGenerator
from ...gemm_configuration import BLASlike, CodeGenerator, GemmForge
from ..common import BatchedOperationsAux
import importlib.util


# Optional modules
gf_spec = importlib.util.find_spec('gemmforge')
try:
  if gf_spec:
    gf = gf_spec.loader.load_module()
except:
  raise ('Cannot load gemmforge.')


class GemmGen(object):
  def __init__(self, arch, descr, gemm_cfg):
    self._arch = arch
    self._descr = descr
    self._gemm_cfg = gemm_cfg
    self._mode = gemm_cfg.operation_name

  def _is_special(self, value, specials):
    result = 'generic'
    try:
      candidate = float(value)
      for special in specials:
        if abs(candidate - special) < 1e-10:
          result = special
          break
    except:
      pass
    result = str(result).replace('-', 'm').replace('.', 'd')
    return result

  def _alpha(self, alpha):
    specials = {-1,1} if self._mode == 'pspamm' else {1}
    return self._is_special(alpha, specials)

  def _beta(self, beta):
    return self._is_special(beta, {0,1})

  def generateRoutineName(self, gemm, sppA, sppB):
    name = self._gemm_cfg.operation_name
    name += '_' + {
      (True, True): 'dense',
      (True, False): 'bsparse',
      (False, True): 'asparse',
      (False, False): 'absparse'
    }[(sppA is None, sppB is None)]
    if sppA is not None:
      sha = hashlib.md5()
      sha.update(str(sppA).encode())
      name += '_' + sha.hexdigest()
    if sppB is not None:
      sha = hashlib.md5()
      sha.update(str(sppB).encode())
      name += '_' + sha.hexdigest()
    return '{name}_m{M}_n{N}_k{K}_ldA{LDA}_ldB{LDB}_ldC{LDC}_alpha{alphaSubs}_beta{betaSubs}_alignedA{alignedA}_alignedC{alignedC}_transA{transA}_transB{transB}_{prefetch}'.format(
      name=name,
      alphaSubs=self._alpha(gemm['alpha']),
      betaSubs=self._beta(gemm['beta']),
      **gemm
    )
  
  def _pointer(self, term, offset2, transpose):
    if transpose:
      # swaps elements of tuple if transpose
      offset2 = offset2[::-1]
    o = term.memoryLayout.subtensorOffset(topLeftEntry=offset2)
    if o > 0:
      return '{} + {}'.format(term.name, o)
    return term.name
    
  def generate(self, cpp, routineCache):
    d = self._descr
    m, n, k = d.mnk()
    ldA = 0 if d.isACsc else d.leftTerm.memoryLayout.stridei(1)
    ldB = 0 if d.isBCsc else d.rightTerm.memoryLayout.stridei(1)
    ldC = d.result.memoryLayout.stridei(1)
    
    assert (d.transA and (k,m) in d.leftTerm.memoryLayout) or (not d.transA and (m,k) in d.leftTerm.memoryLayout)
    assert (d.transB and (n,k) in d.rightTerm.memoryLayout) or (not d.transB and (k,n) in d.rightTerm.memoryLayout)
    assert (m,n) in d.result.memoryLayout

    sppA = None
    sppARows = None
    sppB = None
    sppBRows = None
    flops = 0
    if d.isACsc:
      sppA = d.leftTerm.memoryLayout.entries(m, k)
      sppARows = d.leftTerm.memoryLayout.shape()[0]
    if d.isBCsc:
      sppB = d.rightTerm.memoryLayout.entries(k, n)
      sppBRows = d.rightTerm.memoryLayout.shape()[0]
    
    if d.isACsc and d.isBCsc:
      # count the flops by splitting into outer products (i.e. partition by k)
      # for each outer product, we need to compute all-by-all nonzero entries for m and n
      # in essence: flops = 2 * sum(1 for ae in sppA for be in sppB if ae[1] == be[0]); then simplify this calculation

      mcount = np.bincount([kk for _,kk in sppA], minlength=k.size())
      ncount = np.bincount([kk for kk,_ in sppB], minlength=k.size())
      flops = 2 * int(np.dot(ncount, mcount))
    elif d.isACsc:
      flops = 2 * n.size() * len(sppA)
    elif d.isBCsc:
      flops = 2 * m.size() * len(sppB)
    else:
      flops = 2 * m.size() * n.size() * k.size()
    
    if isinstance(self._gemm_cfg, BLASlike):
      ptr_a = self._pointer(term=d.leftTerm, offset2=(m.start, k.start), transpose=d.transA)
      ptr_b = self._pointer(term=d.rightTerm, offset2=(k.start, n.start), transpose=d.transB)
      ptr_c = self._pointer(term=d.result, offset2=(m.start, n.start), transpose=False)

      cpp(  self._gemm_cfg.call(d.transA,
                                d.transB,
                                m.size(), n.size(), k.size(),
                                d.alpha,
                                ptr_a, ldA,
                                ptr_b, ldB,
                                d.beta, ptr_c, ldC,
                                alignedA=d.alignedA,
                                alignedC=d.alignedC,
                                prefetchName=d.prefetchName))
    elif isinstance(self._gemm_cfg, GemmForge):

      if gf_spec:
        aux = BatchedOperationsAux(self._arch.typename)

        matrix_a = gf.YatetoInterface.produce_dense_matrix((m, k),
                                                           d.leftTerm.memoryLayout.bbox(),
                                                           addressing=aux.deduce_addresing(d.leftTerm),
                                                           transpose=d.transA)

        matrix_b = gf.YatetoInterface.produce_dense_matrix((k, n),
                                                           d.rightTerm.memoryLayout.bbox(),
                                                           addressing=aux.deduce_addresing(d.rightTerm),
                                                           transpose=d.transB)

        matrix_c = gf.YatetoInterface.produce_dense_matrix((m, n),
                                                           d.result.memoryLayout.bbox(),
                                                           addressing=aux.deduce_addresing(d.result),
                                                           transpose=False)

        try:
          vm = gf.vm_factory(self._arch.name, self._arch.backend, fp_type=self._arch.typename)
          forge_generator = gf.GemmGenerator(vm)
          forge_generator.set(d.transA, d.transB, matrix_a, matrix_b, matrix_c, d.alpha, d.beta)
          routine_name = forge_generator.get_base_name()

          args = [aux.deduce_arg(d.leftTerm, as_const=True),
                  aux.deduce_arg(d.rightTerm, as_const=True),
                  aux.deduce_arg(d.result, as_const=False),
                  BatchedOperationsAux.NUM_ELEMENTS_NAME,
                  BatchedOperationsAux.FLAGS_NAME,
                  BatchedOperationsAux.STREAM_PTR_NAME]
          args_str = ', '.join(args)

          if not isinstance(d.alpha, float):
            args_str = f'{d.alpha}, {args_str}'

          cpp(f'{routine_name}({args_str});')

          routineCache.addRoutine(routine_name, GemmForgeWriter(forge_generator, vm.get_headers()))

        except gf.GenerationError as err:
          print(f'ERROR from GemmForge: {err}')
          raise err
      else:
        raise RuntimeError('gemmforge module is not found. You can install it with pip3. '
                           'e.g., pip3 install gemmforge')
    else:
      gemm = {
        'M':            m.size(),
        'N':            n.size(),
        'K':            k.size(),
        'LDA':          ldA,
        'LDB':          ldB,
        'LDC':          ldC,
        'alpha':        self._alpha(d.alpha),
        'beta':         self._beta(d.beta),
        'alignedA':     int(d.alignedA),
        'alignedC':     int(d.alignedC),
        'prefetch':     'BL2viaC' if self._arch.enablePrefetch and d.prefetchName is not None else None,
        'transA': d.transA,
        'transB': d.transB,

      }

      routineName = self.generateRoutineName(gemm, sppA, sppB)

      if self._mode == 'pspamm':
        cpp( '{}({}, {}, {}, {}, {}, {});'.format(
          routineName,
          self._pointer(d.leftTerm, (m.start, k.start), d.transA),
          self._pointer(d.rightTerm, (k.start, n.start), d.transB),
          self._pointer(d.result, (m.start, n.start), False),
          str(d.alpha),
          str(d.beta),
          d.prefetchName if d.prefetchName is not None else 'nullptr'
        ))
      else:
        cpp( '{}({}, {}, {}, nullptr, {}, nullptr);'.format(
          routineName,
          self._pointer(d.leftTerm, (m.start, k.start), d.transA),
          self._pointer(d.rightTerm, (k.start, n.start), d.transB),
          self._pointer(d.result, (m.start, n.start), False),
          d.prefetchName if d.prefetchName is not None else 'nullptr'
        ))

      if self._gemm_cfg.is_internal():
        routineCache.addRoutine(routineName, LibxsmmGemmGen(
          self._arch, gemm, spp, sppRows, self._gemm_cfg))
      else:
        routineCache.addRoutine(routineName, ExecuteGemmGen(self._arch, gemm, sppA, sppARows, sppB, sppBRows, self._gemm_cfg))

    return flops

class ExecuteGemmGen(RoutineGenerator):
  def __init__(self, arch, gemmDescr, sppA, sppARows, sppB, sppBRows, gemm_cfg):
    self._arch = arch
    self._gemmDescr = gemmDescr
    self._sppA = sppA
    self._sppARows = sppARows
    self._sppB = sppB
    self._sppBRows = sppBRows
    self._mode = gemm_cfg.operation_name
    self._cmd = gemm_cfg.cmd
    self._blockSize = gemm_cfg.blockSize(gemmDescr['M'], gemmDescr['N'], gemmDescr['K']) if hasattr(gemm_cfg, 'blockSize') else dict()
  
  def __eq__(self, other):
    return self._arch == other._arch and \
           self._gemmDescr == other._gemmDescr and \
           self._sppA == other._sppA and self._sppB == other._sppB
  
  def header(self, cpp):
    with cpp.PPIfndef('NDEBUG'):
      cpp('extern long long libxsmm_num_total_flops;')
      cpp('extern long long pspamm_num_total_flops;')
    with cpp.PPIf('defined( __SSE3__) || defined(__MIC__)'):
      cpp.includeSys('immintrin.h')

  def _callGenerator(self, argList):
    resultCode = 1
    try:
      strcmd = [str(arg) for arg in argList]
      result = subprocess.run(strcmd, capture_output=True, text=True)
    except OSError:
      raise RuntimeError(f'GEMM code generator executable "{self._cmd}" not found. (Make sure to add the folder containing the executable to your PATH environment variable.)')
    if result.returncode != 0:
      raise RuntimeError(f"""GEMM code generator executable "{self._cmd}" failed. Thus, the kernel generation may be incomplete.
Given command: {' '.join(strcmd)}
Stdout: {result.stdout}
Stderr: {result.stderr}""")
  
  def __call__(self, routineName, fileName):
    cpu_arch = self._arch.host_name if self._arch.host_name else self._arch.name

    if self._mode == 'pspamm':
      pspamm_arch = cpu_arch
      if cpu_arch == 'a64fx':
        pspamm_arch = 'arm_sve512'
      elif cpu_arch in ['thunderx2t99', 'neon'] or cpu_arch.startswith('apple-m'):
        pspamm_arch = 'arm'
      elif cpu_arch.startswith('sve'):
        pspamm_arch = f'arm_{cpu_arch}' # TODO(David): rename to sveLEN only
      elif cpu_arch in ['naples', 'rome', 'milan']:
        # names are Zen1, Zen2, Zen3, respectively
        # no explicit support for these archs yet, but they have the same instruction sets (AVX2+FMA3) that HSW also needs
        pspamm_arch = 'hsw'
      elif cpu_arch in ['bergamo', 'turin']:
        pspamm_arch = 'skx'
      argList = [
        self._cmd,
        self._gemmDescr['M'],
        self._gemmDescr['N'],
        self._gemmDescr['K'],
        self._gemmDescr['LDA'],
        self._gemmDescr['LDB'],
        self._gemmDescr['LDC'],
        self._gemmDescr['alpha'],
        self._gemmDescr['beta'],
        '--arch',
        pspamm_arch,
        '--output_funcname',
        routineName,
        '--output_filename',
        fileName,
        '--precision',
        self._arch.precision
      ]
      if self._gemmDescr['prefetch']:
        argList.extend(['--prefetching', self._gemmDescr['prefetch']])
      if self._gemmDescr['transA']:
        argList.extend(['--atranspose', 'true'])
      if self._gemmDescr['transB']:
        argList.extend(['--btranspose', 'true'])
      for key, val in self._blockSize.items():
        argList.extend(['--' + key, val])
    else:
      libxsmm_arch = cpu_arch
      if cpu_arch in ['naples', 'rome', 'milan']:
        # names are Zen1, Zen2, Zen3, respectively
        # no explicit support for these archs yet, but they have the same instruction sets (AVX2+FMA3) that HSW also needs
        libxsmm_arch = 'hsw'
      elif cpu_arch in ['bergamo', 'turin']:
        libxsmm_arch = 'skx'
      argList = [
        self._cmd,
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
        libxsmm_arch, # libxsmm has no support for rome, hsw works well in practice
        self._gemmDescr['prefetch'] if self._gemmDescr['prefetch'] else 'nopf',
        self._arch.precision + 'P'
      ]
    class SparsityWrapper:
      def __init__(self, shape, spp):
        self._shape = shape
        self._spp = spp
        self._temp = None
      
      def __enter__(self):
        if self._spp is not None:
          self._temp = tempfile.NamedTemporaryFile()
          self._temp.__enter__()
          self._temp.write('%%MatrixMarket matrix coordinate real general\n'.encode())
          self._temp.write('%\n'.encode())
          self._temp.write('{} {} {}\n'.format(self._shape[0], self._shape[1], len(self._spp)).encode())
          for r,c in self._spp:
            self._temp.write('{} {} 1.0\n'.format(r+1,c+1).encode())
          self._temp.flush()
          return self._temp.name
        return None
      
      def __exit__(self, exc_type, exc_val, exc_tb):
        if self._spp is not None:
          self._temp.__exit__(exc_type, exc_val, exc_tb)
    
    with SparsityWrapper((self._gemmDescr['M'], self._gemmDescr['K']), self._sppA) as afile:
      with SparsityWrapper((self._sppBRows if self._mode=='pspamm' else self._gemmDescr['K'], self._gemmDescr['N']), self._sppB) as bfile:
        if self._mode == 'libxsmm':
          assert afile is None or bfile is None
          if afile is not None:
            argList[1] = 'sparse'
            argList.append(afile)
          if bfile is not None:
            argList[1] = 'sparse'
            argList.append(bfile)
        if self._mode == 'pspamm':
          if afile is not None:
            argList.extend(['--amtx_filename', afile])
          if bfile is not None:
            # actually bmtx_filename
            # take mtx_filename (alias) instead for backwards compatibility
            argList.extend(['--mtx_filename', bfile])
        self._callGenerator(argList)

    if self._mode == 'pspamm':
      return 'void {name}(const {type}* A, const {type}* B, {type}* C, {type} alpha, {type} beta, const {type}* prefetch);'.format(name=routineName, type=self._arch.typename)
    return 'void {name}(const {type}* A, const {type}* B, {type}* C, const {type}* A_prefetch, const {type}* B_prefetch, const {type}* C_prefetch);'.format(name=routineName, type=self._arch.typename)
  

class GemmForgeWriter(GpuRoutineGenerator):
  def __init__(self, forge_generator, headers):
    self._generator = forge_generator
    self._basename = forge_generator.get_base_name()
    self._headers = headers

  def __eq__(self, other):
    if isinstance(other, GemmForgeWriter):
      return self._basename == other._basename
    else:
      return False

  def header(self, cpp):
    cpp.includes(self._headers)

  def __call__(self, routineName, fileName):
    self._generator.generate()
    declaration = self._generator.get_launcher_header()
    launcher = self._generator.get_launcher()
    kernel = self._generator.get_kernel()

    with open(fileName, "a") as file:
      file.write(kernel)
      file.write(launcher)

    return declaration


class LibxsmmGemmGen(ExecuteGemmGen):
  def __init__(self,
               arch,
               gemm_descr,
               sppA, sppARows, sppB, sppBRows,
               gemm_cfg):
    super().__init__(arch, gemm_descr, sppA, sppARows, sppB, sppBRows, gemm_cfg)

  def header(self, cpp):
    super().header(cpp)
    cpp.include('libxsmm.h')

  def _kernel(self, routine_name):
    M = self._gemmDescr['M']
    N = self._gemmDescr['N']
    K = self._gemmDescr['K']
    ldA = self._gemmDescr['LDA']
    ldB = self._gemmDescr['LDB']
    ldC = self._gemmDescr['LDC']
    alpha = self._gemmDescr['alpha']
    beta = self._gemmDescr['beta']
    alignedA = self._gemmDescr['alignedA']
    alignedC = self._gemmDescr['alignedC']
    prefetch = self._gemmDescr['prefetch']
    transA = self._gemmDescr['transA']
    transB = self._gemmDescr['transB']

    flags = ["LIBXSMM_GEMM_FLAG_NONE"]
    if transA:
      flags += ['LIBXSMM_GEMM_FLAG_TRANS_A']
    if transB:
      flags += ['LIBXSMM_GEMM_FLAG_TRANS_B']

    # Note: Alignment is currently a bit buggy.
    # Enabling alignedC leads to wrong results currenty.
    # See:
    # https://github.com/SeisSol/yateto/issues/15
    #if alignedA:
    #flags += ["LIBXSMM_GEMM_FLAG_ALIGN_A"]
    #if alignedC:
    #flags += ["LIBXSMM_GEMM_FLAG_ALIGN_C"]
    libxsmm_flag_str = " | ".join(flags)

    prefetch_flag =  "LIBXSMM_GEMM_PREFETCH_NONE" if not self._arch.enablePrefetch else "LIBXSMM_GEMM_PREFETCH_BL2_VIA_C"

    kernel_var_name = f'{routine_name}_var'
    return """
static auto {kernel_var_name} = libxsmm_mmfunction<{prec}>(
  {flag}, // flag
  {M}, // M
  {N}, // N
  {K}, // K
  {ldA}, // lda
  {ldB}, // ldb
  {ldC}, // ldc
  {alpha}, // alpha
  {beta}, // beta
  {prefetch_flag} // prefetch
); 
""".format(kernel_var_name=kernel_var_name,
           prec=self._arch.typename, M=M, N=N, K=K,
           ldA=ldA, ldB=ldB, ldC=ldC,
           alpha=alpha, beta=beta,
           flag=libxsmm_flag_str,
           prefetch_flag=prefetch_flag,
           prefetch=prefetch,
 )

  def _call(self, routineName):
    # See: https://github.com/libxsmm/libxsmm/blob/180aae45a74731c3f8fb697b62dcb6c336c218ac/src/generator_gemm_common.c#L1971
    #flops=2*M*N*K
    M = self._gemmDescr['M']
    N = self._gemmDescr['N']
    K = self._gemmDescr['K']
    flops = 2 * M * N * K;
    return f"""
{{
    {routineName}_var(A, B, C, A_prefetch, B_prefetch, C_prefetch);
#ifndef NDEBUG
#ifdef _OPENMP
#pragma omp atomic
#endif
    libxsmm_num_total_flops += {flops}; // 2 * {M} * {N} * {K}
#endif
}}
"""

  def _functionSignature(self, routineName):
    return 'void {routineName}(const {type}* A, const {type}* B, {type}* C, const {type}* A_prefetch, const {type}* B_prefetch, const {type}* C_prefetch)'.format(routineName=routineName, type=self._arch.typename)

  def __call__(self, routineName, fileName):
    func_signature = self._functionSignature(routineName)
    with open(fileName, "a") as file:
      file.write(self._kernel(routineName))
      file.write(f"{func_signature}")
      file.write(self._call(routineName))
    return func_signature + ";"
