import hashlib
import subprocess
import tempfile
from abc import ABC

from ..cache import RoutineGenerator, GpuRoutineGenerator
from ...gemm_configuration import BLASlike, CodeGenerator, GemmForge, libsmm
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
      candidate = int(value)
      if candidate in specials:
        result = candidate
    except:
      pass
    return result

  def _alpha(self, alpha):
    return self._is_special(alpha, {1})

  def _beta(self, beta):
    return self._is_special(beta, {0,1})

  def generateRoutineName(self, gemm, spp):
    name = self._gemm_cfg.operation_name
    if spp is not None:
      sha = hashlib.md5()
      sha.update(str(spp).encode())
      name += 'sparse_' + sha.hexdigest()
    return '{name}_m{M}_n{N}_k{K}_ldA{LDA}_ldB{LDB}_ldC{LDC}_alpha{alphaSubs}_beta{betaSubs}_alignedA{alignedA}_alignedC{alignedC}_transA{transA}_transB{transB}_{prefetch}'.format(
      name=name,
      alphaSubs=self._alpha(gemm['alpha']),
      betaSubs=self._beta(gemm['beta']),
      **gemm
    )

  def _offset(self, term, offset2, transpose):
    if transpose:
      # swaps elements of tuple if transpose
      offset2 = offset2[::-1]
    return term.memoryLayout.subtensorOffset(topLeftEntry=offset2)

  def _pointer(self, term, offset2, transpose):
    o = self._offset(term, offset2, transpose)
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

          args = [aux.deduce_ptr_arg(d.leftTerm, as_const=True),
                  aux.deduce_offset_arg(d.leftTerm),
                  aux.deduce_ptr_arg(d.rightTerm, as_const=True),
                  aux.deduce_offset_arg(d.rightTerm),
                  aux.deduce_ptr_arg(d.result, as_const=False),
                  aux.deduce_offset_arg(d.result),
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
    elif isinstance(self._gemm_cfg, libsmm):
      aux = BatchedOperationsAux(self._arch.typename)
      gemm = {
        'M':            m.size(),
        'N':            n.size(),
        'K':            k.size(),
        'LDA':          ldA,
        'addrA':        aux.deduce_addresing(d.leftTerm),
        'distA':        d.leftTerm.memoryLayout.requiredReals(),
        'LDB':          ldB,
        'addrB':        aux.deduce_addresing(d.rightTerm),
        'distB':        d.rightTerm.memoryLayout.requiredReals(),
        'LDC':          ldC,
        'addrC':        aux.deduce_addresing(d.result),
        'distC':        d.result.memoryLayout.requiredReals(),
        'alpha':        self._alpha(d.alpha),
        'beta':         self._beta(d.beta),
        'transA': d.transA,
        'transB': d.transB,
      }
      routine_name = 'libsmm_wrapper_m{M}_n{N}_k{K}_ldA{LDA}_{addrA}_{distA}_ldB{LDB}_{addrB}_{distB}_ldC{LDC}_{addrC}_{distC}_alpha{alpha}_beta{beta}_transA{transA}_transB{transB}'.format(**gemm)


      offset_a = self._offset(term=d.leftTerm, offset2=(m.start, k.start), transpose=d.transA)
      offset_b = self._offset(term=d.rightTerm, offset2=(k.start, n.start), transpose=d.transB)
      offset_c = self._offset(term=d.result, offset2=(m.start, n.start), transpose=False)
      args = [aux.deduce_ptr_arg(d.leftTerm, as_const=True),
              f'{aux.deduce_offset_arg(d.leftTerm)} + {offset_a}',
              aux.deduce_ptr_arg(d.rightTerm, as_const=True),
              f'{aux.deduce_offset_arg(d.rightTerm)} + {offset_b}',
              aux.deduce_ptr_arg(d.result, as_const=False),
              f'{aux.deduce_offset_arg(d.result)} + {offset_c}',
              BatchedOperationsAux.NUM_ELEMENTS_NAME,
              BatchedOperationsAux.STREAM_PTR_NAME]
      args = ', '.join(args)

      cpp(f'{routine_name}({args});')

      routineCache.addRoutine(routine_name, LibsmmGemmGen(self._arch, gemm))
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
        'prefetch':     'BL2viaC' if self._arch.enablePrefetch and d.prefetchName is not None else 'pfsigonly',
        'transA': d.transA,
        'transB': d.transB,

      }

      routineName = self.generateRoutineName(gemm, spp)


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
        routineCache.addRoutine(routineName, ExecuteGemmGen(self._arch, gemm, spp, sppRows, self._gemm_cfg))

    return flops

class ExecuteGemmGen(RoutineGenerator):  
  def __init__(self, arch, gemmDescr, spp, sppRows, gemm_cfg):
    self._arch = arch
    self._gemmDescr = gemmDescr
    self._spp = spp
    self._sppRows = sppRows
    self._mode = gemm_cfg.operation_name
    self._cmd = gemm_cfg.cmd
    self._blockSize = gemm_cfg.blockSize(gemmDescr['M'], gemmDescr['N'], gemmDescr['K']) if hasattr(gemm_cfg, 'blockSize') else dict()
  
  def __eq__(self, other):
    return self._arch == other._arch and \
           self._gemmDescr == other._gemmDescr and \
           self._spp == other._spp
  
  def header(self, cpp):
    with cpp.PPIfndef('NDEBUG'):
      cpp('extern long long libxsmm_num_total_flops;')
      cpp('extern long long pspamm_num_total_flops;')
    with cpp.PPIf('defined( __SSE3__) || defined(__MIC__)'):
      cpp.includeSys('immintrin.h')

  def _callGenerator(self, argList):
    resultCode = 1
    try:
      resultCode = subprocess.call([str(arg) for arg in argList])
    except OSError:
      raise RuntimeError('GEMM code generator executable "{}" not found. (Make sure to add the folder containing the executable to your PATH.)'.format(self._cmd))
    if resultCode != 0:
      raise RuntimeError('GEMM code generator executable "{}" failed. Thus, the kernel generation may be incomplete.'.format(self._cmd))
  
  def __call__(self, routineName, fileName):
    cpu_arch = self._arch.host_name if self._arch.host_name else self._arch.name

    if self._mode == 'pspamm':
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
        cpu_arch,
        '--prefetching',
        self._gemmDescr['prefetch'],
        '--output_funcname',
        routineName,
        '--output_filename',
        fileName,
        '--precision',
        self._arch.precision
      ]
      for key, val in self._blockSize.items():
        argList.extend(['--' + key, val])
    else:
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
        'hsw' if cpu_arch == 'rome' else cpu_arch, # libxsmm has no support for rome, hsw works well in practice
        self._gemmDescr['prefetch'],
        self._arch.precision + 'P'
      ]
    if self._spp is not None:
      cols = self._gemmDescr['K'] if self._gemmDescr['LDA'] == 0 else self._gemmDescr['N']
      rows = self._gemmDescr['M'] if self._gemmDescr['LDA'] == 0 else self._gemmDescr['K']
      if self._mode == 'pspamm':
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
        if self._mode == 'pspamm':
          argList.append('--mtx_filename')
        argList.append(temp.name)
        self._callGenerator(argList)
    else:
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
               spp,
               spp_rows,
               gemm_cfg):
    super().__init__(arch, gemm_descr, spp, spp_rows, gemm_cfg)

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

    prefetch_flag =  "LIBXSMM_GEMM_PREFETCH_SIGONLY" if not self._arch.enablePrefetch else "LIBXSMM_GEMM_PREFETCH_BL2_VIA_C"

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

class LibsmmGemmGen(GpuRoutineGenerator):
  def __init__(self, arch, gemm_descr):
      self.arch = arch
      self.gemm_descr = gemm_descr

  def __eq__(self, other):
    return self.arch == other.arch and self.gemm_descr == other.gemm_descr

  def header(self, cpp):
    cpp.include('smm/configuration.hpp')
    cpp.include('smm/gemm.hpp')
    cpp.includeSys('CL/sycl.hpp')

  def _functionSignature(self, routineName):
    typ = self.arch.typename
    stars = lambda x: '**' if x == 'pointer_based' else '*'
    starsA = stars(self.gemm_descr['addrA'])
    starsB = stars(self.gemm_descr['addrB'])
    starsC = stars(self.gemm_descr['addrC'])
    return f'void {routineName}({typ} const{starsA} A, int offsetA, {typ} const{starsB} B, int offsetB, {typ}{starsC} C, int offsetC, unsigned {BatchedOperationsAux.NUM_ELEMENTS_NAME}, void* {BatchedOperationsAux.STREAM_PTR_NAME})'

  def address_mode(self, addr, dist):
      if addr == 'pointer_based':
          return 'smm::pointers{}'
      elif addr == 'none':
          return 'smm::strided{0}'
      elif addr == 'strided':
          return f'smm::strided{{{dist}}}'
      raise NameError(addr)

  def __call__(self, routineName, fileName):
    func_signature = self._functionSignature(routineName)
    with open(fileName, "a") as f:
      aA = self.address_mode(self.gemm_descr['addrA'], self.gemm_descr['distA'])
      aB = self.address_mode(self.gemm_descr['addrB'], self.gemm_descr['distB'])
      aC = self.address_mode(self.gemm_descr['addrC'], self.gemm_descr['distC'])

      T = lambda x: 'smm::transpose::T' if x else 'smm::transpose::N'
      tA = T(self.gemm_descr['transA'])
      tB = T(self.gemm_descr['transB'])
      alpha = self.gemm_descr['alpha']
      beta = self.gemm_descr['beta']

      f.write(f'{func_signature} {{\n')
      f.write('  static auto kernel = smm::make_gemm<{typ}>({tA}, {tB}, {M}, {N}, {K}, {LDA}, {aA}, {LDB}, {aB}, {LDC}, {aC}, *static_cast<::sycl::queue*>({stream}));\n'.format(typ=self.arch.typename, aA=aA, aB=aB, aC=aC, tA=tA, tB=tB, stream=BatchedOperationsAux.STREAM_PTR_NAME, **self.gemm_descr))
      f.write(f'  kernel({alpha}, A, B, {beta}, C, {BatchedOperationsAux.NUM_ELEMENTS_NAME}).wait();\n')
      f.write('}\n')
    return func_signature + ";"
