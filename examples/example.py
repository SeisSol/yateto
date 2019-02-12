#!/usr/bin/env python3

import sys
sys.path.append('..')

import os, errno
import argparse
import importlib.util
from yateto import *
from yateto.ast.visitor import PrettyPrinter, FindTensors, PrintEquivalentSparsityPatterns
from yateto.codegen.code import Cpp

cmdLineParser = argparse.ArgumentParser()
cmdLineParser.add_argument('--arch', type=str, default='dhsw', help='Architecture (e.g. dsnb for double precision on Sandy Bridge).')
cmdLineParser.add_argument('--variant', type=str, default='', help='Example specific variant (e.g. onlyblas).')
cmdLineParser.add_argument('example_script', type=str, help='A yateto example script from the examples folder (without file extension).')
cmdLineArgs = cmdLineParser.parse_args()

exampleSpec = importlib.util.find_spec(cmdLineArgs.example_script)
try:
  example = exampleSpec.loader.load_module()
except:
  raise RuntimeError('Could not find example ' + cmdLineArgs.example_script)

targetFlopsPerSec = 40.0e9

variantSuffix = '_' + cmdLineArgs.variant if cmdLineArgs.variant else ''
outDir = os.path.join(cmdLineArgs.example_script, cmdLineArgs.arch + variantSuffix)
try:
  os.makedirs(outDir)
except OSError as e:
  if e.errno == errno.EEXIST:
    pass

arch = useArchitectureIdentifiedBy(cmdLineArgs.arch)

g = Generator(arch)
example.add(g)
gemm_cfg = example.gemm_cfg(arch, cmdLineArgs.variant) if hasattr(example, 'gemm_cfg') else None
g.generate(outDir, gemm_cfg=gemm_cfg)

for kernel in g.kernels():
  title = 'AST of {}'.format(kernel.name)
  print(title)
  print('='*len(title))
  PrettyPrinter().visit(kernel.ast)
  print(' ')

printEqspp = example.printEqspp() if hasattr(example, 'printEqspp') else False
if printEqspp:
  for kernel in g.kernels():
    d = os.path.join(outDir, kernel.name)
    os.makedirs(d, exist_ok=True)
    PrintEquivalentSparsityPatterns(d).visit(kernel.ast)

formatArrayName = lambda tensor: '{0}__{1}'.format(tensor.baseName(), '_'.join([str(g) for g in tensor.group()]))
formatGroup = lambda tensor: ','.join([str(g) for g in tensor.group()])

trashTheCache = example.cold() if hasattr(example, 'cold') else False
# 128 MB to trash the cache
trashSize = 128 * 1024**2

with Cpp(os.path.join(outDir, 'trashTheCache.cpp')) as cpp:
  with cpp.Function('trashTheCache', arguments='double* trash, int size'):
        with cpp.For('int i = 0; i < size; ++i'):
          cpp('trash[i] += trash[i];')

with Cpp(os.path.join(outDir, 'performance.cpp')) as cpp:
  cpp.includeSys('cstdlib')
  cpp.includeSys('cstdio')
  cpp.includeSys('cmath')
  cpp.include('kernel.h')
  cpp.include('tensor.h')
  cpp.include('Stopwatch.h')
  cpp.include('Util.h')
  cpp('using namespace yateto;')
  cpp.functionDeclaration('trashTheCache', arguments='double* trash, int size')
  with cpp.Function('main', arguments='int argc, char** argv', returnType='int'):
    cpp('int _fixedReps = (argc >= 2) ? atoi(argv[1]) : -1;')
    cpp('int _reps, _error;')
    if trashTheCache:
      cpp('double* _trash = new double[{}];'.format(trashSize))
    cpp('Stopwatch _sw;');
    cpp('double _time, _nzflops, _flops;')
    cpp('printf("kernel,repetitions,time,numnzflop,numflop,nzgflops,gflops\\n");')
    for kernel in g.kernels():
      with cpp.AnonymousScope():
        tensors = FindTensors().visit(kernel.ast).items()
        for key,tensor in tensors:
          arrayName = formatArrayName(tensor)
          cpp('real* {};'.format(arrayName))
          cpp('_error = posix_memalign(reinterpret_cast<void**>(&{0}), ALIGNMENT, tensor::{1}::size({2})*sizeof(real));'.format(
                arrayName,
                tensor.baseName(),
                formatGroup(tensor)))
        for key,tensor in tensors:
          cpp('fillWithStuff({0}, tensor::{1}::size({2}));'.format(formatArrayName(tensor), tensor.baseName(), formatGroup(tensor)))
        if trashTheCache:
          cpp('_reps = 1;')
        else:
          cpp('_reps = _fixedReps;')
          with cpp.If('_reps < 0'):
            cpp('_reps = ceil({0}/kernel::{1}::HardwareFlops);'.format(targetFlopsPerSec, kernel.name))
        kobj = '_kernel_{0}'.format(kernel.name)
        cpp('kernel::{} {};'.format(kernel.name, kobj))
        for key,tensor in tensors:
          cpp('{0}.{1} = {2};'.format(kobj, key, formatArrayName(tensor)))
        if trashTheCache:
          cpp('trashTheCache(_trash, {});'.format(trashSize))
          cpp('_sw.start();')
          cpp('{}.execute();'.format(kobj))
        else:
          cpp('{}.execute();'.format(kobj))
          cpp('_sw.start();')
          with cpp.For('int i = 0; i < _reps; ++i'):
            cpp('{}.execute();'.format(kobj))
        cpp('_time = _sw.stop();')
        cpp('_nzflops = static_cast<double>(kernel::{0}::NonZeroFlops) * _reps / _time / 1.0e9;'.format(kernel.name))
        cpp('_flops = static_cast<double>(kernel::{0}::HardwareFlops) * _reps / _time / 1.0e9;'.format(kernel.name))
        cpp('printf("{0},%u,%lf,%lu,%lu,%lf,%lf\\n", _reps, _time, kernel::{0}::NonZeroFlops, kernel::{0}::HardwareFlops, _nzflops, _flops);'.format(kernel.name))
        for key,tensor in tensors:
          cpp('free({});'.format(formatArrayName(tensor)))
    if trashTheCache:
      cpp('delete[] _trash;')
    cpp('return 0;')
