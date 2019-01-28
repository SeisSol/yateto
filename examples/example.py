#!/usr/bin/env python3

import sys
sys.path.append('..')

import os, errno
import argparse
import importlib.util
from yateto import *
from yateto.ast.visitor import PrettyPrinter, FindTensors
from yateto.codegen.code import Cpp

cmdLineParser = argparse.ArgumentParser()
cmdLineParser.add_argument('--arch', type=str, default='dhsw', help='Architecture (e.g. dsnb for double precision on Sandy Bridge).')
cmdLineParser.add_argument('example_script', type=str, help='A yateto example script from the examples folder (without file extension).')
cmdLineArgs = cmdLineParser.parse_args()

exampleSpec = importlib.util.find_spec(cmdLineArgs.example_script)
try:
  example = exampleSpec.loader.load_module()
except:
  raise RuntimeError('Could not find example ' + cmdLineArgs.example_script)

targetFlopsPerSec = 40.0e9

outDir = cmdLineArgs.example_script
try:
  os.makedirs(outDir)
except OSError as e:
  if e.errno == errno.EEXIST:
    pass

arch = useArchitectureIdentifiedBy(cmdLineArgs.arch)

g = Generator(arch)
example.add(g)
gemm_cfg = example.gemm_cfg() if hasattr(example, 'gemm_cfg') else None
g.generate(outDir, gemm_cfg=gemm_cfg)

for kernel in g.kernels():
  title = 'AST of {}'.format(kernel.name)
  print(title)
  print('='*len(title))
  PrettyPrinter().visit(kernel.ast)
  print(' ')

formatArrayName = lambda tensor: '{0}__{1}'.format(tensor.baseName(), '_'.join([str(g) for g in tensor.group()]))
formatGroup = lambda tensor: ','.join([str(g) for g in tensor.group()])

with Cpp(os.path.join(outDir, 'performance.cpp')) as cpp:
  cpp.includeSys('cstdlib')
  cpp.includeSys('cstdio')
  cpp.includeSys('cmath')
  cpp.include('kernel.h')
  cpp.include('tensor.h')
  cpp.include('Stopwatch.h')
  cpp.include('Util.h')
  cpp('using namespace yateto;')
  with cpp.Function('main', arguments='int argc, char** argv', returnType='int'):
    cpp('int _fixedReps = (argc >= 2) ? atoi(argv[1]) : -1;')
    cpp('int _reps, _error;')
    cpp('Stopwatch _sw;');
    cpp('double _time, _flops;')
    cpp('printf("kernel,repetitions,time,numflop,gflops\\n");')
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
        cpp('_reps = _fixedReps;')
        with cpp.If('_reps < 0'):
          cpp('_reps = ceil({0}/kernel::{1}::HardwareFlops);'.format(targetFlopsPerSec, kernel.name))
        kobj = '_kernel_{0}'.format(kernel.name)
        cpp('kernel::{} {};'.format(kernel.name, kobj))
        for key,tensor in tensors:
          cpp('{0}.{1} = {2};'.format(kobj, key, formatArrayName(tensor)))
        cpp('{}.execute();'.format(kobj))
        cpp('_sw.start();')
        with cpp.For('int i = 0; i < _reps; ++i'):
          cpp('{}.execute();'.format(kobj))
        cpp('_time = _sw.stop();')
        cpp('_flops = static_cast<double>(kernel::{0}::HardwareFlops) * _reps / _time / 1.0e9;'.format(kernel.name))
        cpp('printf("{0},%u,%lf,%lu,%lf\\n", _reps, _time, kernel::{0}::HardwareFlops, _flops);'.format(kernel.name))
        for key,tensor in tensors:
          cpp('free({});'.format(formatArrayName(tensor)))
    cpp('return 0;')
