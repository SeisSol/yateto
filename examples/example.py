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

outDir = cmdLineArgs.example_script + '/generated_code'
try:
  os.makedirs(outDir)
except OSError as e:
  if e.errno == errno.EEXIST:
    pass

arch = useArchitectureIdentifiedBy(cmdLineArgs.arch)

g = Generator(arch)
example.add(g)
g.generate(outDir)

tensors = []
for kernel in g.kernels():
  title = 'AST of {}'.format(kernel.name)
  print(title)
  print('='*len(title))
  PrettyPrinter().visit(kernel.ast)
  print(' ')

  tensors.extend(FindTensors().visit(kernel.ast).values())

tensors = set((tensor.baseName(), tensor.group()) for tensor in tensors)
formatArrayName = lambda tensor: '{0}__{1}'.format(tensor[0], '_'.join([str(g) for g in tensor[1]]))
formatGroup = lambda tensor: ','.join([str(g) for g in tensor[1]])

with Cpp(os.path.join(cmdLineArgs.example_script, 'performance.cpp')) as cpp:
  cpp.includeSys('cstdlib')
  cpp.includeSys('cstdio')
  cpp.includeSys('cmath')
  cpp.include('generated_code/kernel.h')
  cpp.include('generated_code/tensor.h')
  cpp.include('Stopwatch.h')
  cpp.include('Util.h')
  cpp('using namespace yateto;')
  with cpp.Function('main', arguments='int argc, char** argv', returnType='int'):
    cpp('int _fixedReps = (argc >= 2) ? atoi(argv[1]) : -1;')
    cpp('int _reps;')
    for tensor in tensors:
      cpp('real {0}[tensor::{1}::size({2})] __attribute__((aligned(ALIGNMENT)));'.format(formatArrayName(tensor), tensor[0], formatGroup(tensor)))
    for tensor in tensors:
      cpp('fillWithStuff({0}, tensor::{1}::size({2}));'.format(formatArrayName(tensor), tensor[0], formatGroup(tensor)))
    cpp('Stopwatch _sw;');
    cpp('double _time, _flops;')
    cpp('printf("kernel,repetitions,time,numflop,gflops\\n");')
    for kernel in g.kernels():
      cpp('_reps = _fixedReps;')
      with cpp.If('_reps < 0'):
        cpp('_reps = ceil({0}/kernel::{1}::HardwareFlops);'.format(targetFlopsPerSec, kernel.name))
      kobj = '_kernel_{0}'.format(kernel.name)
      cpp('kernel::{} {};'.format(kernel.name, kobj))
      for key,tensor in FindTensors().visit(kernel.ast).items():
        cpp('{0}.{1} = {2};'.format(kobj, key, formatArrayName((tensor.baseName(), tensor.group()))))
      cpp('{}.execute();'.format(kobj))
      cpp('_sw.start();')
      with cpp.For('int i = 0; i < _reps; ++i'):
        cpp('{}.execute();'.format(kobj))
      cpp('_time = _sw.stop();')
      cpp('_flops = static_cast<double>(kernel::{0}::HardwareFlops) * _reps / _time / 1.0e9;'.format(kernel.name))
      cpp('printf("{0},%u,%lf,%u,%lf\\n", _reps, _time, kernel::{0}::HardwareFlops, _flops);'.format(kernel.name))
    cpp('return 0;')
