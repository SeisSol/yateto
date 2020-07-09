#!/usr/bin/env python3

import os, errno
import argparse
import importlib.util
from yateto import *
from yateto.ast.visitor import PrettyPrinter, FindTensors, PrintEquivalentSparsityPatterns
from yateto.codegen.code import Cpp

cmdLineParser = argparse.ArgumentParser()
cmdLineParser.add_argument('--arch', type=str, default='dhsw', help='Architecture (e.g. dsnb for double precision on Sandy Bridge).')
cmdLineParser.add_argument('--variant', type=str, default='', help='Example specific variant OpenBLAS, LIBXSMM.')
cmdLineParser.add_argument('--output_dir', type=str, default='./', help='output directory for gen. code')
cmdLineParser.add_argument('example_script', type=str, help='A yateto example script from the examples folder (without file extension).')
cmdLineArgs = cmdLineParser.parse_args()

exampleSpec = importlib.util.find_spec(cmdLineArgs.example_script)
try:
  example = exampleSpec.loader.load_module()
except:
  raise RuntimeError('Could not find example ' + cmdLineArgs.example_script)

targetFlopsPerSec = 40.0e9

variantSuffix = '_' + cmdLineArgs.variant if cmdLineArgs.variant else ''
outDir = os.path.join(cmdLineArgs.output_dir,
                      cmdLineArgs.example_script,
                      cmdLineArgs.arch + variantSuffix)

try:
  if not os.path.exists(outDir):
    os.makedirs(outDir)
except OSError as e:
    if e.errno == errno.EEXIST:
      pass

arch = useArchitectureIdentifiedBy(cmdLineArgs.arch)

g = Generator(arch)
example.add(g)

if hasattr(gemm_configuration, cmdLineArgs.variant):
  concrete_gemm = getattr(gemm_configuration, cmdLineArgs.variant)
  gemm_cfg = GeneratorCollection([concrete_gemm(arch)])
else:
  raise RuntimeError(f'YATETO::ERROR: unknown \"{cmdLineArgs.variant}\" GEMM tool. '
                     f'Please, refer to the documentation')
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