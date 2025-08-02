##
# @file
# This file is part of SeisSol.
#
# @author Carsten Uphoff (c.uphoff AT tum.de, http://www5.in.tum.de/wiki/index.php/Carsten_Uphoff,_M.Sc.)
#
# @section LICENSE
# Copyright (c) 2015-2018, SeisSol Group
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from this
#    software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# @section DESCRIPTION
#

from .memory import DenseMemoryLayout
from collections import namedtuple
from typing import Union
import re

class Architecture(object):
  def __init__(self,
               name,
               precision,
               alignment,
               enablePrefetch=False,
               backend='cpp',
               host_name=None):
    """

    Args:
      name (str): name of the compute (main) architecture e.g., skx, thunderx2t99, power9
        sm_60, sm_61, etc.,
      backend (str): backend name e.g., cpp, cuda, hip, oneapi, hipsycl
      precision (str): either 'd' or 's' character which stands for 'double' or 'single' precision
      alignment (int): length of a cache line in bytes
      enablePrefetch (bool): indicates whether the compute (main) architecture supports
          data prefetching
      host_name (str): name of the host (CPU) architecture. If the code is intended to be generated
          for a CPU-like architecture then the field should be equal to None.
    """
    self.name = name
    self.backend = backend
    self.host_name = host_name

    self.precision = precision.upper()
    if self.precision == 'D':
      self.bytesPerReal = 8
      self.typename = 'double'
      self.epsilon = 2.22e-16
    elif self.precision == 'S':
      self.bytesPerReal = 4
      self.typename = 'float'
      self.epsilon = 1.19e-7
    else:
      raise ValueError(f'Unknown precision type {self.precision}')
    self.alignment = alignment
    assert self.alignment % self.bytesPerReal == 0
    self.alignedReals = self.alignment // self.bytesPerReal
    self.enablePrefetch = enablePrefetch

    self.uintTypename = 'unsigned'
    self.ulongTypename = 'unsigned long'

    self._tmpStackLimit = 524288
    self.is_accelerator = backend != 'cpp' and self.host_name != None

    self.host_name = self.host_name or self.name

  def setTmpStackLimit(self, tmpStackLimit):
    self._tmpStackLimit = tmpStackLimit

  def alignedLower(self, index):
    return index - index % self.alignedReals

  def alignedUpper(self, index):
    return index + (self.alignedReals - index % self.alignedReals) % self.alignedReals

  def alignedShape(self, shape):
    return (self.alignedUpper(shape[0]),) + shape[1:]

  def checkAlignment(self, offset):
    return offset % self.alignedReals == 0

  def formatConstant(self, constant):
    return str(constant) + ('f' if self.precision == 'S' else '')

  def onHeap(self, numReals):
    return (numReals * self.bytesPerReal) > self._tmpStackLimit
  
  def __eq__(self, other):
    return self.name == other.name

def _get_name_and_precision(ident):
  return ident[1:], ident[0].upper()

def getHostArchProperties(name):
  arch = {
    'noarch': (16, False),
    'wsm': (16, False),
    'snb': (32, False),
    'hsw': (32, True),
    'skx': (64, True),
    'knc': (64, False),
    'knl': (64, True),
    'naples': (32, True),
    'rome': (32, True),
    'milan': (32, True),
    'bergamo': (64, True),
    'turin': (64, True),
    'thunderx2t99': (16, True),
    'a64fx': (64, True),
    'neon': (16, True),
    'apple-m1': (16, True),
    'apple-m2': (16, True),
    'apple-m3': (16, True),
    'apple-m4': (16, True),
    'sve128': (16, True),
    'sve256': (32, True),
    'sve512': (64, True),
    'sve1024': (128, True),
    'sve2048': (256, True),
    'power9': (16, False),
    'power10': (16, False),
    'power11': (16, False),
    'rvv128': (16, True),
    'rvv256': (32, True),
    'rvv512': (64, True),
    'rvv1024': (128, True),
    'rvv2048': (256, True),
    'avx2-128': (16, True),
    'avx2-256': (32, True),
    'avx10-128': (16, True),
    'avx10-256': (32, True),
    'avx10-512': (64, True),
    'lsx': (16, True),
    'lasx': (32, True),
  }
  if name in arch:
    return arch[name]
  else:
    return (None, None)

def getArchitectureIdentifiedBy(ident):
  name, precision = _get_name_and_precision(ident)

  alignment, prefetch = getHostArchProperties(name)
  return Architecture(name, precision, alignment, prefetch)


def getHeterogeneousArchitectureIdentifiedBy(host_arch, device_arch, device_backend):
  device_arch, device_precision = _get_name_and_precision(device_arch)
  host_name, host_precision = _get_name_and_precision(host_arch)

  if (device_precision != host_precision):
    raise ValueError(f'Precision of host and compute arch. must be the same. '
                     f'Given: {host_arch}, {device_arch}')

  if device_arch.startswith('sm_'):
    alignment = 64
  elif device_arch.startswith('gfx'): 
    alignment = 128
  elif re.match(r"\d+_\d+_\d+", device_arch):
    alignment = 32
  else:
    print(f'Unknown device arch: {device_arch}. Setting alignment to 32.')
    alignment = 32

  return Architecture(device_arch, device_precision, alignment, False, device_backend, host_name)


def useArchitectureIdentifiedBy(host_arch, device_arch=None, device_backend=None):
  if not (device_arch or device_backend):
    arch = getArchitectureIdentifiedBy(host_arch)

  elif (device_arch and device_backend):
    arch = getHeterogeneousArchitectureIdentifiedBy(host_arch, device_arch, device_backend)
  else:
    raise ValueError(f'given an incomplete set of input parameters: '
                     f'{host_arch}, {device_arch}, {device_backend}')

  DenseMemoryLayout.setAlignmentArch(arch)
  return arch

HostArchDefinition = namedtuple('HostArchDefinition', 'archname precision alignment prefetch')
DeviceArchDefinition = namedtuple('DeviceArchDefinition', 'archname vendor backend precision alignment')

def deriveArchitecture(host_def: HostArchDefinition, device_def: Union[DeviceArchDefinition, None]):
  alignment, prefetch = getHostArchProperties(host_def.archname)

  if host_def.alignment is not None:
    alignment = host_def.alignment
  if host_def.prefetch is not None:
    prefetch = host_def.prefetch
  
  if alignment is None:
    raise NotImplementedError(f'The architecture {host_def.archname} is unknown to Yateto, and no custom alignment was given')
  
  if prefetch is None:
    raise NotImplementedError(f'The architecture {host_def.archname} is unknown to Yateto, and no custom prefetching info was given')

  if device_def is not None:
    assert host_def.precision == device_def.precision
    alignment = device_def.alignment
    if alignment is None:
      if device_def.vendor == 'nvidia':
        alignment = 64
      elif device_def.vendor == 'amd':
        alignment = 128
      elif device_def.vendor == 'intel':
        alignment = 32
      else:
        print(f'Unknown device vendor: {device_def.vendor}. Setting alignment to 32.')
        alignment = 32

    return Architecture(device_def.archname, device_def.precision, alignment, False, device_def.backend, host_def.archname)
  else:
    return Architecture(host_def.archname, host_def.precision, alignment, prefetch)

def fixArchitectureGlobal(arch):
  DenseMemoryLayout.setAlignmentArch(arch)
