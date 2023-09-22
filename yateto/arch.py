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


def _get_name_and_precision(ident):
  return ident[1:], ident[0].upper()


def getArchitectureIdentifiedBy(ident):
  name, precision = _get_name_and_precision(ident)

  # NOTE: libxsmm currently supports prefetch only for KNL kernels
  arch = {
    'noarch': Architecture(name, precision, 16, False),
    'wsm': Architecture(name, precision, 16, False),
    'snb': Architecture(name, precision, 32, False),
    'hsw': Architecture(name, precision, 32, False),
    'skx': Architecture(name, precision, 64, True),
    'knc': Architecture(name, precision, 64, False),
    'knl': Architecture(name, precision, 64, True),
    'naples': Architecture(name, precision, 32, False),
    'rome': Architecture(name, precision, 32, False),
    'milan': Architecture(name, precision, 32, False),
    'bergamo': Architecture(name, precision, 64, True),
    'thunderx2t99': Architecture(name, precision, 16, False),
    'a64fx': Architecture(name, precision, 64, True),
    'neon': Architecture(name, precision, 16, False),
    'apple-m1': Architecture(name, precision, 16, False),
    'apple-m2': Architecture(name, precision, 16, False),
    'sve128': Architecture(name, precision, 16, False),
    'sve256': Architecture(name, precision, 32, False),
    'sve512': Architecture(name, precision, 64, False),
    'sve1024': Architecture(name, precision, 128, False),
    'sve2048': Architecture(name, precision, 256, False),
    'power9': Architecture(name, precision, 16, False),
  }
  return arch[name]


def getHeterogeneousArchitectureIdentifiedBy(host_arch, device_arch, device_backend):
  device_arch, device_precision = _get_name_and_precision(device_arch)
  host_name, host_precision = _get_name_and_precision(host_arch)

  if (device_precision != host_precision):
    raise ValueError(f'Precision of host and compute arch. must be the same. '
                     f'Given: {host_arch}, {device_arch}')

  if 'sm_' in device_arch:
    alignment = 64
  elif 'gfx' in device_arch: 
    alignment = 128
  elif device_arch in ['dg1', 'bdw', 'skl', 'Gen8', 'Gen9', 'Gen11', 'Gen12LP']:
    alignment = 32
  else:
    raise ValueError(f'Unknown device arch: {device_arch}')

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
