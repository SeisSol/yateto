"""Putting adjacent contractions into the chain a backend takes them as."""

from ... import ir
from .factory import Description, available, generator


def fuseChains(region, arch, gemm_cfg, target, attrs=None):
  """Replace every run of adjacent matrix products by the chain they form.

  A backend that takes a chain writes one kernel for all of it, and what one
  product leaves behind the next reads without going through the memory the
  kernel was given. Which products it may take together is a question about
  what stands next to what, so it is asked of the region.

  Two statements that do not run together are not next to each other here:
  a guarded statement stands in a region of its own, so a chain never spans
  one guard and another, nor a guard and no guard at all.
  """
  if not available(gemm_cfg, target):
    return region

  for op in region.ops:
    for nested in op.regions():
      fuseChains(nested, arch, gemm_cfg, target, attrs)

  ops = []
  run = []
  for op in region.ops:
    if isinstance(op, ir.LoopOverGEMM) and op.isPureGEMM():
      run.append(op)
      continue
    ops.extend(_chain(run, arch, gemm_cfg, target, attrs))
    run = []
    ops.append(op)
  ops.extend(_chain(run, arch, gemm_cfg, target, attrs))

  region.ops = ops
  return region


def _chain(statements, arch, gemm_cfg, target, attrs):
  """The run as one statement, or nothing where the run is empty."""
  if not statements:
    return []
  description = Description(statements)
  return [ir.FusedGEMMs(statements,
                        generator(arch, description, gemm_cfg, target, attrs))]
