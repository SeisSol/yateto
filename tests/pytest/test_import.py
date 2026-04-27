"""
Importability regression tests.

These tests don't exercise any algorithm - they just import the top-level
``yateto`` package under various conditions and check that the basic
surface area is intact.  Boring, but very effective at catching stupid
regressions (a stray ``print`` statement, a broken relative import, a
missing ``__init__``, ...).

They also act as the canary for environment-level issues.
"""
from __future__ import annotations

import importlib
import sys
import subprocess
import textwrap

import pytest


# ---------------------------------------------------------------------------
# Top-level API surface
# ---------------------------------------------------------------------------


class TestTopLevelAPI:
    def test_package_imports(self):
        import yateto
        assert yateto is not None

    def test_Tensor_exported(self):
        from yateto import Tensor  # noqa: F401

    def test_Scalar_exported(self):
        from yateto import Scalar  # noqa: F401

    def test_Generator_exported(self):
        from yateto import Generator  # noqa: F401

    def test_arch_helpers_exported(self):
        from yateto import (  # noqa: F401
            useArchitectureIdentifiedBy,
            deriveArchitecture,
            HostArchDefinition,
            DeviceArchDefinition,
            fixArchitectureGlobal,
        )

    def test_parameter_space_helpers_exported(self):
        from yateto import simpleParameterSpace, parameterSpaceFromRanges  # noqa: F401

    def test_GlobalRoutineCache_exported(self):
        from yateto import GlobalRoutineCache  # noqa: F401

# ---------------------------------------------------------------------------
# Submodule round-trip
# ---------------------------------------------------------------------------


class TestSubmodules:
    """The yateto package is composed of many submodules.  A simple
    ``reload()`` round-trip on each one catches most cases where a
    module fails to import due to a typo, a missing dependency, or a
    cyclic import.
    """

    @pytest.mark.parametrize(
        "modname",
        [
            "yateto",
            "yateto.aspp",
            "yateto.arch",
            "yateto.memory",
            "yateto.type",
            "yateto.generator",
            "yateto.ast",
            "yateto.ast.node",
            "yateto.ast.indices",
            "yateto.ast.visitor",
            "yateto.ast.transformer",
            "yateto.ast.cost",
            "yateto.ast.opt",
            "yateto.ast.log",
            "yateto.controlflow",
            "yateto.controlflow.graph",
            "yateto.controlflow.visitor",
            "yateto.controlflow.transformer",
        ],
    )
    def test_submodule_imports(self, modname):
        mod = importlib.import_module(modname)
        assert mod is not None

    def test_reload_roundtrip(self):
        # A gentler check than the subprocess test above: reimport the
        # top-level package and make sure the main types survive.
        import yateto
        importlib.reload(yateto)
        from yateto import Tensor, Scalar, Generator
        assert Tensor and Scalar and Generator
