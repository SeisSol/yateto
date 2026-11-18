"""
Tests for ``yateto.input`` - the matrix files, and what is declared by parsing them.

Most of SeisSol's matrices are not constructed by the code that knows what
they are: a parser makes them, clones included. So the parse is where their
attributes are given, keyed by the name a matrix has in the file, as
``clones`` is.
"""
from __future__ import annotations

import json
import tempfile

import pytest

from yateto import Generator, GeneratorCollection, Tensor
from yateto.arch import useArchitectureIdentifiedBy
from yateto.codegen.factory import ExportGenerator
from yateto.input import parseJSONMatrixFile, parseXMLMatrixFile

INVARIANT = {"temporal": "invariant"}
STARS = ["star(0)", "star(1)", "star(2)"]


@pytest.fixture
def equation(tmp_path):
    """An equation file as SeisSol has them: a pattern to clone, and another."""
    path = tmp_path / "equation.json"
    path.write_text(json.dumps([
        {"name": "star", "rows": 3, "columns": 3,
         "entries": [[1, 1], [2, 3], [3, 2]]},
        {"name": "ET", "rows": 3, "columns": 3, "entries": [[1, 1]]},
    ]))
    return str(path)


@pytest.fixture
def constants(tmp_path):
    path = tmp_path / "constants.xml"
    path.write_text(
        '<matrices>'
        '<matrix name="kDivM" rows="2" columns="2">'
        '<entry row="1" column="2" value="0.5"/></matrix>'
        '<matrix name="M" rows="2" columns="2">'
        '<entry row="1" column="1" value="2.0"/></matrix>'
        '</matrices>')
    return str(path)


def _claims(db, names):
    return {name: db.byName(name).attributes().as_dict() for name in names}


class TestAttributesFromAJSONFile:
    def test_without_attrs_nothing_is_claimed(self, equation):
        db = parseJSONMatrixFile(equation, {"star": STARS})
        assert _claims(db, STARS + ["ET"]) == {name: {} for name in STARS + ["ET"]}

    def test_a_matrix_gives_its_clones_its_attributes(self, equation):
        db = parseJSONMatrixFile(equation, {"star": STARS}, attrs={"star": INVARIANT})
        assert _claims(db, STARS) == {name: INVARIANT for name in STARS}
        assert db.ET.attributes().as_dict() == {}

    def test_a_matrix_that_is_not_cloned_gets_them_too(self, equation):
        db = parseJSONMatrixFile(equation, attrs={"star": INVARIANT, "ET": INVARIANT})
        assert _claims(db, ["star", "ET"]) == {"star": INVARIANT, "ET": INVARIANT}

    def test_a_name_the_file_does_not_contain_is_refused(self, equation):
        # A claim for nothing attaches to nothing, and nothing would say so.
        with pytest.raises(ValueError, match=r"no matrix named 'stars'.*has ET, star"):
            parseJSONMatrixFile(equation, attrs={"stars": INVARIANT})

    def test_a_clone_is_named_by_the_matrix_it_is_cloned_from(self, equation):
        with pytest.raises(ValueError, match=r"no matrix named 'star\(0\)'.*cloned from"):
            parseJSONMatrixFile(equation, {"star": STARS}, attrs={"star(0)": INVARIANT})

    def test_attributes_are_checked_as_a_tensor_checks_them(self, equation):
        with pytest.raises(ValueError, match=r"equation\.json, matrix 'star': unknown tensor attribute\(s\) 'temporl'"):
            parseJSONMatrixFile(equation, attrs={"star": {"temporl": "invariant"}})
        with pytest.raises(ValueError, match=r"'temporal' takes 'invariant', not 'always'"):
            parseJSONMatrixFile(equation, attrs={"star": {"temporal": "always"}})

    def test_the_mapping_given_is_not_kept(self, equation):
        given = {"star": dict(INVARIANT)}
        db = parseJSONMatrixFile(equation, attrs=given)
        given["star"]["temporal"] = "something else"
        assert db.star.attributes().timeInvariant is True


class TestAttributesFromAnXMLFile:
    def test_a_constant_can_be_given_them(self, constants):
        db = parseXMLMatrixFile(constants, attrs={"kDivM": INVARIANT})
        assert db.kDivM.attributes().timeInvariant is True
        assert db.kDivM.is_compute_constant()
        assert db.M.attributes().as_dict() == {}

    def test_a_name_the_file_does_not_contain_is_refused(self, constants):
        with pytest.raises(ValueError, match=r"no matrix named 'kDivMT'"):
            parseXMLMatrixFile(constants, attrs={"kDivMT": INVARIANT})


class _Collector(ExportGenerator):
    def __init__(self, arch, attrs=None):
        super().__init__(arch, attrs)
        self.tensors = []

    def add_kernel(self, description):
        self.tensors = description["tensors"]


def _exported(statement):
    arch = useArchitectureIdentifiedBy("dhsw", "dsm_86", "cuda")
    made = {}

    def make(a, attrs=None):
        made["it"] = _Collector(a, attrs)
        return made["it"]

    generator = Generator(arch)
    generator.add("k", statement, target="gpu")
    with tempfile.TemporaryDirectory() as out:
        generator.generate(out, gemm_cfg=GeneratorCollection([]),
                           routine_exporters={"gpu": make})
    return {tensor["name"]: tensor for tensor in made["it"].tensors}


class TestTheClaimReachesTheExport:
    """From the parse, through the kernel, to the description."""

    @staticmethod
    def _derivative(db):
        Q = Tensor("Q", (3, 3))
        dQ = Tensor("dQ", (3, 3))
        return dQ["kp"] <= sum(Q["kq"] * db.byName(star)["qp"] for star in STARS)

    def test_every_clone_is_described_with_it(self, equation):
        db = parseJSONMatrixFile(equation, {"star": STARS}, attrs={"star": INVARIANT})
        described = _exported(self._derivative(db))
        assert {name: described[name].get("attrs") for name in STARS} == \
            {name: INVARIANT for name in STARS}
        assert "attrs" not in described["Q"]

    def test_unclaimed_matrices_are_described_as_before(self, equation):
        db = parseJSONMatrixFile(equation, {"star": STARS})
        described = _exported(self._derivative(db))
        assert all("attrs" not in tensor for tensor in described.values())
