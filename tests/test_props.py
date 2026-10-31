"""n2s_props(Joback 法ほかの物性の推算)のテスト。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pytest
from rdkit import Chem

import n2s_props as pr


def test_joback_acetone_matches_literature():       # Joback & Reid の例: アセトン
    r = pr.joback(Chem.MolFromSmiles("CC(C)=O"))
    assert r["Tb"] == pytest.approx(322.11, abs=0.01) and r["Tc"] == pytest.approx(500.56, abs=0.01)
    assert r["Pc"] == pytest.approx(48.02, abs=0.01) and r["Vc"] == pytest.approx(209.5, abs=0.01)
    assert r["Tm"] == pytest.approx(173.5, abs=0.01) and r["Hf"] == pytest.approx(-217.83, abs=0.01)
    assert r["Gf"] == pytest.approx(-154.54, abs=0.01)


def test_aromatic_and_functional_groups_counted():
    g = pr.joback_groups(Chem.MolFromSmiles("OC(=O)c1ccccc1"))      # 安息香酸
    assert g == {"-COOH": 1, "r=C<": 1, "r=CH-": 5}
    assert pr.joback_groups(Chem.MolFromSmiles("c1ccccc1[N+](=O)[O-]"))["-NO2"] == 1
    assert pr.joback_groups(Chem.MolFromSmiles("c1ccncc1"))["r-N="] == 1


def test_unsupported_atom_gives_na_but_other_properties_remain():
    rows = {k: v for k, _, v, _, _ in pr.properties(Chem.MolFromSmiles("C[Si](C)(C)C"))}
    assert rows["Tb"] == "N/A" and rows["logP"] != "N/A" and rows["pKa"] == "N/A" and rows["henry"] == "N/A"


def test_hydrocarbon_tpsa_zero_and_largest_fragment_used():
    rows = {k: v for k, _, v, _, _ in pr.properties(Chem.MolFromSmiles("c1ccccc1.[Na+].[Cl-]"))}
    assert float(rows["tPSA"]) == 0.0 and rows["Tb"] != "N/A"
