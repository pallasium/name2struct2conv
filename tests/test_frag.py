"""n2s_frag(規則ベースのフラグメント予測)のテスト。OPSIN もモデルも使わない。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pytest
from rdkit import Chem

import n2s_frag as fr


def peaks(smi, ion="[M+H]+", engine="rules"):
    ps, note, prec = fr.predict(Chem.MolFromSmiles(smi), ion, engine=engine)
    return ps, note, prec


def has(ps, mz, tol=0.002):
    return [p for p in ps if abs(p.mz - mz) < tol]


def test_aspirin_known_fragments():
    ps, _, prec = peaks("CC(=O)Oc1ccccc1C(=O)O")          # アスピリン [M+H]+ 181.0495
    assert prec == pytest.approx(181.0495, abs=1e-3)
    assert has(ps, 139.0390) and "C2H2O" in has(ps, 139.0390)[0].label      # −ケテン(実測でも強い)
    assert has(ps, 121.0284)                                                # −CH3COOH → サリチル酸系
    assert has(ps, 43.0178)                                                 # アセチリウム CH3CO+
    assert has(ps, 163.0390)                                                # −H2O


def test_benzylamine_gives_tropylium():
    ps, _, prec = peaks("NCc1ccccc1")                       # [M+H]+ 108.0808
    assert prec == pytest.approx(108.0808, abs=1e-3)
    p91 = has(ps, 91.0542)
    assert p91 and p91[0].formula == "C7H7+" and p91[0].rel == pytest.approx(100.0, abs=1e-6)   # 最強


def test_negative_mode_decarboxylation():
    ps, _, prec = peaks("OC(=O)c1ccccc1", "[M-H]-")        # 安息香酸 [M-H]- 121.0295 → −CO2 → 77.0397
    assert prec == pytest.approx(121.0295, abs=1e-3)
    p = has(ps, 77.0397)
    assert p and p[0].rel == pytest.approx(100.0, abs=1e-6)


def test_precursor_included_and_sorted():
    ps, _, prec = peaks("CCO")
    assert [p.mz for p in ps] == sorted(p.mz for p in ps)
    assert any(p.kind == "precursor" and abs(p.mz - prec) < 1e-9 for p in ps)
    assert all(p.rel <= 100.0 + 1e-9 for p in ps)


def test_charged_molecule_is_rejected():
    with pytest.raises(ValueError):
        fr.predict(Chem.MolFromSmiles("C[N+](C)(C)C"), "[M+H]+", engine="rules")


def test_cleavage_peaks_carry_atoms_and_smiles():
    ps, _, _ = peaks("NCc1ccccc1")
    p = has(ps, 91.0542)[0]
    assert p.kind in ("cleavage", "loss") and p.atoms


def test_limonene_rda_base_peak():
    ps, _, prec = peaks("CC1=CCC(CC1)C(C)=C", "M+")          # リモネン M+• 136.1247、塩基ピークは RDA の 68
    assert prec == pytest.approx(136.1247, abs=1e-3)
    top = max((p for p in ps if p.kind != "precursor"), key=lambda p: p.rel)
    assert top.mz == pytest.approx(68.0621, abs=2e-3) and top.kind == "ring" and "RDA" in top.label


def test_second_generation_fragments():
    ps, _, _ = peaks("NCc1ccccc1", "M+")                    # トロピリウム 91 → C2H2 脱離 → 65(C5H5+)、フェニル 77 → 51
    p65 = has(ps, 65.0386)
    assert p65 and p65[0].gen == 2 and "C2H2" in p65[0].label
    assert has(ps, 51.0229)


def test_default_ion_is_m_plus():
    ps, _, prec = fr.predict(Chem.MolFromSmiles("CCO"), engine="rules")
    assert prec == pytest.approx(46.0413, abs=1e-4)           # 既定は M+(電子 1 個ぶん軽い)


# ---------------------------------------------------------------- MassBank から学習した予測(モデルが無いときは飛ばす)
import n2s_fragml as fml

need_model = pytest.mark.skipif(not fml.available("[M+H]+"), reason="学習済みモデル(data/fragml)が無い")


@need_model
def test_ml_predicts_paracetamol_ketene_loss_as_base_peak():
    ps, note, prec = peaks("CC(=O)Nc1ccc(O)cc1", "[M+H]+", engine="ml")     # 実測の塩基ピークは 110(−ケテン)
    assert prec == pytest.approx(152.0706, abs=1e-3)
    top = max((p for p in ps if p.kind != "precursor"), key=lambda p: p.rel)
    assert top.mz == pytest.approx(110.0600, abs=2e-3) and "MassBank" in note


@need_model
def test_ml_peaks_are_well_formed_and_auto_uses_model():
    ps, note, prec = peaks("CC(=O)Oc1ccccc1C(=O)O", "[M+H]+", engine="auto")
    assert "MassBank" in note                                   # 学習したモデルが使われた
    assert [p.mz for p in ps] == sorted(p.mz for p in ps)
    assert all(0 < p.rel <= 100.0 + 1e-9 for p in ps) and any(p.kind == "precursor" for p in ps)
    assert all(p.mz < prec + 0.01 for p in ps)


def test_trained_on_flag_is_stable():
    m = Chem.MolFromSmiles("CCO")
    assert fml.trained_on(m) == fml.trained_on(Chem.MolFromSmiles("OCC"))      # 骨格(InChIKey 前半)で決まる


def test_engine_rules_still_available_and_unknown_ion_falls_back():
    ps, note, _ = peaks("CCO", "[M+Na]+", engine="auto")
    assert ps and "Na" in note


def _gnn_ready(ion):
    try:
        import n2s_fragml as fml
        return fml.available(ion) and fml.gnn_for(ion) is not None
    except Exception:
        return False


@pytest.mark.skipif(not _gnn_ready("[M+H]+"), reason="学習済みの GNN(data/fragnn)が無い")
def test_gnn_engine_matches_candidates_and_differs_from_gbm():
    import n2s_fragml as fml
    mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
    ps, note, prec = fr.predict(mol, "[M+H]+", engine="auto")
    assert fml.last_engine == "gnn" and "グラフニューラルネット" in note
    assert prec == pytest.approx(181.0495, abs=1e-3)
    assert [p.mz for p in ps] == sorted(p.mz for p in ps) and any(p.kind == "precursor" for p in ps)
    top = max((p for p in ps if p.kind != "precursor"), key=lambda p: p.rel)
    assert top.rel == pytest.approx(100.0) and top.atoms
    gb, note2, _ = fr.predict(mol, "[M+H]+", engine="gbm")
    assert fml.last_engine == "gbm" and "グラフニューラルネット" not in note2
    assert {round(p.mz, 3) for p in ps} != {round(p.mz, 3) for p in gb}


@pytest.mark.skipif(not _gnn_ready("M+"), reason="学習済みの GNN(data/fragnn)が無い")
def test_gnn_falls_back_to_gbm_when_unavailable(monkeypatch):
    import n2s_fragml as fml
    monkeypatch.setitem(fml._gnns, "EI", None)
    ps, note, _ = fr.predict(Chem.MolFromSmiles("NCc1ccccc1"), "M+", engine="auto")
    assert fml.last_engine == "gbm" and ps
