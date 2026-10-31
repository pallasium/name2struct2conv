"""n2s_ms(元素分析・マススペクトルの計算)のテスト。OPSIN もモデルも使わない。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pytest
from rdkit import Chem

import n2s_ms as ms


def mol(smi):
    return Chem.MolFromSmiles(smi)


def test_elemental_analysis_ethanol():
    mw, rows = ms.elemental_analysis(mol("CCO"))
    d = {s: (n, p) for s, n, p in rows}
    assert [s for s, _, _ in rows] == ["C", "H", "O"]                    # Hill 順
    assert d["C"][0] == 2 and d["H"][0] == 6 and d["O"][0] == 1
    assert d["C"][1] == pytest.approx(52.14, abs=0.01)
    assert d["H"][1] == pytest.approx(13.13, abs=0.01)
    assert d["O"][1] == pytest.approx(34.73, abs=0.01)
    assert sum(p for _, _, p in rows) == pytest.approx(100.0, abs=1e-6)
    assert mw == pytest.approx(46.069, abs=0.005)


def test_hill_order_without_carbon():
    _, rows = ms.elemental_analysis(mol("O"))     # 水
    assert [s for s, _, _ in rows] == ["H", "O"]


def test_ions_ethanol():
    ion = ms.ions(mol("CCO"))
    assert ion["M"] == pytest.approx(46.04186, abs=1e-4)
    assert ion["[M+H]+"] == pytest.approx(47.04914, abs=1e-4)
    assert ion["[M+Na]+"] == pytest.approx(69.03111, abs=1e-4)
    assert ion["[M-H]-"] == pytest.approx(45.03458, abs=1e-4)


def test_chlorine_isotope_pattern():
    pts = ms.spectrum(mol("Clc1ccccc1"), "M")      # クロロベンゼン: M と M+2 が約 3:1(中性の M)
    top = max(pts, key=lambda p: p[1])
    assert top[0] == pytest.approx(112.0080, abs=1e-3) and top[1] == pytest.approx(100.0)
    m2 = [rel for mz, rel in pts if abs(mz - 114.005) < 0.01][0]
    assert m2 == pytest.approx(32.0, abs=1.0)


def test_bromine_pattern_is_one_to_one():
    pts = ms.spectrum(mol("CBr"), "M")
    rel = {round(mz): r for mz, r in pts}
    assert rel[94] == pytest.approx(100.0, abs=0.5) and rel[96] == pytest.approx(97.3, abs=1.0)


def test_pattern_sums_to_one_after_normalisation():
    pts = ms.neutral_pattern(mol("CC(=O)Oc1ccccc1C(=O)O"))   # アスピリン C9H8O4
    assert pts[0][0] == pytest.approx(180.0423, abs=1e-3)
    assert max(r for _, r in pts) == pytest.approx(100.0)


def test_charged_and_salt():
    cat = ms.ions(mol("C[N+](C)(C)C"))             # テトラメチルアンモニウム
    assert list(cat) == ["M+"] and cat["M+"] == pytest.approx(74.0964, abs=1e-3)
    salt = mol("[Na+].[Cl-]")                        # 塩: 元素分析は全成分、MS は測るイオンに合う成分
    _, rows = ms.elemental_analysis(salt)
    d = {s: p for s, _, p in rows}
    assert d["Na"] == pytest.approx(39.34, abs=0.01) and d["Cl"] == pytest.approx(60.66, abs=0.01)
    assert ms.spectrum(salt, "[M+H]+")[0][0] == pytest.approx(22.98922, abs=1e-4)      # Na+
    assert ms.spectrum(salt, "[M-H]-")[0][0] == pytest.approx(34.96940, abs=1e-4)      # Cl-(電子 1 個ぶん重い)
    tma = mol("C[N+](C)(C)C.[Br-]")
    assert ms.spectrum(tma, "[M+H]+")[0][0] == pytest.approx(74.0964, abs=1e-3)


def test_unknown_element_does_not_crash():
    m = mol("[Se]=C=O")
    assert ms.approx(m) and ms.spectrum(m, "M")


def test_text_helpers():
    assert ms.ea_text(mol("CCO")) == "C 52.14  H 13.13  O 34.73 (%)"
    assert ms.ms_text(mol("CCO")).startswith("M+: 46.0413(100)")                # 既定は M+
    assert ms.ms_text(mol("CCO"), "[M+H]+").startswith("[M+H]+: 47.0491(100)")


def test_molecular_ion_label_m_plus():
    pts = ms.spectrum(mol("CCO"), "M+")                     # ポジティブモードの分子イオン M+(電子 1 個ぶん軽い)
    assert pts[0][0] == pytest.approx(46.04186 - ms.ELECTRON, abs=1e-5)
    assert ms.ms_text(mol("CCO"), "M+").startswith("M+: 46.0413")
