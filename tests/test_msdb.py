"""n2s_msdb(MassBank の索引・照合)のテスト。小さな偽の msp を作って使う(ネットワーク不要)。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pytest
from rdkit import Chem

import n2s_msdb as db

MSP = """Name: Paracetamol
Synon: Acetaminophen
DB#: MSBNK-TEST-0001
InChIKey: RZVAJINKPMORJF-UHFFFAOYSA-N
SMILES: CC(=O)Nc1ccc(O)cc1
Precursor_type: [M+H]+
Spectrum_type: MS2
PrecursorMZ: 152.0706
Instrument_type: LC-ESI-QTOF
Ion_mode: POSITIVE
Collision_energy: 20
Formula: C8H9NO2
Num Peaks: 4
93.0335 120
110.0600 1000
111.0640 60
152.0706 200

Name: Paracetamol
DB#: MSBNK-TEST-0002
InChIKey: RZVAJINKPMORJF-UHFFFAOYSA-N
SMILES: CC(=O)Nc1ccc(O)cc1
Precursor_type: [M-H]-
Spectrum_type: MS2
PrecursorMZ: 150.0561
Instrument_type: LC-ESI-QTOF
Ion_mode: NEGATIVE
Formula: C8H9NO2
Num Peaks: 2
107.0502 400
108.0535 1000

Name: Other
DB#: MSBNK-TEST-0003
InChIKey: AAAAAAAAAAAAAA-UHFFFAOYSA-N
SMILES: CCO
Precursor_type: [M+H]+
Spectrum_type: MS1
PrecursorMZ: 47.05
Num Peaks: 1
47.0491 10
"""


@pytest.fixture()
def tmpdb(tmp_path, monkeypatch):
    msp = tmp_path / "t.msp"
    msp.write_text(MSP, encoding="utf-8")
    dbp = tmp_path / "t.sqlite"
    n = db.build(msp, dbp)
    monkeypatch.setattr(db, "DB", dbp)
    return n


def test_build_keeps_ms2_only(tmpdb):
    assert tmpdb == 2                                     # MS1 は除く
    assert db.stats() == {"spectra": 2, "compounds": 1}


def test_lookup_by_skeleton_and_ion_order(tmpdb):
    mol = Chem.MolFromSmiles("CC(=O)Nc1ccc(O)cc1")
    recs = db.lookup(mol, "[M-H]-")
    assert [r.precursor_type for r in recs][0] == "[M-H]-"        # 指定したイオンが先
    assert {r.acc for r in db.lookup(mol, "[M+H]+")} == {"MSBNK-TEST-0001", "MSBNK-TEST-0002"}
    assert db.lookup(Chem.MolFromSmiles("CCO"), "[M+H]+") == []
    assert recs[0].peaks[0][1] == pytest.approx(40.0) or recs[0].peaks[1][1] == pytest.approx(100.0)   # 最大 = 100%


def test_compare_explained_and_precision(tmpdb):
    rec = [r for r in db.lookup(Chem.MolFromSmiles("CC(=O)Nc1ccc(O)cc1"), "[M+H]+") if r.precursor_type == "[M+H]+"][0]
    assert rec.hires and db.tolerance(rec) == 0.01
    m = db.compare([(110.0600, 100.0), (93.0335, 40.0), (130.0, 30.0)], rec)
    assert m.explained == pytest.approx((100 + 12) / (100 + 12 + 6 + 20), abs=0.001)   # 実測の主要ピーク(5% 以上)は 110(100)・93(12)・111(6)・152(20) の 4 つ。予測と一致したのは 110 と 93
    assert m.pred_hit[0] == pytest.approx(100.0) and m.pred_hit[2] is None         # 130 は実測に無い
    assert m.precision == pytest.approx(2 / 3, abs=0.01)
    assert m.top5 == pytest.approx(2 / 4, abs=0.01)                                # 4 ピークのうち 2


def test_no_database(monkeypatch, tmp_path):
    monkeypatch.setattr(db, "DB", tmp_path / "none.sqlite")
    assert not db.available() and db.lookup(Chem.MolFromSmiles("CCO")) == [] and db.stats() == {}
