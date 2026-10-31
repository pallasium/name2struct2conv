"""監査で直した不具合の再発防止テスト(モデル・ネットワーク不要)。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pytest
from rdkit import Chem

import n2s_ms as ms
import n2s_msdb as db
import n2s_props as pr

ROOT = Path(__file__).resolve().parents[1]
HAS_OPSIN = any((ROOT / "lib").glob("opsin-cli-*.jar")) and (ROOT / "lib" / "OpsinServer.class").exists()


def test_props_hypervalent_sulfur_is_not_silently_skipped():
    """3 本以上の結合をもつ S(Joback に無い基)を数えずに飛ばさず、推算しない(N/A)。"""
    assert pr.joback_groups(Chem.MolFromSmiles("FS(F)(F)(F)C")) is None
    rows = {k: v for k, _, v, _, _ in pr.properties(Chem.MolFromSmiles("FS(F)(F)(F)C"))}
    assert rows["Tb"] == "N/A"


def test_props_and_ea_on_empty_molecule():
    empty = Chem.Mol()
    assert all(v == "N/A" for _, _, v, _, _ in pr.properties(empty))
    assert ms.elemental_analysis(empty) == (0.0, [])
    assert ms.ea_text(empty) == ""


def test_msdb_build_failure_leaves_no_partial_index(tmp_path):
    """読めない msp で失敗しても、作りかけの .tmp や壊れた索引を残さない。"""
    out = tmp_path / "x.sqlite"
    with pytest.raises(FileNotFoundError):
        db.build(tmp_path / "missing.msp", out)
    assert not out.exists() and not out.with_suffix(".tmp").exists()


def test_xlsx_output_does_not_turn_input_into_formula(tmp_path):
    pytest.importorskip("openpyxl")
    from openpyxl import load_workbook

    import n2s_core as core
    import name2struct
    r = core.Result("=HYPERLINK(\"http://example.com\")", "名前→構造", level="失敗", message="=1+1")
    dst = tmp_path / "out.xlsx"
    name2struct.write_results([r], dst)
    ws = load_workbook(dst).active
    cells = [c for c in ws[2] if isinstance(c.value, str) and c.value.startswith("=")]
    assert cells and all(c.data_type == "s" for c in cells)


def test_spell_fix_works_without_translation_models(monkeypatch):
    """綴りの修正は機械学習を使わない。tokenizer が読めなくても例外にしない。"""
    import n2s_core as core
    monkeypatch.setattr(core, "_iupac_tok", False)
    assert core._rarity("methyl") == 0.0
    if not HAS_OPSIN:
        pytest.skip("OPSIN が無い")
    r = core.name_to_struct("2-metylpropan-1-ol", ml_fallback=False)
    assert r.level == "推定" and Chem.MolToSmiles(r.mol) == Chem.MolToSmiles(Chem.MolFromSmiles("CC(C)CO"))


@pytest.mark.skipif(not HAS_OPSIN, reason="OPSIN が無い")
def test_opsin_restarts_cleanly_after_interrupted_read(monkeypatch):
    """読み取りの途中で例外が出ても、次の問い合わせの答えがずれない(サーバーを起動し直す)。"""
    import n2s_opsin as opsin
    srv = opsin.server()

    class Flaky:   # 2 行目を読むところで中断させる
        def __init__(self, f):
            self.f, self.n = f, 0

        def readline(self):
            self.n += 1
            if self.n == 2:
                raise KeyboardInterrupt
            return self.f.readline()
    monkeypatch.setattr(srv.proc, "stdout", Flaky(srv.proc.stdout))
    with pytest.raises(KeyboardInterrupt):
        opsin.parse(["ethanol", "methanol", "propan-1-ol"])
    r = opsin.parse(["benzene"])[0]
    assert r.ok and Chem.MolToSmiles(r.mol) == "c1ccccc1"
    assert opsin.server() is not srv
