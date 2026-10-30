"""name2struct2 のテスト。実行: python -m pytest tests -q(初回はモデル読み込みで 30 秒ほど)"""
import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import n2s_core as core          # noqa: E402
import n2s_opsin as opsin        # noqa: E402
import n2s_stereo as stereo      # noqa: E402
from rdkit import Chem           # noqa: E402
from rdkit.Chem import rdMolDescriptors  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def key(smi):
    return Chem.MolToInchiKey(Chem.MolFromSmiles(smi))


# ---------------------------------------------------------------- 辞書

def test_dictionary_formulas():
    """names_dict.tsv の全項目が OPSIN で読め、書いた分子式と一致する。"""
    rows = [l.split("\t") for l in (ROOT / "names_dict.tsv").read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.startswith("#")]
    res = opsin.parse([r[1] for r in rows])
    bad = [(r[0], r[1]) for r, x in zip(rows, res)
           if x.mol is None or rdMolDescriptors.CalcMolFormula(x.mol) != r[2].strip()]
    assert not bad


# ---------------------------------------------------------------- 名前 -> 構造

@pytest.mark.parametrize("text, smiles, level", [
    ("2-acetyloxybenzoic acid", "CC(=O)Oc1ccccc1C(=O)O", "OPSIN"),
    ("(2S)-2-amino-3-phenylpropanoic acid", "N[C@@H](Cc1ccccc1)C(=O)O", "OPSIN"),
    ("DMF", "CN(C)C=O", "辞書"),
    ("安息香酸", "O=C(O)c1ccccc1", "辞書"),
    ("ｴﾀﾉｰﾙ", "CCO", "辞書"),                          # 半角カナも NFKC で同じ
    ("EDC·HCl", "CCN=C=NCCCN(C)C.Cl", "辞書"),
    ("benzoic", "O=C(O)c1ccccc1", "OPSIN"),           # acid の書き忘れ
    ("2-metylpropan-1-ol", "CC(C)CO", "推定"),         # 1 文字抜け
    ("hexaflurobenzene", "Fc1c(F)c(F)c(F)c(F)c1F", "推定"),
])
def test_name_to_struct(text, smiles, level):
    r = core.convert(text)
    assert r.direction == "名前→構造"
    assert r.level == level
    assert r.info["inchikey"] == key(smiles)


def test_name_unreadable():
    r = core.name_to_struct("qwertyzzz", ml_fallback=False)
    assert r.level == "失敗" and not r.ok


# ---------------------------------------------------------------- 構造 -> 名前

@pytest.mark.parametrize("smiles", [
    "CCO",
    "CC(=O)Oc1ccccc1C(=O)O",
    "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O",                 # 立体 5 個
    "C/C=C/C(=O)O",                                               # E/Z(モデルの語彙に / が無い)
    "CCCCNC(=O)c1ccccc1",
    "CC(C)(C)OC(=O)N[C@@H](Cc1ccccc1)C(=O)O",
    "[Na+].[Na+].[O-]C(=O)CCC([O-])=O",                            # 塩
    "NCCN.Cl.Cl",                                                 # 酸付加塩
])
def test_smiles_to_name_verified(smiles):
    r = core.smiles_to_name(smiles)
    assert r.level == "一致", (r.name, r.candidates[:3])
    back = opsin.parse([r.name])[0].mol
    assert Chem.MolToInchiKey(back) == key(smiles)


def test_smiles_to_name_style():
    assert core.smiles_to_name("O=C(O)c1ccccc1").name == "benzoic acid"
    assert core.smiles_to_name("CC(=O)Nc1ccc(O)cc1").name == "N-(4-hydroxyphenyl)acetamide"


def test_aliases_shown_for_structure():
    assert "DMSO" in core.smiles_to_name("CS(C)=O").aliases


def test_inchi_and_molblock_input():
    assert core.convert("InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3").name == "ethanol"
    block = Chem.MolToMolBlock(Chem.MolFromSmiles("CCO"))
    assert core.convert(block).name == "ethanol"


# ---------------------------------------------------------------- 立体の付け直し

@pytest.mark.parametrize("name, smiles, want", [
    ("4-(1-hydroxyethyl)cyclohexan-1-ol", "C[C@@H](O)C1CCC(O)CC1", "4-[(1R)-1-hydroxyethyl]cyclohexan-1-ol"),
    ("methyl 2-aminopropanoate", "C[C@H](N)C(=O)OC", "methyl (2S)-2-aminopropanoate"),
    ("octadeca-9,12-dienoic acid", "CCCCC/C=C\\C/C=C\\CCCCCCCC(=O)O", "(9Z,12Z)-octadeca-9,12-dienoic acid"),
    ("2-(1,2-dihydroxyethyl)-3,4-dihydroxy-2H-furan-5-one", "O=C1O[C@H]([C@@H](O)CO)C(O)=C1O",
     "(2R)-2-[(1S)-1,2-dihydroxyethyl]-3,4-dihydroxy-2H-furan-5-one"),
])
def test_stereo_assign(name, smiles, want):
    assert stereo.assign(name, Chem.MolFromSmiles(smiles)) == want


def test_strip_and_brackets():
    assert stereo.strip("(2S,3R)-2-amino-3-[(1R)-1-hydroxyethyl]butanoic acid") == \
        "2-amino-3-1-hydroxyethyl]butanoic acid".replace("3-1-", "3-[1-")
    assert stereo.tidy_brackets("4-[[2-(methyl)ethyl]amino]") == "4-{[2-(methyl)ethyl]amino}"
    assert stereo.tidy_brackets("cyclopenta[a]phenanthrene") == "cyclopenta[a]phenanthrene"


# ---------------------------------------------------------------- 一括変換

def test_batch(tmp_path):
    import name2struct
    src = tmp_path / "in.csv"
    src.write_text("input\nCCO\nDMF\nqwertyzzz\n", encoding="utf-8")
    for ext in (".csv", ".xlsx", ".sdf"):
        dst = tmp_path / f"out{ext}"
        name2struct.run_batch(src, dst)
        assert dst.exists() and dst.stat().st_size > 0
    rows = list(csv.DictReader(open(tmp_path / "out.csv", encoding="utf-8-sig")))
    assert [r["check"] for r in rows][:2] == ["一致", "辞書"]
