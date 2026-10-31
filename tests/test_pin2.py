"""n2s_pin2(構造化版の PIN 書き換え)の単体テスト。OPSIN もモデルも使わない(母体は固定の一覧で判定)。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pytest

import n2s_pin2 as p2

PARENTS = {"phenol", "benzoic acid", "propanamide", "ethan-1-amine", "ethanamine", "butane", "phenyl", "benzene",
           "methane", "pyrimidin-5-yl", "methyl", "ethyl", "ethanol", "urea", "butane-1,4-diamine", "nonanamide",
           "benzamide", "propan-2-yl", "ethanone", "pyridine"}


def po(tail):
    return tail in PARENTS


@pytest.mark.parametrize("src, want", [
    # 英数字順(置換基の頭文字)
    ("3,5-dimethoxy-4-hydroxybenzoic acid", "4-hydroxy-3,5-dimethoxybenzoic acid"),
    # 複合置換基は括弧、単純なものは括弧なし
    ("2-methylphenol", "2-methylphenol"),
    # bis(chloro) -> dichloro、複合の倍数は bis
    ("N-[3,4-bis(chloro)phenyl]propanamide", "N-(3,4-dichlorophenyl)propanamide"),
    # 入れ子の括弧は内側から ( [ {
    ("N-(2-(diethylamino)ethyl)-2-(4-methoxyphenoxy)propanamide", "N-[2-(diethylamino)ethyl]-2-(4-methoxyphenoxy)propanamide"),
])
def test_normalize(src, want):
    assert p2.normalize(src, po) == want


@pytest.mark.parametrize("name", [
    "4-hydroxy-3,5-dimethoxybenzoic acid", "N-[2-(diethylamino)ethyl]-2-(4-methoxyphenoxy)propanamide",
    "N-(3,4-dichlorophenyl)propanamide", "2-methylphenol",
])
def test_idempotent(name):
    once = p2.normalize(name, po)
    assert p2.normalize(once, po) == once


def test_unparsable_is_kept():
    assert p2.normalize("zzz", po) == "zzz"
    assert p2.normalize("N-(unclosed", po) == "N-(unclosed"


def test_urea_and_diamine_locants():
    assert p2.normalize("1-(3-chlorophenyl)-3-methylurea", po) == "N-(3-chlorophenyl)-N'-methylurea"
    assert p2.normalize("N-methyl-N'-methylbutane-1,4-diamine", po) == "N1,N4-dimethylbutane-1,4-diamine"


def test_group_inside_leaf():
    PARENTS.update({"acetic acid", "oxy"})
    assert p2.normalize("2-[2,4,5-tris(chloro)phenyl]oxyacetic acid", po) == "2-[(2,4,5-trichlorophenyl)oxy]acetic acid"


def test_biphenyl_rename_inside_nested_group():
    PARENTS.update({"carbamate"})
    assert p2.normalize("N-(3-phenylphenyl)carbamate", po) == "N-([1,1'-biphenyl]-3-yl)carbamate"


def test_rule_ids_and_toggle():
    assert len(set(p2.RULES)) == len(p2.RULES)
    assert all(p2.on(i) for i in p2.RULES)                       # 既定はすべて有効
    src = "3,5-dimethoxy-4-hydroxybenzoic acid"
    assert p2.normalize(src, po) == "4-hydroxy-3,5-dimethoxybenzoic acid"
    with p2.disabled("R-ORDER"):
        assert p2.normalize(src, po) == "3,5-dimethoxy-4-hydroxybenzoic acid"
    assert p2.normalize(src, po) == "4-hydroxy-3,5-dimethoxybenzoic acid"   # 抜けたら元に戻る
    assert len(p2.global_rule_ids()) == len(p2.pin._GLOBAL)


def test_promote_suffix():
    PARENTS.update({"quinoline", "pyridine"})
    assert p2.normalize("8-hydroxy-5,7-diiodoquinoline", po) == "5,7-diiodoquinolin-8-ol"
    assert p2.normalize("3-amino-2-methylpyridine", po) == "2-methylpyridin-3-amine"
    # 優先する基があるときは何もしない
    assert p2.normalize("3-hydroxy-2-carboxypyridine", po) == "2-carboxy-3-hydroxypyridine"
    with p2.disabled("R-SUFFIX"):
        assert p2.normalize("8-hydroxy-5,7-diiodoquinoline", po) == "8-hydroxy-5,7-diiodoquinoline"


def test_promote_suffix_leaves_ketones_alone():
    PARENTS.update({"naphthalene-1,4-dione", "pentan-2-one", "quinoline"})
    assert p2.normalize("5-hydroxynaphthalene-1,4-dione", po) == "5-hydroxynaphthalene-1,4-dione"
    assert p2.normalize("4-hydroxy-4-methylpentan-2-one", po) == "4-hydroxy-4-methylpentan-2-one"
    assert p2.normalize("8-hydroxy-5,7-diiodoquinoline", po) == "5,7-diiodoquinolin-8-ol"


def test_hydro_prefix_not_alphabetized():
    PARENTS.update({"1H-inden-4-yl", "prop-2-enoic acid"})
    src = "(2E)-3-[(4S,7R,7aR)-3,7-dimethyl-2,4,5,6,7,7a-hexahydro-1H-inden-4-yl]-2-methylprop-2-enoic acid"
    assert p2.normalize(src, po) == src


def test_locant_vector_includes_indicated_hydrogen():
    assert p2.pin.locant_vector("1H-pyrazole") < p2.pin.locant_vector("2H-pyrazole")
    assert p2.pin.locant_vector("2-hydroxy-3-methyl-1H-indole") == (1, 2, 3)


def test_penalty_fixes():
    pen = p2.pin.pin_penalty
    assert pen("1,1,1-tribromo-1-fluoromethane") > pen("tribromo(fluoro)methane")          # メタンに位置番号
    assert pen("phenyldiazane") > pen("phenylhydrazine")
    PARENTS.update({"prop-1-ene"})
    assert pen("1,1,3,3,3-pentafluoro-2-(trifluoromethyl)prop-1-ene", po) == 0               # trifluoromethyl は括弧が要る
    PARENTS.update({"xanthene", "thioxanthene", "pyrimidin-4-amine"})
    assert pen("9H-thioxanthene", po) == 0                                                  # thio は環名の一部
    assert pen("2-chloro-N,N-dimethyl-6-methylpyrimidin-4-amine", po) > pen("2-chloro-N,N,6-trimethylpyrimidin-4-amine", po)
