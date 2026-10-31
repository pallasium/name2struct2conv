"""エステル・リンの酸のエステルを、部分に分けて名付けてから組み立てる(塩の名前の作り方と同じ考え方)。

  カルボン酸エステル  R'C(=O)O-R  ->  「R の yl」+ 空白 + 「R'COOH の ate」        例: ethyl acetate
  リンのエステル      P(=X)(YR)(YR')(YR'')  ->  「O,O-diethyl S-propyl phosphorothioate」の形

部分の名前は name_fn(Mol) -> [名前...](OPSIN で戻ると確かめ済みの候補)で付ける。
組み立てた名前が元の構造に戻るかは呼び出し側が OPSIN で確かめる。
"""
from __future__ import annotations

import re
from itertools import product

from rdkit import Chem

import n2s_pin as pin

_ESTER = Chem.MolFromSmarts("[#6][CX3](=O)[OX2;H0][#6;!$(C=O)]")
_PHOS = Chem.MolFromSmarts("[PX4](=[O,S])")


def _frags(mol: Chem.Mol, break_bond: tuple[int, int], cap: tuple[int, int]):
    """break_bond の結合を切り、(atom_a 側に H を付ける, atom_b 側に OH を付ける)。cap = (H を付ける側, OH を付ける側)。"""
    rw = Chem.RWMol(mol)
    a, b = break_bond
    rw.RemoveBond(a, b)
    rw.AddBond(cap[1], rw.AddAtom(Chem.Atom(8)), Chem.BondType.SINGLE)
    for at in rw.GetAtoms():
        at.SetNoImplicit(False)
        at.SetNumExplicitHs(0)
    m = rw.GetMol()
    Chem.SanitizeMol(m)
    return Chem.GetMolFrags(m, asMols=True)


_CHAIN = re.compile(r"(?<![a-z(])((?:tert-)?[a-z]*?(?:sulfanyl|sulfinyl|sulfonyl|oxy|amino))(methyl|ethyl|propyl|butyl|phenyl)\b")


def _paren_chain(yl: str) -> str:
    """'ethylsulfanylmethyl' -> '(ethylsulfanyl)methyl'、'ethylsulfinyl' は 'ethanesulfinyl' に。"""
    yl = _CHAIN.sub(lambda m: f"({pin.rename_sub(m.group(1))}){m.group(2)}", yl)
    return re.sub(r"(?<![a-z])(methyl|ethyl)(sulfinyl|sulfonyl)", lambda m: {"methyl": "methane", "ethyl": "ethane"}[m.group(1)] + m.group(2), yl)


def alcohol_to_yl(name: str) -> str | None:
    """'2-(diethylamino)ethan-1-ol' -> '2-(diethylamino)ethyl' など。"""
    y = _alcohol_to_yl(name)
    return _paren_chain(re.sub(r"-(phenyl)$", r"\1", y)) if y else None


def _alcohol_to_yl(name: str) -> str | None:
    rules = [
        (r"^(.*?)phenol$", r"\1phenyl"), (r"^(.*)an-1-ol$", r"\1yl"), (r"^(.*)anol$", r"\1yl"),
        (r"^(.*)an-(\d+)-ol$", r"\1an-\2-yl"), (r"^(.*)-(\d+)-ol$", r"\1-\2-yl"), (r"^(.*)ol$", r"\1yl"),
    ]
    if re.search(r"(?:di|tri)ol\b|\bol\b.*\bol\b|-ol\b.*-ol\b|\d,\d+-diol", name) or " " in name.strip():
        return None
    for pat, rep in rules:
        if re.match(pat, name):
            return re.sub(pat, rep, name)
    return None


def acid_to_ate(name: str) -> str | None:
    n = re.sub(r"ic acid$", "ate", name)
    if n == name:
        return None
    n = re.sub(r"(?<!m)ethanoate$", "acetate", n)
    if n.endswith("acetate") and re.search(r"(?<![\d,])2(?:,2)*-", n) and not re.search(r"(?<![\d,])(?:[013-9]\d*|2\d)", n.replace("acetate", "")):
        n = re.sub(r"(?<![\d,])2(?:,2)*-", "", n)   # 酢酸の置換は位置番号を書かない(bromoacetate)
    return n


_ACID_OH = Chem.MolFromSmarts("C(=O)[OH]")


def ester_names(mol: Chem.Mol, name_fn) -> list[str]:
    ms = mol.GetSubstructMatches(_ESTER)
    if len(ms) != 1 or Chem.GetFormalCharge(mol) != 0 or len(Chem.GetMolFrags(mol)) != 1:
        return []
    c_r, c_acyl, o_dbl, o_est, c_alk = ms[0]
    try:
        frags = _frags(mol, (o_est, c_alk), (o_est, c_alk))
    except Exception:
        return []
    if len(frags) != 2:
        return []
    acid = next((f for f in frags if f.HasSubstructMatch(_ACID_OH)), None)
    alc = next((f for f in frags if f is not acid), None)
    if acid is None or alc is None:
        return []
    ates = [a for a in (acid_to_ate(n) for n in name_fn(acid)[:3]) if a]
    yls = [y for y in (alcohol_to_yl(n) for n in name_fn(alc)[:3]) if y]
    return [f"{y} {a}" for y, a in product(yls, ates)]


# ---------------------------------------------------------------- アミド・尿素

_AMIDE = Chem.MolFromSmarts("[#6][CX3](=O)[NX3]")
_UREA = Chem.MolFromSmarts("[NX3][CX3](=O)[NX3]")


def acid_to_amide(name: str) -> str | None:
    for pat, rep in ((r"carboxylic acid$", "carboxamide"), (r"oic acid$", "amide"), (r"ic acid$", "amide")):
        if re.search(pat, name):
            return re.sub(pat, rep, name)
    return None


def _cap(mol: Chem.Mol, cut: tuple[int, int], oh_on: int) -> list[Chem.Mol]:
    """cut の結合を切り、oh_on の原子に OH を付けて、できた断片を返す。"""
    rw = Chem.RWMol(mol)
    rw.RemoveBond(*cut)
    o = rw.AddAtom(Chem.Atom(8))
    rw.AddBond(oh_on, o, Chem.BondType.SINGLE)
    for at in rw.GetAtoms():
        at.SetNoImplicit(False)
        at.SetNumExplicitHs(0)
    m = rw.GetMol()
    Chem.SanitizeMol(m)
    return list(Chem.GetMolFrags(m, asMols=True)), [list(x) for x in Chem.GetMolFrags(m)]


def _n_substituents(mol, n_idx, exclude):
    """窒素に付いている炭素の置換基(アシル炭素 exclude 以外)。"""
    return [a.GetIdx() for a in mol.GetAtomWithIdx(n_idx).GetNeighbors()
            if a.GetIdx() not in exclude and a.GetSymbol() == "C"]


def _yls(mol, n_idx, r_idx, name_fn):
    frags, idxs = _cap(mol, (n_idx, r_idx), r_idx)
    frag = next(f for f, ix in zip(frags, idxs) if r_idx in ix)
    return [y for y in (alcohol_to_yl(n) for n in name_fn(frag)[:3]) if y]


def _wrap_yl(y: str) -> str:
    return y if pin._is_simple(y) else f"({y})"


def amide_names(mol: Chem.Mol, name_fn) -> list[str]:
    ms = mol.GetSubstructMatches(_AMIDE)
    if len(ms) != 1 or Chem.GetFormalCharge(mol) != 0 or len(Chem.GetMolFrags(mol)) != 1:
        return []
    c_r, c_acyl, o, n = ms[0]
    if mol.GetAtomWithIdx(n).GetIsAromatic() or mol.GetAtomWithIdx(c_acyl).GetIsAromatic():
        return []
    subs = _n_substituents(mol, n, {c_acyl})
    if not subs or len(subs) > 2:
        return []
    try:
        frags, idxs = _cap(mol, (c_acyl, n), c_acyl)
        acid = next(f for f, ix in zip(frags, idxs) if c_acyl in ix)
        amides = [x for x in (acid_to_amide(nm) for nm in name_fn(acid)[:3]) if x]
        yl_lists = [_yls(mol, n, r, name_fn) for r in subs]
    except Exception:
        return []
    if not amides or any(not y for y in yl_lists):
        return []
    outs = []
    for am, *ys in product(amides, *[y[:2] for y in yl_lists]):
        if len(ys) == 1:
            outs.append(f"N-{_wrap_yl(ys[0])}" + ("" if am[:1].isalpha() else "-") + am)
        elif ys[0] == ys[1]:
            outs.append((f"N,N-di{_wrap_yl(ys[0])}" if pin._is_simple(ys[0]) else f"N,N-bis{_wrap_yl(ys[0])}") + ("" if am[:1].isalpha() else "-") + am)
        else:
            outs.append(f"N-{_wrap_yl(ys[0])}-N-{_wrap_yl(ys[1])}" + ("" if am[:1].isalpha() else "-") + am)
    return outs


def urea_names(mol: Chem.Mol, name_fn) -> list[str]:
    ms = mol.GetSubstructMatches(_UREA)
    if len(ms) != 1 or Chem.GetFormalCharge(mol) != 0 or len(Chem.GetMolFrags(mol)) != 1:
        return []
    n1, c, _o, n2 = ms[0]
    pieces = []   # (N の記号, 置換基の yl 候補)
    try:
        for tag, n in (("N", n1), ("N'", n2)):
            for r in _n_substituents(mol, n, {c}):
                pieces.append((tag, _yls(mol, n, r, name_fn)))
    except Exception:
        return []
    if not pieces or any(not y for _, y in pieces):
        return []
    outs = []
    for combo in product(*[y[:2] for _, y in pieces]):
        outs.append("-".join(f"{tag}-{_wrap_yl(yl)}" for (tag, _), yl in zip(pieces, combo)) + "urea")
    return outs


# ---------------------------------------------------------------- リン

_MULT = pin._MULT


def phosphorus_names(mol: Chem.Mol, name_fn) -> list[str]:
    ps = mol.GetSubstructMatches(_PHOS)
    if len(ps) != 1 or Chem.GetFormalCharge(mol) != 0 or len(Chem.GetMolFrags(mol)) != 1:
        return []
    p_idx, x_idx = ps[0]
    p = mol.GetAtomWithIdx(p_idx)
    x_sym = mol.GetAtomWithIdx(x_idx).GetSymbol()
    ys = [n for n in p.GetNeighbors() if n.GetIdx() != x_idx]
    if len(ys) != 3:
        return []
    groups = []   # (locant 'O' / 'S' / '' (直接結合の C), 結合を切る原子, R 側の原子)
    for y in ys:
        sym = y.GetSymbol()
        others = [n for n in y.GetNeighbors() if n.GetIdx() != p_idx]
        if sym in ("O", "S") and len(others) == 1 and others[0].GetSymbol() == "C":
            groups.append((sym, y.GetIdx(), others[0].GetIdx()))
        elif sym == "C":
            groups.append(("", p_idx, y.GetIdx()))
        else:
            return []
    direct = [g for g in groups if g[0] == ""]
    if len(direct) > 1:
        return []
    s = (x_sym == "S") + sum(g[0] == "S" for g in groups)
    thio = {0: "", 1: "othio", 2: "odithio", 3: "otrithio", 4: "otetrathio"}[s]
    # 各 R を R-OH にして名付ける
    names_per = []
    cache: dict[str, list[str]] = {}
    for loc, a, r in groups:
        try:
            frag = _r_fragment(mol, loc, a, r, p_idx)
        except Exception:
            return []
        can = Chem.MolToSmiles(frag)
        if can not in cache:
            cache[can] = [y for y in (alcohol_to_yl(n) for n in name_fn(frag)[:3]) if y]
        if not cache[can]:
            return []
        names_per.append(cache[can])
    outs = []
    for combo in product(*[n[:2] for n in names_per]):
        items = list(zip([g[0] for g in groups], combo))
        outs.append(_compose(items, thio))
    return [o for o in dict.fromkeys(outs) if o]


def _r_fragment(mol, loc, a, r, p_idx):
    """P-Y-R の R 側を R-OH(直接結合の C は P の代わりに OH)にして返す。"""
    rw = Chem.RWMol(mol)
    if loc == "":
        rw.RemoveBond(p_idx, r)
        o = rw.AddAtom(Chem.Atom(8))
        rw.AddBond(r, o, Chem.BondType.SINGLE)
    else:
        rw.RemoveBond(p_idx, a)   # Y(O/S) と R はつないだまま。Y が O なら R-OH、S なら R-SH だが O に置き換える
        rw.GetAtomWithIdx(a).SetAtomicNum(8)
    for at in rw.GetAtoms():
        at.SetNoImplicit(False)
        at.SetNumExplicitHs(0)
    m = rw.GetMol()
    Chem.SanitizeMol(m)
    frags = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=True)
    idxs = Chem.GetMolFrags(m)
    for f, ix in zip(frags, idxs):
        if r in ix:
            return f
    raise ValueError


def _compose(items, thio) -> str:
    direct = [y for loc, y in items if loc == ""]
    esters = [(loc, y) for loc, y in items if loc]
    words = []
    for y in sorted(set(y for _, y in esters), key=lambda y: pin.alpha_key(y if pin._is_simple(y) else f"({y})")):
        locs = sorted(loc for loc, yy in esters if yy == y)
        n = len(locs)
        simple = pin._is_simple(y)
        located = bool(re.fullmatch(r"[a-z]+-\d+-yl", y))   # propan-2-yl: 括弧に入れ、倍数は di(...)
        pre = (pin._MULT[n] if (simple or located) else pin._MULT_C[n]) if n > 1 else ""
        wrap = (not simple) and bool(thio or located or n > 1)   # 位置記号つきの語・位置番号を持つ置換基の倍数は括弧に入れる
        body = f"({y})" if wrap else y
        words.append((f"{','.join(locs)}-" if thio else "") + f"{pre}{body}")
    core = ("phosphon" if direct else "phosphor" if thio else "phosph") + (thio + "ate" if thio else "ate")
    if direct:
        core = (direct[0] if pin._is_simple(direct[0]) else f"({direct[0]})") + core
    return pin.tidy_nesting(" ".join(words + [core]))
