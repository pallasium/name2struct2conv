"""指示水素・水素化(hydro 接頭語)・付加水素を、分子の構造から決めて名前に付ける(PIN の書き方)。

OPSIN は hydro の組み合わせが多少おかしくても同じ構造に読んでしまうので、名前だけでは選べない。
そこで環の母体名(purine など)を OPSIN で読んで分子に重ね、環の各位置が飽和しているか(環内二重結合が無いか)を調べる。

  S = 環内二重結合を持たない環原子(CH2・NR・C=O の炭素など)
  O = そのうち =O / =S を持つもの(接尾語 -one / -thione の位置)
  N = S - O(hydro 接頭語・指示水素で表す位置)
母体が指示水素を要する(mancude でも sp3 が 1 つ要る: purine, indole, pyran…)かどうかは、母体名だけを読んで調べる。
呼び出し側が OPSIN で(互変異性体まで)同じ構造に戻るか確かめる。
"""
from __future__ import annotations

import re
from functools import lru_cache

from rdkit import Chem

import n2s_opsin as opsin
import n2s_pin as pin

_HYDRO = {2: "di", 4: "tetra", 6: "hexa", 8: "octa", 10: "deca"}


def _loc_key(loc: str):
    m = re.match(r"(\d+)([a-z]?)", loc)
    return (int(m.group(1)), m.group(2)) if m else (999, loc)


@lru_cache(maxsize=512)
def _ring(token: str):
    """母体名 -> (骨格だけの検索用 Mol, {原子 index: 位置番号}, 指示水素が要るか)。読めなければ None。"""
    try:
        r = opsin.parse([token], lenient=True, cml=True)[0]
        m = opsin.cml_to_mol(r.cml) if r.ok else None
    except Exception:
        return None
    if m is None or not m.GetRingInfo().NumRings():
        return None
    k = Chem.Mol(m)
    try:
        Chem.Kekulize(k, clearAromaticFlags=True)
    except Exception:
        return None
    ring_atoms = {a for ring in m.GetRingInfo().AtomRings() for a in ring}
    locs = {}
    for i in ring_atoms:
        loc = opsin.numeric_locant(m.GetAtomWithIdx(i))
        if not loc:
            return None
        locs[i] = loc
    dbl = {i: False for i in ring_atoms}
    nh = {i: m.GetAtomWithIdx(i).GetTotalNumHs() for i in ring_atoms}
    for b in k.GetBonds():
        a, c = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if b.GetBondType() == Chem.BondType.DOUBLE and a in ring_atoms and c in ring_atoms:
            dbl[a] = dbl[c] = True
    needs_ind = any(not dbl[i] and nh[i] > 0 for i in ring_atoms)
    fixed = {locs[i] for i in ring_atoms if not dbl[i] and nh[i] == 0}   # 橋頭の N など、H を持たない飽和位置
    q = Chem.RWMol()
    old2new = {}
    for i in sorted(ring_atoms):
        old2new[i] = q.AddAtom(Chem.Atom(m.GetAtomWithIdx(i).GetAtomicNum()))
    for b in m.GetBonds():
        a, c = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if a in ring_atoms and c in ring_atoms:
            q.AddBond(old2new[a], old2new[c], Chem.BondType.SINGLE)
    return q.GetMol(), {old2new[i]: l for i, l in locs.items()}, needs_ind, fixed


def _skeleton(mol: Chem.Mol) -> Chem.Mol:
    s = Chem.RWMol(mol)
    for b in s.GetBonds():
        b.SetBondType(Chem.BondType.SINGLE)
        b.SetIsAromatic(False)
    for a in s.GetAtoms():
        a.SetIsAromatic(False)
        a.SetFormalCharge(0)
        a.SetNoImplicit(False)
    return s.GetMol()


def _positions(mol: Chem.Mol, token: str):
    """母体の環を分子に重ねたときの [(S の位置番号, O の位置番号, 指示水素が要るか)]。"""
    ring = _ring(token)
    if ring is None:
        return []
    q, locs, needs_ind, fixed = ring
    k = Chem.Mol(mol)
    try:
        Chem.Kekulize(k, clearAromaticFlags=True)
    except Exception:
        return []
    skel = _skeleton(mol)
    out = []
    seen = set()
    for match in skel.GetSubstructMatches(q, uniquify=False, maxMatches=200):
        at = {qi: mi for qi, mi in enumerate(match)}
        atoms = set(match)
        dbl, oxo = set(), set()
        for b in k.GetBonds():
            if b.GetBondType() != Chem.BondType.DOUBLE:
                continue
            a, c = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if a in atoms and c in atoms:
                dbl.update((a, c))
            elif a in atoms and k.GetAtomWithIdx(c).GetSymbol() in ("O", "S"):
                oxo.add(a)
            elif c in atoms and k.GetAtomWithIdx(a).GetSymbol() in ("O", "S"):
                oxo.add(c)
        # 二価の環の O・S は飽和位置に数えない
        S = {locs[qi] for qi, mi in at.items() if mi not in dbl and locs[qi] not in fixed
             and not (k.GetAtomWithIdx(mi).GetSymbol() in ("O", "S", "Se", "Te") and k.GetAtomWithIdx(mi).GetDegree() == 2)}
        O = {locs[qi] for qi, mi in at.items() if mi in oxo and mi not in dbl}
        key = (tuple(sorted(S, key=_loc_key)), tuple(sorted(O, key=_loc_key)))
        if key not in seen:
            seen.add(key)
            out.append((S, O, needs_ind))
    return out


def prefixes(S: set, O: set, needs_ind: bool):
    """-> (hydro 位置のリスト, 指示水素の位置 or None, 付加水素の位置 or None)。決められなければ None。
    母体が指示水素を要するなら、飽和位置のうち一番小さい位置(=O の位置でもよい)に指示水素を付ける(1H-indene-1,3(2H)-dione)。
    残った(=O でない)飽和位置が偶数なら hydro 接頭語、奇数なら 1 つを付加水素にして残りを hydro 接頭語にする(=O が要る)。"""
    N = sorted(S - O, key=_loc_key)
    ind = min(S, key=_loc_key) if needs_ind else None
    rem = [x for x in N if x != ind]
    added = None
    if len(rem) % 2 == 1:
        if not O:
            return None
        added, rem = rem[0], rem[1:]
    if rem and len(rem) not in _HYDRO:
        return None
    return rem, ind, added


_NONE = object()


def hydro_names(name: str, mol: Chem.Mol, parent_ok=None, _nested: bool = True) -> list[str]:
    """名前の環の母体に hydro 接頭語・指示水素・付加水素を付けた候補(複数の重ね方があれば複数)。
    括弧の中の置換基(…(2-oxopyridin-1-yl)…)の環にも、同じ規則を当てる。"""
    words = re.sub(r" (acid|oxide)\b", r"~\1", name).split(" ")
    outs = []
    for wi, w in enumerate(words):
        w = w.replace("~", " ")
        sp = pin.split(w, parent_ok)
        if not sp:
            continue
        stereo, segs, parent = sp
        had = any(re.search(r"hydro$", b) for _, b in segs) or bool(re.match(r"\d*H-", parent))
        segs = [(l, b) for l, b in segs if not re.search(r"hydro$", b)]   # 既にあるものは構造から作り直す
        parent = re.sub(r"^\d*H-", "", parent)
        parent = re.sub(r"\(\d+H\)", "", parent)
        i0 = re.search(r"[a-z]", parent).start() if re.search(r"[a-z]", parent) else 0   # 1,3-benzodioxole の頭の位置番号は環名の一部
        head = parent[:i0] + re.split(r"-(?=[\d(\[])", parent[i0:], maxsplit=1)[0]
        pos = []
        for cand in dict.fromkeys([head, head + "e"] if not head.endswith("e") else [head]):   # pyridin -> pyridine
            for n in range(len(cand), 3, -1):   # 長い方から: indole -> indol -> ... で最初に環として読めたもの
                pos = _positions(mol, cand[:n])
                if pos:
                    break
            if pos:
                break
        if not pos:
            continue
        pos = sorted(pos, key=lambda t: [_loc_key(x) for x in sorted(t[0], key=_loc_key)])   # 位置番号の小さい重ね方を先に(1H-pyrazole)
        body = pin.join(stereo, segs, "")
        sep = "-" if body and not body.endswith(("-", "(")) else ""
        my = re.match(r"^(.*?)-(\d+[a-z]?)-yl$", parent)
        has_oxo_prefix = any(b in ("oxo", "thioxo") for _, b in segs)
        for S, O, needs_ind in pos:
            if not S:
                continue
            if my and has_oxo_prefix and O:   # 2-oxopyridin-1(2H)-yl: 付加水素は =O の位置、結合する位置 A は数えない
                A = my.group(2)
                if A in S and not (S - {A} - O):
                    words2 = list(words)
                    rest_yl = f"{my.group(1)}-{A}({','.join(o + 'H' for o in sorted(O, key=_loc_key))})-yl"
                    words2[wi] = body + ("-" if body and not body.endswith(("-", "(")) and rest_yl[:1].isdigit() else "") + rest_yl
                    outs.append(" ".join(x.replace("~", " ") for x in words2))
                    continue
            p = prefixes(S, O, needs_ind)
            if p is None:
                continue
            hydro, ind, added = p
            if not hydro and ind is None and added is None:
                continue
            pre = (f"{','.join(hydro)}-{_HYDRO[len(hydro)]}hydro-" if hydro else "") + (f"{ind}H-" if ind else "")
            if pre.endswith("hydro-") and not parent[:1].isdigit():   # 5,6,7,8-tetrahydronaphthalene のようにつなぐ(環名が 1,3-oxazole のように数字で始まるときはハイフンを残す)
                pre = pre[:-1]
            rest = parent
            if added:
                rest = re.sub(r"(-\d+[a-z]?(?:,\d+[a-z]?)*)(-(?:di|tri)?(?:one|thione))", rf"\1({added}H)\2", parent, count=1)
                if rest == parent:
                    continue
            words2 = list(words)
            words2[wi] = body + sep + pre + rest
            outs.append(" ".join(x.replace("~", " ") for x in words2))
    if _nested:   # 括弧の中の置換基の環(それぞれ単独の断片として扱い、できた名前を元の位置に戻す)
        stack, pairs = [], []
        for i, ch in enumerate(name):
            if ch in "([{":
                stack.append(i)
            elif ch in ")]}" and stack:
                pairs.append((stack.pop(), i))
        for a, b in sorted(pairs, key=lambda x: x[1] - x[0])[:8]:
            inner = name[a + 1:b]
            if len(inner) < 5 or not re.search(r"[a-z]{3}", inner):
                continue
            for v in hydro_names(inner, mol, parent_ok, _nested=False):
                if v != inner:
                    outs.append(name[:a + 1] + v + name[b:])
    return list(dict.fromkeys(pin.tidy_nesting(o) for o in outs))
