"""名前の立体記号(R/S・E/Z)を外す・付け直す。

付け直しの手順:
  1. 立体記号を外した名前を OPSIN で読み、CML から各原子の位置番号を得る
  2. 入力構造と原子を対応させ、RDKit の CIP 判定(新方式)で R/S・E/Z を決める
  3. 立体要素を 1 つずつ、名前の先頭・空白の後・置換基の括弧の直後に入れてみて、
     OPSIN で読み戻した構造が「その要素まで指定した入力構造」と一致する位置を採る
"""
from __future__ import annotations

import re
from itertools import product

from rdkit import Chem
from rdkit.Chem import rdCIPLabeler

import n2s_opsin as opsin

_DESC = r"(?:\d+[a-z]?'*|[A-Z]\d*'*)?(?:[RSEZrs]\*?|[RS]\*)"
_GROUP = re.compile(rf"\((?:{_DESC})(?:,\s*(?:{_DESC}))*\)-?")
_RACREL = re.compile(r"\b(?:rac|rel)-|\(±\)-|\(\+/-\)-|\(RS\)-|\(SR\)-")


def strip(name: str) -> str:
    """立体記号のかたまり((2S,3R)- や [(1R)-...] の中の (1R)-)をすべて外す。"""
    s = _RACREL.sub("", _GROUP.sub("", name))
    return re.sub(r"\s+", " ", s).strip()


def inchikey(mol) -> str:
    try:
        return (Chem.MolToInchiKey(mol) or "") if mol is not None else ""
    except Exception:
        return ""


def count(mol: Chem.Mol) -> int:
    n = len(Chem.FindMolChiralCenters(mol, includeUnassigned=False, useLegacyImplementation=False))
    n += sum(1 for b in mol.GetBonds() if b.GetStereo() not in (Chem.BondStereo.STEREONONE, Chem.BondStereo.STEREOANY))
    return n


def _elements(mol: Chem.Mol):
    """入力構造の立体要素: ('atom', idx, 'R') / ('bond', (i, j), 'E')"""
    m = Chem.Mol(mol)
    rdCIPLabeler.AssignCIPLabels(m)
    out = []
    for a in m.GetAtoms():
        if a.HasProp("_CIPCode") and a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
            out.append(("atom", a.GetIdx(), a.GetProp("_CIPCode")))
    for b in m.GetBonds():
        if b.HasProp("_CIPCode") and b.GetStereo() not in (Chem.BondStereo.STEREONONE, Chem.BondStereo.STEREOANY):
            out.append(("bond", (b.GetBeginAtomIdx(), b.GetEndAtomIdx()), b.GetProp("_CIPCode")))
    return out


def _partial(mol: Chem.Mol, keep: list) -> Chem.Mol:
    """keep に入っている立体要素だけを残した構造。"""
    m = Chem.Mol(mol)
    atoms = {e[1] for e in keep if e[0] == "atom"}
    bonds = {frozenset(e[1]) for e in keep if e[0] == "bond"}
    for a in m.GetAtoms():
        if a.GetIdx() not in atoms:
            a.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    for b in m.GetBonds():
        if frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx())) not in bonds:
            b.SetStereo(Chem.BondStereo.STEREONONE)
    smi = Chem.MolToSmiles(m)
    return Chem.MolFromSmiles(smi)


def _locant_key(loc: str):
    m = re.match(r"(\d+)([a-z]?)('*)", loc)
    return (int(m.group(1)), m.group(2), m.group(3)) if m else (10**6, loc, "")


def _positions(name: str, in_substituent: bool) -> list[int]:
    """立体記号を入れてよい位置。置換基の原子らしければ括弧の直後を先に、
    そうでなければ空白の後(エステルの酸の側 methyl (2S)-...)・先頭を先に試す。"""
    front, space, bracket = [0], [], []
    for i, ch in enumerate(name):
        if ch == " " and i + 1 < len(name):
            space.append(i + 1)
        elif ch in "([{" and i + 1 < len(name):
            # (2,3) や縮環の [a] のような括弧は除く
            close = name.find(")" if ch == "(" else "]" if ch == "[" else "}", i)
            inner = name[i + 1:close] if close > 0 else ""
            if (not re.fullmatch(r"[\d,'a-zA-Z.]{0,8}", inner) or "-" in inner) and (name[i + 1] not in "([{" or _GROUP.match(name, i + 1)):
                bracket.append(i + 1)
    bracket.reverse()  # 内側の括弧(後ろに出てくるもの)から
    order = bracket + space + front if in_substituent else space + front + bracket
    return list(dict.fromkeys(order))


def _bracket_pairs(name: str):
    stack, pairs = [], []
    for i, ch in enumerate(name):
        if ch in "([{":
            stack.append(i)
        elif ch in ")]}" and stack:
            pairs.append((stack.pop(), i))
    return pairs if not stack else None


def tidy_brackets(name: str) -> str:
    """囲みの入れ子を IUPAC の順 ( ) → [ ] → { } に直す。縮環・橋かけの [a] [2.2.1] は触らない。"""
    pairs = _bracket_pairs(name)
    if not pairs:
        return name
    fixed = {o for o, c in pairs if name[o] == "[" and re.fullmatch(r"[\d,.'a-z-]*", name[o + 1:c])
             and not re.search(r"[a-z]{2}", name[o + 1:c])}
    height = {}
    for o, c in sorted(pairs, key=lambda p: p[1] - p[0]):
        inner = [height[(o2, c2)] for (o2, c2) in height if o < o2 and c2 < c and o2 not in fixed]
        height[(o, c)] = (max(inner) + 1) if inner else 0
    chars = list(name)
    for (o, c), h in height.items():
        if o in fixed:
            continue
        chars[o], chars[c] = "([{"[h % 3], ")]}"[h % 3]
    return "".join(chars)


def _scope(name: str, pos: int) -> float:
    """pos に置いた立体記号が効く範囲の長さ(先頭は無限大)。"""
    if pos == 0:
        return float("inf")
    if name[pos - 1] == " ":
        return len(name) - pos
    depth = 0
    for i in range(pos, len(name)):
        if name[i] in "([{":
            depth += 1
        elif name[i] in ")]}":
            if depth == 0:
                return i - pos
            depth -= 1
    return len(name) - pos


def _insert(name: str, pos: int, desc: str) -> str:
    """pos に "(desc)-" を入れる。すでに立体記号のかたまりがあればその中に足す。"""
    m = _GROUP.match(name, pos)
    if m:
        inner = name[m.start() + 1:name.index(")", m.start())]
        parts = sorted([p.strip() for p in inner.split(",")] + [desc], key=_locant_key)
        return name[:pos] + "(" + ",".join(parts) + ")-" + name[m.end():]
    return name[:pos] + f"({desc})-" + name[pos:]


def assign(name: str, mol: Chem.Mol) -> str | None:
    """立体を外した名前に、mol の立体を位置番号つきで付ける。できなければ None。"""
    base = strip(name)
    target_key = inchikey(mol)
    elems = _elements(mol)
    if not elems:
        return base if inchikey(opsin.parse([base])[0].mol) == target_key else None
    r = opsin.parse([base], cml=True)[0]
    cmol = opsin.cml_to_mol(r.cml) if r.ok else None
    if cmol is None:
        return None
    flat = Chem.Mol(mol)
    Chem.RemoveStereochemistry(flat)
    match = cmol.GetSubstructMatch(flat)
    if not match or cmol.GetNumAtoms() != flat.GetNumAtoms():
        return None

    descs = []
    for kind, where, code in elems:
        if kind == "atom":
            loc = opsin.numeric_locant(cmol.GetAtomWithIdx(match[where]))
        else:
            locs = [opsin.numeric_locant(cmol.GetAtomWithIdx(match[i])) for i in where]
            locs = [l for l in locs if l]
            loc = min(locs, key=_locant_key) if locs else ""
        descs.append((loc, code))

    seen: dict[str, int] = {}
    for a in cmol.GetAtoms():
        lab = opsin.numeric_locant(a)
        seen[lab] = seen.get(lab, 0) + 1

    current, kept = base, []
    for elem, (loc, code) in zip(elems, descs):
        kept.append(elem)
        want = inchikey(_partial(mol, kept))
        # OPSIN は、括弧の中や 2 語目以降の前に置いた立体記号をその範囲の中だけで探すが、
        # 名前の先頭に置いたものはどこの原子にでも当てる。そこで「範囲が限られる位置」で照合が通ったもののうち
        # 一番狭い範囲を採り、先頭はほかに通る位置が無いときだけ使う。
        tries, prefs = [], []
        for order, p in enumerate(_positions(current, seen.get(loc, 0) > 1)):
            merged = bool(_GROUP.match(current, p))
            span = _scope(current, p)
            for k, d in enumerate(([f"{loc}{code}"] if loc else []) + [code]):
                cand = _insert(current, p, d)
                if cand not in tries:
                    tries.append(cand)
                    prefs.append((k, span, not merged, order))
        hits = [(pref, cand) for cand, pref, res in zip(tries, prefs, opsin.parse(tries))
                if res.ok and inchikey(res.mol) == want]
        if not hits:
            return None
        current = min(hits)[1]
    if inchikey(opsin.parse([current])[0].mol) != target_key:
        return None
    tidy = tidy_brackets(current)
    if tidy != current and inchikey(opsin.parse([tidy])[0].mol) == target_key:
        current = tidy
    return current


def brute(name: str, mol: Chem.Mol, limit: int = 256) -> str | None:
    """位置番号なしの (R,S,E) などを総当たり(assign が使えないとき用。要素数が少ないときだけ)。"""
    elems = _elements(mol)
    kinds = ["RS" if e[0] == "atom" else "EZ" for e in elems]
    if not elems or 2 ** len(elems) > limit:
        return None
    base = strip(name)
    key = inchikey(mol)
    tries = [f"({','.join(c)})-{base}" for c in product(*kinds)]
    for cand, res in zip(tries, opsin.parse(tries)):
        if res.ok and inchikey(res.mol) == key:
            return cand
    return None
