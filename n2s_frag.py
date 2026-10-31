"""規則ベースのフラグメント(MS/MS)予測。実測ではなく、目安。

  predict(mol, ion)  -> (ピークのリスト, 注記, 前駆イオンの m/z)

2 種類のピークを作る:
  cleavage  環の外の単結合を切って 2 つに分け、片方をイオン(カチオン、負イオンモードならアニオン)、もう片方を中性として
            脱離させる。例: ベンジル位の C–N を切るとトロピリウム C7H7+(91.0542)。
  loss      官能基に特徴的な中性脱離(H2O・NH3・CO・CO2・HCOOH・CH3OH・HCl・SO2・NO2・ケテンなど)と、その 2 段階の組み合わせ。
  ring      環の 2 本の結合を切って 2 つに分ける(シクロヘキセンの逆ディールス・アルダー反応を含む)。
  第二段    強い第一段のフラグメントが、さらに中性脱離(CO・C2H2・HCN・…)や結合開裂で壊れる(アシリウム → CO、トロピリウム → C2H2 など)。
強度は、カチオンの安定性(芳香環・ベンジル位・アシリウム・α ヘテロ原子・分岐)や官能基ごとの経験則から付けた点数で、
最大を 100% に正規化した値。電荷をもつ分子(第四級アンモニウムなど)は未対応。
"""
from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem

import n2s_ms as ms

E = ms.ELECTRON

# 点数付けの重み(既定値)。MassBank の実測との照合(msdb_bench.py)で調整する。1.0 が調整前の値。
W = {"arom": 1.0, "benzylic": 1.0, "acylium": 1.0, "alpha": 1.0, "allylic": 1.0, "branch": 1.0, "basicN": 1.0,
     "small": 1.0,          # 小さな断片(m/z < 50)への罰の強さ(大きいほど強く罰する)
     "loss": 1.0,           # 中性脱離の点
     "gen2": 0.5,           # 第二段の点の倍率(0 で第二段を出さない)。MassBank で調整した値
     "ring": 0.5,           # 環開裂の点の倍率(0 で環開裂を出さない)。RDA は除く。MassBank で調整した値
     "alkane": 1.0,         # メタンなどの小さなアルキルの脱離への罰の強さ
     "anion": 1.0,
     "htrans": 0.0}         # 水素移動を伴うイオン [X+H]+ の点の倍率(0 で出さない)


@dataclass
class Peak:
    mz: float
    rel: float
    label: str            # 例: "C7H7+ (−C2H6N)" / "−H2O"
    formula: str          # イオンの組成式(例: C7H7+)
    smiles: str           # フラグメントの SMILES(切った位置は *)。中性脱離は前駆イオンからの脱離の説明
    atoms: tuple          # 元の分子の原子番号(強調表示用): cleavage / ring = イオン側、loss = 脱離する基
    kind: str             # "precursor" / "cleavage"(結合開裂)/ "loss"(中性脱離)/ "ring"(環開裂)
    gen: int = 1          # 何段階目のフラグメントか(2 = 第一段のフラグメントがさらに壊れたもの)


def _mass(sym: str) -> float:
    return ms.ISO[sym][0][0] if sym in ms.ISO else ms._PT.GetMostCommonIsotopeMass(sym)


def _counts(m: Chem.Mol, atoms) -> dict[str, int]:
    out: dict[str, int] = {}
    for i in atoms:
        a = m.GetAtomWithIdx(i)
        out[a.GetSymbol()] = out.get(a.GetSymbol(), 0) + 1
        h = a.GetTotalNumHs()
        if h:
            out["H"] = out.get("H", 0) + h
    return out


def _add(a: dict[str, int], b: dict[str, int], sign: int = 1) -> dict[str, int]:
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0) + sign * v
    return {k: v for k, v in out.items() if v}


def _mass_of(counts: dict[str, int]) -> float:
    return sum(_mass(s) * n for s, n in counts.items())


def _formula(counts: dict[str, int], charge: str = "") -> str:
    parts = []
    for s in ms.hill_order(counts):
        n = counts[s]
        parts.append(s if n == 1 else f"{s}{n}")
    return "".join(parts) + charge


def _side(m: Chem.Mol, start: int, other: int) -> set[int]:
    seen, stack = {start}, [start]
    while stack:
        i = stack.pop()
        for n in m.GetAtomWithIdx(i).GetNeighbors():
            j = n.GetIdx()
            if j not in seen and not (i == start and j == other):
                seen.add(j)
                stack.append(j)
    return seen


def _score_cation(m: Chem.Mol, side: set[int], anchor: int, mass: float) -> float:
    """カチオンの安定性の目安点(大きいほどピークが強い)。"""
    a = m.GetAtomWithIdx(anchor)
    s = 10.0
    if any(m.GetAtomWithIdx(i).GetIsAromatic() for i in side):
        s += 22 * W["arom"]
    nb = [n for n in a.GetNeighbors() if n.GetIdx() in side]
    if a.GetSymbol() == "C":
        if a.GetIsAromatic():
            s -= 18                                        # アリールカチオンは不安定(ほかのボーナスは付けない)
        else:
            if any(n.GetIsAromatic() for n in nb):
                s += 38 * W["benzylic"]                       # ベンジル(トロピリウム)
            if any(b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtom(a).GetSymbol() in ("O", "N")
                   for b in a.GetBonds() if b.GetOtherAtomIdx(a.GetIdx()) in side):
                s += 32 * W["acylium"]                        # アシリウム・ニトリリウム
            if any(n.GetSymbol() in ("N", "O", "S") for n in nb):
                s += 30 * W["alpha"]                          # α ヘテロ原子(イミニウム・オキソニウム)
            if any(b.GetBondType() == Chem.BondType.DOUBLE for n in nb for b in n.GetBonds() if n.GetSymbol() == "C"):
                s += 16 * W["allylic"]                        # アリル
            s += W["branch"] * {0: 0, 1: 0, 2: 6, 3: 12}.get(len(nb), 12)    # 置換が多いほど安定
    elif a.GetSymbol() in ("N", "O", "S"):
        s += 4
    if any(m.GetAtomWithIdx(i).GetSymbol() == "N" and not any(b.GetBondType() == Chem.BondType.DOUBLE and
           b.GetOtherAtom(m.GetAtomWithIdx(i)).GetSymbol() == "O" for b in m.GetAtomWithIdx(i).GetBonds()) for i in side):
        s += 12 * W["basicN"]                              # 塩基性の窒素を含む側に電荷が残りやすい
    if mass < 30:
        s *= 0.3 ** W["small"]
    elif mass < 50:
        s *= 0.55 ** W["small"]
    return max(s, 1.0)


def _score_anion(m: Chem.Mol, side: set[int], anchor: int) -> float | None:
    a = m.GetAtomWithIdx(anchor)
    if a.GetSymbol() not in ("O", "S", "N", "Cl", "Br", "I", "F"):
        return None
    s = 15.0
    if a.GetSymbol() == "O":
        if any(b.GetBondType() == Chem.BondType.DOUBLE for n in a.GetNeighbors() for b in n.GetBonds()
               if n.GetIdx() in side and b.GetOtherAtom(n).GetSymbol() == "O"):
            s += 40                                        # カルボキシラート
        if any(n.GetIsAromatic() for n in a.GetNeighbors()):
            s += 25                                        # フェノキシド
    if a.GetSymbol() in ("Cl", "Br", "I"):
        s += 30
    if any(m.GetAtomWithIdx(i).GetIsAromatic() for i in side):
        s += 10
    return s


# 中性脱離: (化学式, SMARTS, 正イオンの点, 負イオンの点, 脱離する原子の SMARTS 内の位置)
_LOSSES = [
    ({"H": 2, "O": 1}, "[OX2H][CX4]", 55, 18, (0,)),            # アルコールの脱水
    ({"H": 2, "O": 1}, "[OX2H]C=O", 40, 10, (0,)),               # カルボン酸(アシリウムが残る)
    ({"H": 2, "O": 1}, "[OX2H]c", 6, 6, (0,)),                   # フェノールの脱水はまれ
    ({"H": 3, "N": 1}, "[NX3;H2,H1;!$(N-C=O);!$(N-S)]", 48, 8, (0,)),
    ({"H": 3, "N": 1}, "[NX3;H2]C=O", 30, 5, (0,)),
    ({"C": 1, "O": 1}, "[CX3]=O", 28, 5, (0, 1)),
    ({"C": 1, "O": 2}, "[CX3](=O)[OX2H]", 22, 100, (0, 1, 2)),
    ({"C": 1, "H": 2, "O": 2}, "[CX3](=O)[OX2H]", 38, 4, (0, 1, 2)),
    ({"C": 1, "H": 4, "O": 1}, "[CH3][OX2][#6]", 46, 6, (0, 1)),
    ({"C": 1, "H": 2, "O": 1}, "[CH3][OX2]c", 22, 12, (0, 1)),
    ({"H": 1, "Cl": 1}, "[CX4]Cl", 50, 20, (1,)),
    ({"H": 1, "Br": 1}, "[CX4]Br", 50, 20, (1,)),
    ({"H": 1, "F": 1}, "[CX4]F", 35, 15, (1,)),
    ({"S": 1, "O": 2}, "[SX4](=O)(=O)", 48, 30, (0, 1, 2)),
    ({"N": 1, "O": 2}, "[N+](=O)[O-]", 34, 40, (0, 1, 2)),
    ({"N": 1, "O": 1}, "[N+](=O)[O-]", 24, 18, (0, 1)),
    ({"C": 2, "H": 2, "O": 1}, "[CH3]C(=O)[N,O]", 58, 10, (0, 1, 2)),   # ケテン
    ({"C": 2, "H": 4, "O": 2}, "[CH3]C(=O)O[#6]", 40, 8, (0, 1, 2, 3)),  # 酢酸
    ({"C": 2, "H": 4}, "[CH2][CH3]", 18, 4, (0, 1)),
]


_PATTS = {sm: Chem.MolFromSmarts(sm) for _, sm, *_ in _LOSSES}    # SMARTS は 1 回だけ読む


def _all_losses(m: Chem.Mol, neg: bool):
    out = []
    for counts, smarts, rp, rn, lost in _LOSSES:
        patt = _PATTS[smarts]
        matches = m.GetSubstructMatches(patt)
        if not matches:
            continue
        # 同じ原子を共有する重複を除く(例: 1 つの OH を 2 回数えない)。強調するのは脱離する原子だけ
        used, uniq = set(), []
        for mt in matches:
            if not used & set(mt):
                uniq.append(tuple(mt[i] for i in lost))
                used |= set(mt)
        out.append((counts, uniq, rn if neg else rp, smarts))
    return out


# 第一段のイオンがさらに壊れるときの、芳香環・複素環に特有の脱離(化学式, 条件, 点)
_RING_LOSSES = [
    ({"C": 2, "H": 2}, "aromatic", 30),        # アセチレン(トロピリウム C7H7+ -> C5H5+、フェニル C6H5+ -> C4H3+)
    ({"H": 1, "C": 1, "N": 1}, "aromatic_N", 40),   # HCN(含窒素芳香環)
    ({"C": 1, "O": 1}, "aryl_O", 36),          # CO(フェノール・アリールエーテル由来)
]


def _pieces(m: Chem.Mol, cut: list[int]) -> list[set[int]] | None:
    """cut の結合をすべて切ったときの連結成分(原子番号の集合)。"""
    rw = Chem.RWMol(m)
    for bi in sorted(cut, reverse=True):
        b = m.GetBondWithIdx(bi)
        rw.RemoveBond(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
    frags = Chem.GetMolFrags(rw.GetMol())
    return [set(f) for f in frags]


def predict(mol: Chem.Mol, ion: str = "M+", max_peaks: int = 60, engine: str = "auto"):
    """-> (ピーク, 注記, 前駆イオンの m/z)。ピークは m/z 順。強度は目安。
    engine = "auto"(学習したモデルがあれば使い、無ければ規則。GNN があれば GNN、無ければ勾配ブースティング)/ "ml"(同上、規則に戻らない)/ "gbm"(勾配ブースティングに固定)/ "rules"。"""
    if engine in ("auto", "ml", "gbm"):
        try:
            import n2s_fragml as fml
            m0 = ms.pick_fragment(mol, ion)
            ion0 = "[M+H]+" if ion == "[M+Na]+" else ion
            if not Chem.GetFormalCharge(m0) and m0.GetNumAtoms() <= 120 and fml.available(ion0):
                rows = fml.predict_ml(m0, ion0, max_peaks, min_rel=0.5, gnn=engine != "gbm")   # 表示は 0.5% まで(評価は上位 15 ピークだけを使う)
                if rows:
                    base = ms._mono(ms.composition(m0)) if hasattr(ms, "_mono") else 0.0
                    prec = base + {"[M+H]+": ms.PROTON, "[M-H]-": -ms.PROTON, "M+": -E, "M+•": -E}.get(ion0, ms.PROTON)
                    peaks = [Peak(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]) for r in rows]
                    cnt = _counts(m0, range(m0.GetNumAtoms()))
                    pc = _add(cnt, {"H": 1}) if ion0 == "[M+H]+" else (_add(cnt, {"H": 1}, -1) if ion0 == "[M-H]-" else cnt)
                    peaks.append(Peak(prec, 100.0, ion0, _formula(pc, "-" if ion0 == "[M-H]-" else "+"), "", (), "precursor"))
                    peaks.sort(key=lambda p: p.mz)
                    note = ("Na 付加分子は [M+H]+ として計算した。 " if ion == "[M+Na]+" else "") + \
                           ("MassBank の実測から、グラフニューラルネットで予測しました" if fml.last_engine == "gnn" else "MassBank の実測から学習した予測です") + "(強度は目安。実測ではありません)。"
                    if fml.trained_on(m0):
                        note += " この化合物のスペクトルは学習に使われています(実測との照合は独立ではありません)。"
                    return peaks, note, prec
        except Exception:
            if engine in ("ml", "gbm"):
                raise
    if ion == "[M+Na]+":
        note = "Na 付加分子は [M+H]+ として計算した(Na を含むフラグメントは予測しない)。"
        ion = "[M+H]+"
    else:
        note = ""
    m = ms.pick_fragment(mol, ion)
    if Chem.GetFormalCharge(m):
        raise ValueError("電荷をもつ分子は、フラグメントの予測に未対応です。")
    if m.GetNumAtoms() > 120:
        raise ValueError("分子が大きすぎます(重原子 120 個まで)。")
    neg = ion == "[M-H]-"
    radical = ion == "M+" or ion == "M+•"
    allatoms = range(m.GetNumAtoms())
    total = _counts(m, allatoms)
    mono = _mass_of(total)
    prec = mono + {"[M+H]+": ms.PROTON, "[M-H]-": -ms.PROTON, "M+": -E, "M+•": -E}.get(ion, ms.PROTON)
    ch = "-" if neg else "+"
    pc = _add(total, {"H": 1}, -1) if neg else (_add(total, {"H": 1}) if ion == "[M+H]+" else total)   # 前駆イオンの組成
    # (mz, raw, label, formula, smiles, atoms, kind, gen)
    cand: list[tuple[float, float, str, str, str, tuple, str, int]] = []
    ions1: list[tuple[float, float, set, int, str]] = []        # 第二段に進めるイオン (mz, raw, 原子集合, 電荷の位置, 組成式)

    # ---- 第一段: 結合開裂
    for bond in m.GetBonds():
        if bond.GetBondType() != Chem.BondType.SINGLE or bond.IsInRing() or bond.GetIsAromatic():
            continue
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        sa = _side(m, a, b)
        sb = set(allatoms) - sa
        if not sa or not sb:
            continue
        for X, anchor, Y in ((sa, a, sb), (sb, b, sa)):
            cx, cy = _counts(m, X), _counts(m, Y)
            if neg:
                if cy.get("H", 0) < 1:
                    continue
                raw = _score_anion(m, X, anchor)
                if raw is None:
                    continue
                mz = _mass_of(cx) + E
                loss = _add(cy, {"H": 1}, -1)
            else:
                mz = _mass_of(cx) - E
                raw = _score_cation(m, X, anchor, mz)
                loss = cy if radical else _add(cy, {"H": 1})
            if mz < 14:
                continue
            if set(loss) <= {"C", "H"} and loss.get("C", 0) <= 2 and not neg:
                raw *= 0.25 ** W["alkane"]   # メタンなどの小さなアルキルの脱離は、ESI のイオンではまれ
            piece = Chem.FragmentOnBonds(m, [bond.GetIdx()], addDummies=True)
            frags = Chem.GetMolFrags(piece, asMols=True)
            fidx = Chem.GetMolFrags(piece)
            smi = next((Chem.MolToSmiles(f) for f, ix in zip(frags, fidx) if anchor in ix), "")
            fm = _formula(cx, ch)
            cand.append((mz, raw, f"{fm}  (−{_formula(loss)})", fm, smi, tuple(sorted(X)), "cleavage", 1))
            if not neg:
                ions1.append((mz, raw, set(X), anchor, fm))
                if W["htrans"] > 0 and not radical and cy.get("H", 0) >= 1 and any(
                        m.GetAtomWithIdx(i).GetSymbol() in ("O", "N", "S") for i in X):
                    # 水素移動: 中性側の H が 1 つイオン側へ移る [X+H]+(中性は Y から H を引いたもの)
                    mzh = mz + ms.ISO["H"][0][0]
                    fmh = _formula(_add(cx, {"H": 1}), ch)
                    cand.append((mzh, raw * W["htrans"], f"{fmh}  (H 移動, −{_formula(_add(cy, {'H': 1}, -1))})", fmh,
                                 smi, tuple(sorted(X)), "cleavage", 1))

    # ---- 第一段: 環の開裂(2 本の結合を切る。シクロヘキセンの逆ディールス・アルダーを含む)
    if not neg and W["ring"] > 0:
        ri = m.GetRingInfo()
        for ring in ri.BondRings():
            if not (5 <= len(ring) <= 8) or any(m.GetBondWithIdx(bi).GetIsAromatic() for bi in ring):
                continue
            atoms_r = ri.AtomRings()[list(ri.BondRings()).index(ring)]
            dbl = [bi for bi in ring if m.GetBondWithIdx(bi).GetBondType() == Chem.BondType.DOUBLE]
            for i in range(len(ring)):
                for j in range(i + 1, len(ring)):
                    b1, b2 = m.GetBondWithIdx(ring[i]), m.GetBondWithIdx(ring[j])
                    if {b1.GetBeginAtomIdx(), b1.GetEndAtomIdx()} & {b2.GetBeginAtomIdx(), b2.GetEndAtomIdx()}:
                        continue
                    if b1.GetBondType() != Chem.BondType.SINGLE or b2.GetBondType() != Chem.BondType.SINGLE:
                        continue
                    parts = _pieces(m, [ring[i], ring[j]])
                    if len(parts) != 2:
                        continue
                    rda = False
                    if len(ring) == 6 and len(dbl) == 1:   # C1=C2 の両側の 2 つ先の結合を切ると、ジエン + アルケン
                        d = m.GetBondWithIdx(dbl[0])
                        ends = {d.GetBeginAtomIdx(), d.GetEndAtomIdx()}
                        near = {n.GetIdx() for e in ends for n in m.GetAtomWithIdx(e).GetNeighbors() if n.GetIdx() in atoms_r} - ends
                        cut_atoms = {b1.GetBeginAtomIdx(), b1.GetEndAtomIdx(), b2.GetBeginAtomIdx(), b2.GetEndAtomIdx()}
                        rda = len(near & cut_atoms) == 2 and not (ends & cut_atoms)
                    for X, Y in ((parts[0], parts[1]), (parts[1], parts[0])):
                        cx, cy = _counts(m, X), _counts(m, Y)
                        mz = _mass_of(cx) - E + (0.0 if radical else ms.PROTON)
                        if mz < 20:
                            continue
                        anchors = [a2 for b in (b1, b2) for a2 in (b.GetBeginAtomIdx(), b.GetEndAtomIdx()) if a2 in X]
                        base = max(_score_cation(m, X, a2, mz) for a2 in anchors) * 0.45
                        raw = max(base * W["ring"], 70.0) if rda else base * W["ring"]
                        fm = _formula(_add(cx, {"H": 1}) if not radical else cx, ch)
                        cand.append((mz, raw, f"{fm}  (環開裂{'・RDA' if rda else ''}, −{_formula(cy)})", fm,
                                     "", tuple(sorted(X)), "ring", 1))

    # ---- 第一段: 中性脱離(1 段階と 2 段階)
    losses = _all_losses(m, neg)
    singles = []
    for counts, uniq, raw, smarts in losses:
        lm = _mass_of(counts)
        atoms = tuple(sorted(set(i for mt in uniq[:1] for i in mt)))
        mz = prec - lm
        fm = _formula(_add(pc, counts, -1), ch)
        raw = raw * W["loss"]
        singles.append((counts, uniq, raw, lm))
        cand.append((mz, float(raw), f"−{_formula(counts)}", fm, f"{ion} − {_formula(counts)}", atoms, "loss", 1))
    for i, (c1, u1, r1, m1) in enumerate(singles):
        for j in range(i, len(singles)):
            c2, u2, r2, m2 = singles[j]
            if i == j and len(u1) < 2:
                continue
            if i != j and set(sum(u1[:1], ())) & set(sum(u2[:1], ())):
                continue
            both = _add(c1, c2)
            mz = prec - m1 - m2
            if mz < 14:
                continue
            atoms = tuple(sorted(set(sum(u1[:1], ())) | set(sum(u2[:1], ())) if i != j else set(sum(u1[:2], ()))))
            fm = _formula(_add(pc, both, -1), ch)
            cand.append((mz, 0.62 * min(r1, r2), f"−{_formula(c1)} −{_formula(c2)}", fm,
                         f"{ion} − {_formula(c1)} − {_formula(c2)}", atoms, "loss", 1))

    # ---- 第二段: 強い第一段のフラグメントイオンが、さらに壊れる
    if not neg and W["gen2"] > 0:
        loss_tab = [(counts, [mt for mt in m.GetSubstructMatches(_PATTS[sm])], rp, lost)
                    for counts, sm, rp, rn, lost in _LOSSES]
        for mz1, raw1, X, anchor, fm1 in sorted(ions1, key=lambda t: -t[1])[:14]:
            cx = _counts(m, X)
            # (a) 官能基の脱離(脱離する原子がすべて断片の中にあるもの)
            for counts, matches, rp, lost in loss_tab:
                for mt in matches:
                    la = [mt[i] for i in lost]
                    if set(mt) <= X and all(cx.get(k, 0) >= v for k, v in counts.items()):
                        mz2 = mz1 - _mass_of(counts)
                        if mz2 >= 14:
                            cand.append((mz2, 0.5 * W["gen2"] * min(raw1, rp), f"{_formula(_add(cx, counts, -1), ch)}  (m/z {mz1:.2f} から −{_formula(counts)})",
                                         _formula(_add(cx, counts, -1), ch), "", tuple(sorted(la)), "loss", 2))
                        break
            # (b) 芳香環・複素環の脱離
            has_arom = any(m.GetAtomWithIdx(i).GetIsAromatic() for i in X)
            has_aromN = any(m.GetAtomWithIdx(i).GetIsAromatic() and m.GetAtomWithIdx(i).GetSymbol() == "N" for i in X)
            has_arylO = any(m.GetAtomWithIdx(i).GetSymbol() == "O" and any(n.GetIsAromatic() for n in m.GetAtomWithIdx(i).GetNeighbors()) for i in X)
            for counts, cond, rp in _RING_LOSSES:
                ok = {"aromatic": has_arom, "aromatic_N": has_aromN, "aryl_O": has_arylO}[cond]
                if ok and all(cx.get(k, 0) >= v for k, v in counts.items()):
                    mz2 = mz1 - _mass_of(counts)
                    if mz2 >= 14:
                        cand.append((mz2, 0.55 * W["gen2"] * min(raw1, rp), f"{_formula(_add(cx, counts, -1), ch)}  (m/z {mz1:.2f} から −{_formula(counts)})",
                                     _formula(_add(cx, counts, -1), ch), "", tuple(sorted(X)), "loss", 2))
            # (c) 断片の中の結合開裂(電荷の位置を含む側が脱離する)
            for bond in m.GetBonds():
                u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                if u not in X or v not in X or bond.GetBondType() != Chem.BondType.SINGLE or bond.IsInRing() or bond.GetIsAromatic():
                    continue
                su = _side(m, u, v) & X
                sv = X - su
                if not su or not sv:
                    continue
                lost_side, keep, a_keep = (su, sv, v) if anchor in su else (sv, su, u)
                if anchor not in lost_side:
                    continue
                mz2 = _mass_of(_counts(m, keep)) - E
                if mz2 < 14:
                    continue
                raw2 = 0.5 * W["gen2"] * min(raw1, _score_cation(m, keep, a_keep, mz2))
                fm2 = _formula(_counts(m, keep), ch)
                cand.append((mz2, raw2, f"{fm2}  (m/z {mz1:.2f} から −{_formula(_counts(m, lost_side))})", fm2,
                             "", tuple(sorted(keep)), "cleavage", 2))

    # 同じ m/z(±0.005)は最強のものにまとめる
    cand.sort(key=lambda c: c[0])
    merged: list[list] = []
    for c in cand:
        if merged and abs(c[0] - merged[-1][0]) < 0.005:
            if c[1] > merged[-1][1]:
                keep_label = merged[-1][2]
                merged[-1] = list(c)
                merged[-1][2] = c[2] + ("  /  " + keep_label if keep_label != c[2] else "")
            elif c[2] not in merged[-1][2]:
                merged[-1][2] += "  /  " + c[2]
        else:
            merged.append(list(c))
    merged = [c for c in merged if c[0] < prec - 0.5]
    if not merged:
        return [Peak(prec, 100.0, ion, _formula(pc, ch), "", (), "precursor")], "予測できるフラグメントがありません。", prec
    top = max(c[1] for c in merged)
    peaks = [Peak(c[0], 100.0 * c[1] / top, c[2], c[3], c[4], c[5], c[6], c[7]) for c in merged]
    peaks = [p for p in peaks if p.rel >= 2.0]
    peaks = sorted(sorted(peaks, key=lambda p: -p.rel)[:max_peaks], key=lambda p: p.mz)
    peaks.append(Peak(prec, 100.0, ion, _formula(pc, ch), "", (), "precursor"))
    peaks.sort(key=lambda p: p.mz)
    return peaks, (note + " 規則ベースの予測で、強度は目安です(実測ではありません)。").strip(), prec
