"""構造から計算で求める 元素分析(質量パーセント)と マススペクトル(同位体パターン)。実測値ではない。

  elemental_analysis(mol)  -> (MW, [(元素, 個数, 質量%), ...])    Hill 順(C, H, あとはアルファベット順)
  spectrum(mol, ion)       -> [(m/z, 相対強度%), ...]            nominal mass ごとに束ねた同位体パターン(低分解能の見え方)
  ions(mol)                -> {イオン名: 単一同位体 m/z}
  ea_text(mol) / ms_text(mol)  CSV・コマンド用の 1 行の文字列

元素分析は塩・混合物では全成分の合算(試料全体の組成)。マススペクトルは、測定するイオンに合う成分(陽イオンの測定には陽イオン、
陰イオンの測定には陰イオン、無ければ原子数が最大の成分)について計算する。同位体の存在比は IUPAC の天然存在比(主要な元素のみ)。
表にない元素を含むときは、精密質量は出すが同位体パターンは主な同位体 1 本だけになる(`approx` が True)。
"""
from __future__ import annotations

from rdkit import Chem

ELECTRON = 0.00054857990946   # 電子の質量(u)
PROTON = 1.007276466621       # H+ の質量(u)
NA_ION = 22.98976928 - ELECTRON
K_ION = 38.96370668 - ELECTRON

# 元素 -> [(同位体の質量, 天然存在比), ...]
ISO: dict[str, list[tuple[float, float]]] = {
    "H": [(1.00782503207, 0.999885), (2.0141017778, 0.000115)],
    "Li": [(6.0151223, 0.0759), (7.0160040, 0.9241)],
    "B": [(10.0129370, 0.199), (11.0093054, 0.801)],
    "C": [(12.0, 0.9893), (13.0033548378, 0.0107)],
    "N": [(14.0030740048, 0.99636), (15.0001088982, 0.00364)],
    "O": [(15.99491461956, 0.99757), (16.99913170, 0.00038), (17.9991610, 0.00205)],
    "F": [(18.99840322, 1.0)],
    "Na": [(22.9897692809, 1.0)],
    "Mg": [(23.9850417, 0.7899), (24.9858370, 0.1000), (25.9825930, 0.1101)],
    "Si": [(27.9769265325, 0.92223), (28.976494700, 0.04685), (29.97377017, 0.03092)],
    "P": [(30.97376163, 1.0)],
    "S": [(31.97207100, 0.9499), (32.97145876, 0.0075), (33.96786690, 0.0425), (35.96708076, 0.0001)],
    "Cl": [(34.96885268, 0.7576), (36.96590259, 0.2424)],
    "K": [(38.96370668, 0.932581), (39.96399848, 0.000117), (40.96182576, 0.067302)],
    "Ca": [(39.96259098, 0.96941), (41.95861801, 0.00647), (42.9587666, 0.00135), (43.9554818, 0.02086),
           (45.9536926, 0.00004), (47.952534, 0.00187)],
    "Fe": [(53.9396105, 0.05845), (55.9349375, 0.91754), (56.9353940, 0.02119), (57.9332756, 0.00282)],
    "Cu": [(62.9295975, 0.6915), (64.9277895, 0.3085)],
    "Zn": [(63.9291422, 0.4917), (65.9260334, 0.2773), (66.9271273, 0.0404), (67.9248442, 0.1845), (69.9253193, 0.0061)],
    "Br": [(78.9183371, 0.5069), (80.9162906, 0.4931)],
    "I": [(126.904473, 1.0)],
}

_PT = Chem.GetPeriodicTable()


_POS = ("[M+H]+", "[M+Na]+", "[M+K]+", "M+•", "M+")


def pick_fragment(mol: Chem.Mol, ion: str = "M+") -> Chem.Mol:
    """マススペクトルを計算する成分。塩では、測るイオンの符号に合う成分(最大のもの)。"""
    frags = Chem.GetMolFrags(mol, asMols=True)
    if len(frags) == 1:
        return mol
    biggest = lambda fs: max(fs, key=lambda m: m.GetNumAtoms())
    cations = [f for f in frags if Chem.GetFormalCharge(f) > 0]
    anions = [f for f in frags if Chem.GetFormalCharge(f) < 0]
    if ion in _POS and cations:
        return biggest(cations)
    if ion == "[M-H]-" and anions:
        return biggest(anions)
    return biggest(frags)


def composition(mol: Chem.Mol, whole: bool = False, ion: str = "M+") -> dict[str, int]:
    """水素を含めた元素ごとの個数(同位体ラベルは無視して天然存在比で計算する)。whole なら全成分の合算。"""
    m = Chem.AddHs(mol if whole else pick_fragment(mol, ion))
    out: dict[str, int] = {}
    for a in m.GetAtoms():
        s = a.GetSymbol()
        if s == "*":
            continue
        out[s] = out.get(s, 0) + 1
    return out


def hill_order(counts: dict[str, int]) -> list[str]:
    syms = sorted(counts)
    if "C" in counts:
        return [s for s in ("C", "H") if s in counts] + [s for s in syms if s not in ("C", "H")]
    return syms


def elemental_analysis(mol: Chem.Mol) -> tuple[float, list[tuple[str, int, float]]]:
    """-> (分子量, [(元素, 個数, 質量%)])。塩・混合物は最大の成分。"""
    counts = composition(mol, whole=True)   # 元素分析は試料全体(塩は全成分)
    mw = sum(_PT.GetAtomicWeight(s) * n for s, n in counts.items())
    return mw, [(s, counts[s], 100.0 * _PT.GetAtomicWeight(s) * counts[s] / mw) for s in hill_order(counts)]


def _mono(counts: dict[str, int]) -> float:
    return sum((ISO[s][0][0] if s in ISO else _PT.GetMostCommonIsotopeMass(s)) * n for s, n in counts.items())


def _convolve(a: dict[int, tuple[float, float]], b: dict[int, tuple[float, float]], cut: float) -> dict[int, tuple[float, float]]:
    """nominal mass -> (重み, 平均質量)。同じ nominal mass は重み付き平均で束ねる。"""
    out: dict[int, tuple[float, float]] = {}
    for na, (wa, ma) in a.items():
        for nb, (wb, mb) in b.items():
            w = wa * wb
            if w < cut:
                continue
            n, m = na + nb, ma + mb
            if n in out:
                w0, m0 = out[n]
                out[n] = (w0 + w, (w0 * m0 + w * m) / (w0 + w))
            else:
                out[n] = (w, m)
    return out


def _atom_pattern(sym: str) -> dict[int, tuple[float, float]]:
    d: dict[int, tuple[float, float]] = {}
    for m, ab in ISO[sym]:
        n = round(m)
        w0, m0 = d.get(n, (0.0, 0.0))
        d[n] = (w0 + ab, (w0 * m0 + ab * m) / (w0 + ab))
    return d


def approx(mol: Chem.Mol, ion: str = "M+") -> bool:
    """同位体の表にない元素を含む(パターンを出せない)か。"""
    return any(s not in ISO for s in composition(mol, ion=ion))


def neutral_pattern(mol: Chem.Mol, cut: float = 1e-10, ion: str = "M+") -> list[tuple[float, float]]:
    """-> [(質量, 相対強度%)]。最強のピークを 100% とする。表にない元素は主な同位体 1 本として扱う。"""
    counts = composition(mol, ion=ion)
    total: dict[int, tuple[float, float]] = {0: (1.0, 0.0)}
    for sym, n in counts.items():
        base = _atom_pattern(sym) if sym in ISO else {round(_PT.GetMostCommonIsotopeMass(sym)): (1.0, _PT.GetMostCommonIsotopeMass(sym))}
        power: dict[int, tuple[float, float]] = {0: (1.0, 0.0)}
        k = n
        while k:   # 二乗法で n 個分を畳み込む
            if k & 1:
                power = _convolve(power, base, cut)
            base = _convolve(base, base, cut)
            k >>= 1
        total = _convolve(total, power, cut)
    top = max(w for w, _ in total.values())
    return sorted((m, 100.0 * w / top) for _, (w, m) in total.items())


def ions(mol: Chem.Mol) -> dict[str, float]:
    """計算で求めた単一同位体(最も軽い同位体)の m/z。電荷をもつ分子はそのイオン自体。"""
    m = pick_fragment(mol)
    mono = _mono(composition(m))
    z = Chem.GetFormalCharge(m)
    if z:
        mz = (mono - z * ELECTRON) / abs(z)
        return {f"M{'+' if z > 0 else '-'}" + (f"({abs(z)})" if abs(z) > 1 else ""): mz}
    return {"M": mono, "[M+H]+": mono + PROTON, "[M+Na]+": mono + NA_ION, "[M+K]+": mono + K_ION,
            "[M-H]-": mono - PROTON, "M+•": mono - ELECTRON}


_SHIFT = {"M": 0.0, "[M+H]+": PROTON, "[M+Na]+": NA_ION, "[M+K]+": K_ION, "[M-H]-": -PROTON, "M+•": -ELECTRON, "M+": -ELECTRON}


def spectrum(mol: Chem.Mol, ion: str = "M+", min_rel: float = 0.1) -> list[tuple[float, float]]:
    """-> [(m/z, 相対強度%)]。min_rel% 未満は省く。電荷をもつ分子は ion に関わらずそのイオン。"""
    m = pick_fragment(mol, ion)
    z = Chem.GetFormalCharge(m)
    pat = neutral_pattern(m, ion=ion)
    if z:
        pts = [((mass - z * ELECTRON) / abs(z), rel) for mass, rel in pat]
    else:
        sh = _SHIFT.get(ion, PROTON)
        pts = [(mass + sh, rel) for mass, rel in pat]
    return [(mz, rel) for mz, rel in pts if rel >= min_rel]


def ea_text(mol: Chem.Mol) -> str:
    mw, rows = elemental_analysis(mol)
    return "  ".join(f"{s} {p:.2f}" for s, _, p in rows) + " (%)"


def ms_text(mol: Chem.Mol, ion: str = "M+", top: int = 5) -> str:
    m = pick_fragment(mol, ion)
    z = Chem.GetFormalCharge(m)
    pts = spectrum(mol, ion)
    pts = sorted(pts, key=lambda p: -p[1])[:top]
    pts.sort()
    label = ion if not z else next(iter(ions(m)))
    return f"{label}: " + "  ".join(f"{mz:.4f}({rel:.0f})" if rel >= 1 else f"{mz:.4f}({rel:.1f})" for mz, rel in pts)
