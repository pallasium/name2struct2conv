"""化学物性の推算。構造だけから、基団寄与法などで見積もる(実測ではない)。

  沸点・融点・臨界温度/圧力/体積・生成ギブズエネルギー・生成熱 : Joback 法(K. G. Joback, R. C. Reid, Chem. Eng. Commun. 1987, 57, 233)
  logP・MR(モル屈折)          : Wildman–Crippen(S. A. Wildman, G. M. Crippen, J. Chem. Inf. Comput. Sci. 1999, 39, 868。RDKit の実装)
  tPSA                         : Ertl らの極性表面積(P. Ertl et al., J. Med. Chem. 2000, 43, 3714。RDKit の実装。N・O のみ)
  CLogP・CMR                   : 市販ソフトの値ではなく、Crippen の値で代用(≈。名前は一般的な「計算した logP・モル屈折」の意味)
  logS                         : ESOL(J. S. Delaney, J. Chem. Inf. Comput. Sci. 2004, 44, 1000)の式(≈)
  Henry 則定数・pKa            : 未対応(N/A)
"""
from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

# 基団: 名前 -> (Tc, Pc, Vc, Tb, Tm, Hf, Gf)。None = その基団の値が無い
_G = {
    "-CH3": (.0141, -.0012, 65, 23.58, -5.10, -76.45, -43.96), ">CH2": (.0189, 0, 56, 22.88, 11.27, -20.64, 8.42),
    ">CH-": (.0164, .0020, 41, 21.74, 12.64, 29.89, 58.36), ">C<": (.0067, .0043, 27, 18.25, 46.43, 82.23, 116.02),
    "=CH2": (.0113, -.0028, 56, 18.18, -4.32, -9.63, 3.77), "=CH-": (.0129, -.0006, 46, 24.96, 8.73, 37.97, 48.53),
    "=C<": (.0117, .0011, 38, 24.14, 11.14, 83.99, 92.36), "=C=": (.0026, .0028, 36, 26.15, 17.78, 142.14, 136.70),
    "#CH": (.0027, -.0008, 46, 9.20, -11.18, 79.30, 77.71), "#C-": (.0020, .0016, 37, 27.38, 64.32, 115.51, 109.82),
    "r-CH2-": (.0100, .0025, 48, 27.15, 7.75, -26.80, -3.68), "r>CH-": (.0122, .0004, 38, 21.78, 19.88, 8.67, 40.99),
    "r>C<": (.0042, .0061, 27, 21.32, 60.15, 79.72, 87.88), "r=CH-": (.0082, .0011, 41, 26.73, 8.13, 2.09, 11.30),
    "r=C<": (.0143, .0008, 32, 31.01, 37.02, 46.43, 54.05),
    "-F": (.0111, -.0057, 27, -0.03, -15.78, -251.92, -247.19), "-Cl": (.0105, -.0049, 58, 38.13, 13.55, -71.55, -64.31),
    "-Br": (.0133, .0057, 71, 66.86, 43.43, -29.48, -38.06), "-I": (.0068, -.0034, 97, 93.84, 41.69, 21.06, 5.74),
    "-OH(alc)": (.0741, .0112, 28, 92.88, 44.45, -208.04, -189.20), "-OH(phe)": (.0240, .0184, -25, 76.34, 82.83, -221.65, -197.37),
    "-O-": (.0168, .0015, 18, 22.42, 22.23, -132.22, -105.00), "r-O-": (.0098, .0048, 13, 31.22, 23.05, -138.16, -98.22),
    ">C=O": (.0380, .0031, 62, 76.75, 61.20, -133.22, -120.50), "r>C=O": (.0284, .0028, 55, 94.97, 75.97, -164.50, -126.27),
    "O=CH-": (.0379, .0030, 82, 72.24, 36.90, -162.03, -143.48), "-COOH": (.0791, .0077, 89, 169.09, 155.50, -426.72, -387.87),
    "-COO-": (.0481, .0005, 82, 81.10, 53.60, -337.92, -301.95), "=O": (.0143, .0101, 36, -10.50, 2.08, -247.61, -250.83),
    "-NH2": (.0243, .0109, 38, 73.23, 66.89, -22.02, 14.07), ">NH": (.0295, .0077, 35, 50.17, 52.66, 53.47, 89.39),
    "r>NH": (.0130, .0114, 29, 52.82, 101.51, 31.65, 75.61), ">N-": (.0169, .0074, 9, 11.74, 48.84, 123.34, 163.16),
    "-N=": (.0255, -.0099, None, 74.60, None, 23.61, None), "r-N=": (.0085, .0076, 34, 57.55, 68.40, 55.52, 79.93),
    "=NH": (None, None, None, 83.08, None, 93.70, 119.66), "-CN": (.0496, -.0101, 91, 125.66, 59.89, 88.43, 89.22),
    "-NO2": (.0437, .0064, 91, 152.54, 127.24, -66.57, -16.83),
    "-SH": (.0031, .0084, 63, 63.56, 20.09, -17.33, -22.99), "-S-": (.0119, .0049, 54, 68.78, 34.40, 41.87, 33.12),
    "r-S-": (.0019, .0051, 38, 52.10, 79.93, 39.10, 27.76),
}
_HAL = {"F": "-F", "Cl": "-Cl", "Br": "-Br", "I": "-I"}


def joback_groups(mol: Chem.Mol) -> dict[str, int] | None:
    """Joback の基団の数え上げ。扱えない原子・結合があれば None。"""
    m = Chem.Mol(mol)
    arom = {a.GetIdx(): a.GetIsAromatic() for a in m.GetAtoms()}
    try:
        Chem.Kekulize(m, clearAromaticFlags=True)
    except Exception:
        return None
    used: set[int] = set()
    out: dict[str, int] = {}

    def add(g):
        out[g] = out.get(g, 0) + 1

    def bonds(a):
        return [(b.GetBondType(), b.GetOtherAtom(a)) for b in a.GetBonds()]
    S, D, T = Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE
    for a in sorted(m.GetAtoms(), key=lambda x: x.GetSymbol() != "C"):     # 炭素を先に(COOH・エステル・ニトリルが O・N を取り込むため)
        if a.GetIdx() in used:
            continue
        sym, h, ring = a.GetSymbol(), a.GetTotalNumHs(), a.IsInRing()
        bs = bonds(a)
        if sym == "C":
            if a.GetFormalCharge():
                return None
            dO = [n for t, n in bs if t == D and n.GetSymbol() == "O"]
            tN = [n for t, n in bs if t == T and n.GetSymbol() == "N"]
            if tN:
                used |= {a.GetIdx(), tN[0].GetIdx()}; add("-CN"); continue
            if dO:
                o2 = [n for t, n in bs if t == S and n.GetSymbol() == "O" and n.GetFormalCharge() == 0]
                if o2:
                    o2.sort(key=lambda n: -n.GetTotalNumHs())      # OH があればそれを選ぶ(COOH)
                    used |= {a.GetIdx(), dO[0].GetIdx(), o2[0].GetIdx()}
                    add("-COOH" if o2[0].GetTotalNumHs() else "-COO-"); continue
                used |= {a.GetIdx(), dO[0].GetIdx()}
                add("O=CH-" if h else ("r>C=O" if ring else ">C=O")); continue
            nd = sum(t == D for t, _ in bs)
            nt = sum(t == T for t, _ in bs)
            if nt:
                add("#CH" if h else "#C-")
            elif nd == 2:
                add("=C=")
            elif nd == 1:
                add(("r=CH-" if h else "r=C<") if ring else ("=CH2" if h == 2 else ("=CH-" if h else "=C<")))
            else:
                add({3: "-CH3", 2: "r-CH2-" if ring else ">CH2", 1: "r>CH-" if ring else ">CH-", 0: "r>C<" if ring else ">C<"}.get(h, "") or "?")
                if "?" in out:
                    return None
            if ring and nd == 0 and h == 3:
                return None
        elif sym in _HAL:
            add(_HAL[sym])
        elif sym == "O":
            if a.GetFormalCharge():
                return None
            nb = len(bs)
            if h == 1 and nb == 1:
                add("-OH(phe)" if arom.get(bs[0][1].GetIdx()) else "-OH(alc)")
            elif h == 0 and nb == 2:
                add("r-O-" if ring else "-O-")
            elif nb == 1 and bs[0][0] == D:
                add("=O")
            else:
                return None
        elif sym == "N":
            os_ = [n for t, n in bs if n.GetSymbol() == "O"]
            if a.GetFormalCharge() == 1 and len(os_) == 2 and len(bs) == 3 and all(n.GetDegree() == 1 for n in os_):
                used |= {n.GetIdx() for n in os_}; add("-NO2"); continue
            if a.GetFormalCharge():
                return None
            nd = sum(t == D for t, _ in bs)
            if nd:
                add("=NH" if h else ("r-N=" if ring else "-N="))
            elif h == 2:
                add("-NH2")
            elif h == 1:
                add("r>NH" if ring else ">NH")
            elif h == 0 and len(bs) == 3:
                add(">N-")
            else:
                return None
        elif sym == "S":
            if a.GetFormalCharge() or any(t != S for t, _ in bs):
                return None
            add("-SH" if h == 1 else ("r-S-" if ring else "-S-")) if len(bs) <= 2 else None
        else:
            return None
    return out


def joback(mol: Chem.Mol) -> dict[str, float | None] | None:
    g = joback_groups(mol)
    if not g or any(k not in _G for k in g):
        return None

    def sm(i):
        vals = [(_G[k][i], n) for k, n in g.items()]
        return None if any(v is None for v, _ in vals) else sum(v * n for v, n in vals)
    natoms = Chem.AddHs(mol).GetNumAtoms()
    tb, tm, tcs, pcs, vc, hf, gf = sm(3), sm(4), sm(0), sm(1), sm(2), sm(5), sm(6)
    r: dict[str, float | None] = {
        "Tb": 198.2 + tb if tb is not None else None, "Tm": 122.5 + tm if tm is not None else None,
        "Vc": 17.5 + vc if vc is not None else None, "Hf": 68.29 + hf if hf is not None else None,
        "Gf": 53.88 + gf if gf is not None else None, "Tc": None, "Pc": None}
    if r["Tb"] is not None and tcs is not None and tcs > 0:
        r["Tc"] = r["Tb"] / (0.584 + 0.965 * tcs - tcs ** 2)
    if pcs is not None:
        d = 0.113 + 0.0032 * natoms - pcs
        r["Pc"] = d ** -2 if d > 0 else None
    return r


def esol(mol: Chem.Mol) -> float:
    """Delaney の ESOL(logS、mol/L)。"""
    logp = Crippen.MolLogP(mol)
    rb = rdMolDescriptors.CalcNumRotatableBonds(mol)
    heavy = mol.GetNumHeavyAtoms()
    ap = sum(a.GetIsAromatic() for a in mol.GetAtoms()) / heavy if heavy else 0.0
    return 0.16 - 0.63 * logp - 0.0062 * Descriptors.MolWt(mol) + 0.066 * rb - 0.74 * ap


# (キー, 表示名, 単位, 近似か)
ROWS = [("Tb", "Boiling Point", "K", False), ("Tm", "Melting Point", "K", False), ("Tc", "Critical Temp", "K", False),
        ("Pc", "Critical Pres", "Bar", False), ("Vc", "Critical Vol", "cm³/mol", False), ("Gf", "Gibbs Energy", "kJ/mol", False),
        ("logP", "Log P", "", False), ("MR", "MR", "cm³/mol", False), ("henry", "Henry's Law", "", False),
        ("Hf", "Heat of Form", "kJ/mol", False), ("tPSA", "tPSA", "", False), ("CLogP", "CLogP", "", True),
        ("CMR", "CMR", "", True), ("logS", "LogS", "", True), ("pKa", "pKa", "", False)]


def properties(mol: Chem.Mol) -> list[tuple[str, str, str, str, bool]]:
    """-> [(キー, 表示名, 値の文字列, 単位, 近似か)]。計算できない項目は値 "N/A"。複数成分は最大のものを使う。"""
    frags = Chem.GetMolFrags(mol, asMols=True)
    m = max(frags, key=lambda f: f.GetNumHeavyAtoms())
    jb = joback(m) or {}
    v: dict[str, float | None] = dict(jb)
    try:
        v["logP"] = Crippen.MolLogP(m)
        v["MR"] = Crippen.MolMR(m)
        v["tPSA"] = rdMolDescriptors.CalcTPSA(m)
        v["CLogP"] = v["logP"]
        v["CMR"] = v["MR"] / 10
        v["logS"] = esol(m)
    except Exception:
        pass
    fmt = {"Tb": "{:.2f}", "Tm": "{:.2f}", "Tc": "{:.2f}", "Pc": "{:.2f}", "Vc": "{:.1f}", "Gf": "{:.2f}", "Hf": "{:.2f}",
           "logP": "{:.2f}", "MR": "{:.2f}", "tPSA": "{:.2f}", "CLogP": "{:.3f}", "CMR": "{:.4f}", "logS": "{:.3f}"}
    rows = []
    for key, label, unit, approx in ROWS:
        x = v.get(key)
        rows.append((key, label, fmt[key].format(x) if x is not None else "N/A", unit if x is not None else "", approx and x is not None))
    return rows


def text(mol: Chem.Mol) -> str:
    return "\n".join(f"{lab}: {val}" + (f" [{u}]" if u else "") + (" (≈)" if ap else "") for _, lab, val, u, ap in properties(mol))
