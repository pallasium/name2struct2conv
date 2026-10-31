"""MassBank から学習するフラグメント予測(CFM-ID 型)の候補生成と特徴量。

分子から、起こりうるフラグメント(原子の集合)を全部列挙する:
  - 環の外の単結合を 1 回切る(両側)/ 環の 2 本の結合を切る(両側)
  - 上のフラグメントの中を、もう 1 回切る(2 回目の開裂)
各フラグメントに、水素の増減 k(m/z = 原子の質量 + k×H ± 電子)を付けて候補にする。
どの候補が実測のピークになるか(強度)を、切る結合の周りの構造・フラグメントの組成・中性脱離の組成などから学習する(n2s_fragml_train.py)。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from rdkit import Chem

import n2s_ms as ms

E = ms.ELECTRON
H = ms.ISO["H"][0][0]

_ELEM = ["C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I", "Si", "B", "Se", "Na", "K"]
_EIDX = {e: i for i, e in enumerate(_ELEM)}


def _atom_mass(a: Chem.Atom) -> float:
    s = a.GetSymbol()
    m = ms.ISO[s][0][0] if s in ms.ISO else ms._PT.GetMostCommonIsotopeMass(s)
    return m + a.GetTotalNumHs() * H


@dataclass
class Frag:
    atoms: frozenset
    cuts: tuple               # 切った結合の番号(1 回目、2 回目)
    kind: str                 # "cleave" / "ring"
    gen: int
    mass: float = 0.0
    counts: dict = field(default_factory=dict)


class Mol:
    """分子の前計算(原子ごとの質量・隣接・環)。"""

    def __init__(self, m: Chem.Mol):
        self.m = m
        self.n = m.GetNumAtoms()
        self.masses = np.array([_atom_mass(a) for a in m.GetAtoms()])
        self.adj = [[(nb.GetIdx(), m.GetBondBetweenAtoms(a.GetIdx(), nb.GetIdx()).GetIdx()) for nb in a.GetNeighbors()] for a in m.GetAtoms()]
        self.sym = [a.GetSymbol() for a in m.GetAtoms()]
        self.nh = [a.GetTotalNumHs() for a in m.GetAtoms()]
        ri = m.GetRingInfo()
        self.acyclic = [b.GetIdx() for b in m.GetBonds()
                        if b.GetBondType() == Chem.BondType.SINGLE and not b.IsInRing() and not b.GetIsAromatic()]
        self.ends = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in m.GetBonds()]
        self.side = {bi: self._sides(bi) for bi in self.acyclic}   # 環の外の単結合(橋)で切った両側の原子(分子全体で)
        self.ring_pairs = []
        for ring in ri.BondRings():
            if not (5 <= len(ring) <= 8) or any(m.GetBondWithIdx(i).GetIsAromatic() for i in ring):
                continue
            for i in range(len(ring)):
                for j in range(i + 1, len(ring)):
                    b1, b2 = m.GetBondWithIdx(ring[i]), m.GetBondWithIdx(ring[j])
                    if {b1.GetBeginAtomIdx(), b1.GetEndAtomIdx()} & {b2.GetBeginAtomIdx(), b2.GetEndAtomIdx()}:
                        continue
                    if b1.GetBondType() == Chem.BondType.SINGLE and b2.GetBondType() == Chem.BondType.SINGLE:
                        self.ring_pairs.append((ring[i], ring[j]))

    def _sides(self, bi: int) -> tuple[frozenset, frozenset]:
        """結合 bi(橋)を切ったときの、両端それぞれ側の原子の集合。"""
        out = []
        for s0 in self.ends[bi]:
            comp, stack = {s0}, [s0]
            while stack:
                i = stack.pop()
                for j, b in self.adj[i]:
                    if b != bi and j not in comp:
                        comp.add(j)
                        stack.append(j)
            out.append(frozenset(comp))
        return out[0], out[1]

    def split_bridge(self, atoms: frozenset, bi: int) -> list[frozenset]:
        """連結なフラグメント atoms の中の橋 bi(両端が atoms の中)を切った 2 つの成分。components(atoms, (bi,)) と同じ結果・同じ順序。"""
        su, sv = self.side[bi]
        cu, cv = atoms & su, atoms & sv
        return [cu, cv] if next(iter(atoms)) in su else [cv, cu]

    def components(self, atoms: frozenset, cut: tuple) -> list[frozenset]:
        """atoms の中で、cut の結合をすべて切ったときの連結成分。"""
        cutset = set(cut)
        seen: set[int] = set()
        out = []
        for s in atoms:
            if s in seen:
                continue
            comp, stack = {s}, [s]
            while stack:
                i = stack.pop()
                for j, bi in self.adj[i]:
                    if j in atoms and bi not in cutset and j not in comp:
                        comp.add(j)
                        stack.append(j)
            seen |= comp
            out.append(frozenset(comp))
        return out

    def mass_of(self, atoms) -> float:
        return float(self.masses[list(atoms)].sum())

    def counts_of(self, atoms) -> np.ndarray:
        c = np.zeros(len(_ELEM))
        for i in atoms:
            c[_EIDX.get(self.sym[i], 0)] += 1
            c[1] += self.nh[i]
        return c


def enumerate_fragments(mol: Chem.Mol, max_frag: int = 6000, depth: int = 2) -> tuple[Mol, list[Frag]]:
    """-> (前計算した分子, フラグメントのリスト。原子の集合で重複を除く)。"""
    M = Mol(mol)
    allat = frozenset(range(M.n))
    seen: dict[frozenset, Frag] = {}

    def add(atoms, cuts, kind, gen):
        if not atoms or atoms == allat or atoms in seen or len(seen) >= max_frag:
            return
        f = Frag(atoms, cuts, kind, gen)
        f.mass = M.mass_of(atoms)
        seen[atoms] = f

    level1: list[Frag] = []
    for bi in M.acyclic:
        for comp in M.split_bridge(allat, bi):
            add(comp, (bi,), "cleave", 1)
    for b1, b2 in M.ring_pairs:
        comps = M.components(allat, (b1, b2))
        if len(comps) == 2:
            for comp in comps:
                add(comp, (b1, b2), "ring", 1)
    level1 = list(seen.values())
    def inside_of(f):
        return [bi for bi in M.acyclic if M.ends[bi][0] in f.atoms and M.ends[bi][1] in f.atoms]
    for f in level1:                                  # 2 回目の開裂: フラグメントの中をもう 1 回切る
        for bi in inside_of(f):
            for comp in M.split_bridge(f.atoms, bi):
                add(comp, f.cuts + (bi,), "cleave", 2)
    if depth >= 3:                                     # 3 回目の開裂
        for f in [f for f in seen.values() if f.gen == 2]:
            for bi in inside_of(f):
                for comp in M.split_bridge(f.atoms, bi):
                    add(comp, f.cuts + (bi,), "cleave", 3)
    return M, list(seen.values())


KS = {"pos": (-2, -1, 0, 1, 2), "neg": (-2, -1, 0, 1), "ei": (-2, -1, 0, 1)}


def ion_mz(frag_mass: float, k: int, mode: str) -> float:
    return frag_mass + k * H + (E if mode == "neg" else -E)


# ---------------------------------------------------------------- 特徴量

_HYB = {Chem.HybridizationType.SP: 1, Chem.HybridizationType.SP2: 2, Chem.HybridizationType.SP3: 3}
_NA = float("nan")


def _atom_feat(M: Mol, i: int | None) -> list[float]:
    """原子の局所的な性質(10 個)。i が None なら NaN。"""
    if i is None:
        return [_NA] * 10
    a = M.m.GetAtomWithIdx(i)
    nb = list(a.GetNeighbors())
    carbonyl = float(a.GetSymbol() == "C" and any(b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtom(a).GetSymbol() == "O" for b in a.GetBonds()))
    return [_EIDX.get(a.GetSymbol(), 15), float(a.GetIsAromatic()), float(a.IsInRing()), a.GetDegree(), a.GetTotalNumHs(), carbonyl,
            sum(n.GetSymbol() in ("N", "O", "S") for n in nb), sum(n.GetIsAromatic() for n in nb),
            _HYB.get(a.GetHybridization(), 0), a.GetFormalCharge()]


def _cut_feats(M: Mol, f: Frag, max_cuts: int = 3) -> list[float]:
    out: list[float] = []
    for bi in (list(f.cuts) + [None] * max_cuts)[:max_cuts]:
        if bi is None:
            out += [_NA] * 21
            continue
        b = M.m.GetBondWithIdx(bi)
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        inside = [x for x in (u, v) if x in f.atoms]
        outside = [x for x in (u, v) if x not in f.atoms]
        ion = inside[0] if len(inside) == 1 else None          # このフラグメントがこの結合の片側を含むとき
        neu = outside[0] if len(outside) == 1 else (u if ion is not None and ion == v else None)
        out += _atom_feat(M, ion) + _atom_feat(M, neu) + [float(len(inside))]
    return out


def frag_features(M: Mol, f: Frag, total_counts: np.ndarray, total_mass: float, ntot: int, glob: list[float]) -> list[float]:
    """フラグメント 1 つの特徴量(k とは独立。k は学習時に足す)。"""
    cnt = M.counts_of(f.atoms)
    rest = total_counts - cnt
    n_ar = sum(M.m.GetAtomWithIdx(i).GetIsAromatic() for i in f.atoms)
    n_het = sum(M.sym[i] in ("N", "O", "S") for i in f.atoms)
    carbonyls = 0
    basicN = 0
    for i in f.atoms:
        a = M.m.GetAtomWithIdx(i)
        if a.GetSymbol() == "C" and any(b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtom(a).GetSymbol() == "O" and b.GetOtherAtomIdx(i) in f.atoms for b in a.GetBonds()):
            carbonyls += 1
        if a.GetSymbol() == "N" and not a.GetIsAromatic() and not any(n.GetSymbol() == "C" and any(bb.GetBondType() == Chem.BondType.DOUBLE and bb.GetOtherAtom(n).GetSymbol() == "O" for bb in n.GetBonds()) for n in a.GetNeighbors()):
            basicN += 1
    base = [f.gen, float(f.kind == "ring"), len(f.cuts), len(f.atoms), len(f.atoms) / max(ntot, 1), f.mass, f.mass / max(total_mass, 1.0),
            total_mass - f.mass, n_ar, n_het, carbonyls, basicN]
    return base + list(cnt) + list(rest) + _cut_feats(M, f) + glob


def global_features(M: Mol) -> tuple[list[float], np.ndarray, float]:
    tc = M.counts_of(range(M.n))
    arom = sum(a.GetIsAromatic() for a in M.m.GetAtoms())
    return [M.n, float(M.masses.sum()), arom, M.m.GetRingInfo().NumRings(), tc[_EIDX["N"]], tc[_EIDX["O"]]], tc, float(M.masses.sum())


# ---------------------------------------------------------------- 推論(学習したモデルでフラグメントのピークを予測)

from pathlib import Path as _Path

MODEL_DIR = _Path(__file__).resolve().parent / "data" / "fragml"
_MODE = {"[M+H]+": ("ESI+", "pos"), "[M-H]-": ("ESI-", "neg"), "M+": ("EI", "ei"), "M+•": ("EI", "ei")}
_models: dict = {}
DEPTH = 3


def model_for(ion: str):
    """イオンに合う学習済みモデル(無ければ None)。"""
    if ion not in _MODE:
        return None
    name = _MODE[ion][0]
    if name not in _models:
        f = MODEL_DIR / f"{name}.joblib"
        try:
            import joblib
            _models[name] = joblib.load(f) if f.exists() else None
        except Exception:
            _models[name] = None
    return _models[name]


_gnns: dict = {}
last_engine = "gbm"      # 直近の predict_ml が使った方式("gnn" / "gbm")


def gnn_for(ion: str):
    """イオンに合う学習済みの GNN(data/fragnn/<群>.pt。無い・読めなければ None)。torch は使うときだけ読み込む。"""
    if ion not in _MODE:
        return None
    name = _MODE[ion][0]
    if name not in _gnns:
        try:
            import n2s_fragnn
            _gnns[name] = n2s_fragnn.load_trained(name)
        except Exception:
            _gnns[name] = None
    return _gnns[name]


def available(ion: str) -> bool:
    return model_for(ion) is not None


def trained_on(mol: Chem.Mol) -> bool:
    """この化合物(InChIKey の骨格)のスペクトルが、学習に使われたか(骨格のハッシュが偶数)。"""
    import hashlib
    try:
        ik = Chem.MolToInchiKey(mol)
    except Exception:
        return False
    return bool(ik) and int(hashlib.md5(ik[:14].encode()).hexdigest(), 16) % 2 == 0


def _fmt(counts: dict, charge: str = "") -> str:
    parts = []
    for sym in ms.hill_order({k: v for k, v in counts.items() if v > 0}):
        n = counts[sym]
        parts.append(sym if n == 1 else f"{sym}{n}")
    return "".join(parts) + charge


def _gnn_scores(gnn, mol: Chem.Mol, M: Mol, frs: list, ks: list) -> np.ndarray | None:
    """GNN の予測強度を、predict_ml の候補の並び(k ごとに frs を並べた形)に合わせる。扱えなければ None。
    GNN は原子数が上限(MAXA)を超えるフラグメントを落とすので、その候補の強度は 0 にする。"""
    import n2s_fragnn
    it, sc = n2s_fragnn.predict_scores(gnn, mol, enum=(M, frs))
    if it is None or sc.shape[1] != len(ks):
        return None
    kept = [i for i, f in enumerate(frs) if len(f.atoms) <= n2s_fragnn.MAXA]
    if len(kept) != sc.shape[0]:
        return None
    out = np.zeros((len(ks), len(frs)))
    out[:, kept] = sc.T
    return out.ravel()


def predict_ml(mol: Chem.Mol, ion: str, max_peaks: int = 60, min_rel: float = 2.0, gnn: bool = True):
    """-> [(m/z, 相対強度 %, ラベル, 組成式, SMILES, 原子, 種類, 段階)](m/z 順)。モデルが無ければ None。
    gnn=True で、学習済みの GNN があればそれで強度を出す(無い・扱えない分子は勾配ブースティング)。"""
    model = model_for(ion)
    if model is None:
        return None
    mode = _MODE[ion][1]
    M, frs = enumerate_fragments(mol, depth=DEPTH)
    if not frs:
        return []
    glob, tc, tm = global_features(M)
    ks = KS[mode]
    prec = tm + (-ms.PROTON if mode == "neg" else ms.PROTON) if mode != "ei" else tm - E
    global last_engine
    pred = None
    last_engine = "gbm"
    g = gnn_for(ion) if gnn else None
    if g is not None:
        try:
            pred = _gnn_scores(g, mol, M, frs, ks)
            if pred is not None:
                last_engine = "gnn"
        except Exception:
            pred = None
    if pred is None:
        feats = np.array([frag_features(M, f, tc, tm, M.n, glob) for f in frs], dtype=np.float32)
        X = np.vstack([np.hstack([np.full((len(frs), 1), k, dtype=np.float32), feats]) for k in ks])
        pred = np.clip(model.predict(X), 0, None) ** 2
    mz = np.concatenate([[ion_mz(f.mass, k, mode) for f in frs] for k in ks])
    fidx = np.tile(np.arange(len(frs)), len(ks))
    kk = np.repeat(np.array(ks), len(frs))
    ok = (mz > 12) & (mz < prec - 0.5)
    tol = 0.3 if mode == "ei" else 0.003
    total = {s: int(tc[_EIDX[s]]) for s in _ELEM if tc[_EIDX[s]]}
    pc = dict(total)
    if ion == "[M+H]+":
        pc["H"] = pc.get("H", 0) + 1
    elif ion == "[M-H]-":
        pc["H"] = pc.get("H", 0) - 1
    ch = "-" if mode == "neg" else "+"
    taken: list[float] = []
    rows = []
    for i in np.argsort(-pred):
        if not ok[i] or pred[i] <= 0:
            continue
        if any(abs(mz[i] - t) < tol for t in taken):
            continue
        taken.append(float(mz[i]))
        f = frs[fidx[i]]
        k = int(kk[i])
        fc = {s: int(c) for s, c in zip(_ELEM, M.counts_of(f.atoms)) if c}
        ion_c = dict(fc)
        ion_c["H"] = ion_c.get("H", 0) + k
        loss = {s: pc.get(s, 0) - ion_c.get(s, 0) for s in set(pc) | set(ion_c)}
        neutral_atoms = tuple(sorted(set(range(M.n)) - set(f.atoms)))
        small_loss = 0 < len(neutral_atoms) <= max(2, len(f.atoms) // 3)
        kind = "ring" if f.kind == "ring" else ("loss" if small_loss else "cleavage")
        atoms = neutral_atoms if kind == "loss" else tuple(sorted(f.atoms))
        label = f"{_fmt(ion_c, ch)}  (−{_fmt(loss)})" + ("" if f.gen == 1 else f"  [{f.gen} 回開裂]")
        try:
            smi = Chem.MolFragmentToSmiles(mol, atomsToUse=sorted(f.atoms))
        except Exception:
            smi = ""
        rows.append((float(mz[i]), float(pred[i]), label, _fmt(ion_c, ch), smi, atoms, kind, min(f.gen, 2)))
        if len(rows) >= max_peaks * 2:
            break
    if not rows:
        return []
    top = rows[0][1]
    out = [(r[0], 100.0 * r[1] / top, *r[2:]) for r in rows if 100.0 * r[1] / top >= min_rel][:max_peaks]
    out.sort(key=lambda r: r[0])
    return out
