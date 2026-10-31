"""グラフニューラルネット(GPU)で、フラグメントの出やすさを学習する。n2s_fragml(勾配ブースティング)の置き換え候補。

分子を原子と結合のグラフとして読み(メッセージパッシング)、候補フラグメントごとに、
  含まれる原子の埋め込みの平均・和 / 切った結合(最大 3 本)の両側の原子の埋め込み / 分子全体の埋め込み / 手作りの特徴量(n2s_fragml.frag_features)
をまとめて、水素の増減 k ごとの強度(sqrt)を同時に予測する。PyTorch だけで書く(torch_geometric は使わない)。

  python n2s_fragnn.py --group ESI+ --ntrain 4000 --epochs 40 --save
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, RDLogger

import msdb_bench as bench
import n2s_fragml as fm
import n2s_fragml_train as tr
import n2s_msdb as db

RDLogger.DisableLog("rdApp.*")
MODELS = Path(__file__).resolve().parent / "data" / "fragnn"
MAXA = 48            # フラグメントの原子数の上限(これを超える分子は扱わない)
NATOM_F, NBOND_F = 30, 8
_HY = {Chem.HybridizationType.SP: 0, Chem.HybridizationType.SP2: 1, Chem.HybridizationType.SP3: 2}
_ELEM = fm._ELEM[:10]


# ---------------------------------------------------------------- 前処理(CPU、分子ごと)

def atom_features(m: Chem.Mol) -> np.ndarray:
    out = np.zeros((m.GetNumAtoms(), NATOM_F), dtype=np.float32)
    for a in m.GetAtoms():
        i = a.GetIdx()
        out[i, fm._EIDX.get(a.GetSymbol(), 10) if fm._EIDX.get(a.GetSymbol(), 10) < 10 else 10] = 1.0
        out[i, 11] = float(a.GetIsAromatic()); out[i, 12] = float(a.IsInRing())
        out[i, 13 + min(a.GetDegree(), 4)] = 1.0                 # 13..17
        out[i, 18 + min(a.GetTotalNumHs(), 3)] = 1.0             # 18..21
        out[i, 22 + _HY.get(a.GetHybridization(), 1)] = 1.0      # 22..24
        out[i, 25] = a.GetFormalCharge()
        out[i, 26] = float(a.GetSymbol() == "C" and any(b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtom(a).GetSymbol() == "O" for b in a.GetBonds()))
        out[i, 27] = sum(n.GetSymbol() in ("N", "O", "S") for n in a.GetNeighbors()) / 3.0
        out[i, 28] = sum(n.GetIsAromatic() for n in a.GetNeighbors()) / 3.0
        out[i, 29] = a.GetMass() / 100.0
    return out


def bond_arrays(m: Chem.Mol):
    src, dst, ef = [], [], []
    bt = {Chem.BondType.SINGLE: 0, Chem.BondType.DOUBLE: 1, Chem.BondType.TRIPLE: 2, Chem.BondType.AROMATIC: 3}
    for b in m.GetBonds():
        f = np.zeros(NBOND_F, dtype=np.float32)
        f[bt.get(b.GetBondType(), 0)] = 1.0
        f[4] = float(b.GetIsConjugated()); f[5] = float(b.IsInRing())
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        for a, c in ((u, v), (v, u)):
            src.append(a); dst.append(c); ef.append(f)
    return np.array(src, dtype=np.int64), np.array(dst, dtype=np.int64), (np.array(ef, dtype=np.float32) if ef else np.zeros((0, NBOND_F), dtype=np.float32))


def prep_molecule(smiles: str, mode: str, recs=None, depth: int = 3, enum=None):
    """1 分子の学習 / 推論データ。recs があれば正解(強度)も付ける。enum = (Mol, フラグメント)を渡すと、列挙をやり直さない。"""
    mol = smiles if isinstance(smiles, Chem.Mol) else Chem.MolFromSmiles(smiles)     # Mol を渡すと原子の番号がそのまま使われる
    M, frs = enum or fm.enumerate_fragments(mol, depth=depth)
    frs = [f for f in frs if len(f.atoms) <= MAXA]
    if not frs:
        return None
    glob, tc, tm = fm.global_features(M)
    hand = np.array([fm.frag_features(M, f, tc, tm, M.n, glob) for f in frs], dtype=np.float32)
    nf = len(frs)
    atoms = np.full((nf, MAXA), -1, dtype=np.int16)
    cut_ion = np.full((nf, 3), -1, dtype=np.int16)
    cut_neu = np.full((nf, 3), -1, dtype=np.int16)
    for i, f in enumerate(frs):
        la = sorted(f.atoms)
        atoms[i, :len(la)] = la
        for c, bi in enumerate(f.cuts[:3]):
            b = M.m.GetBondWithIdx(bi)
            u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            inn = [x for x in (u, v) if x in f.atoms]
            if len(inn) == 1:
                cut_ion[i, c] = inn[0]
                cut_neu[i, c] = v if inn[0] == u else u
    ks = fm.KS[mode]
    mz = np.array([[fm.ion_mz(f.mass, k, mode) for k in ks] for f in frs])
    prec = tm + (tr.db_proton(mode)) if mode != "ei" else tm - fm.E
    valid = (mz > 12) & (mz < prec + 0.5)
    out = {"atom_x": atom_features(M.m), "bond": bond_arrays(M.m), "atoms": atoms, "cut_ion": cut_ion, "cut_neu": cut_neu,
           "hand": hand, "mz": mz.astype(np.float32), "valid": valid, "n_atoms": M.n,
           "gen": np.array([f.gen for f in frs]), "frag_atoms": [f.atoms for f in frs], "kinds": [f.kind for f in frs],
           "fmass": np.array([f.mass for f in frs]), "smiles": smiles if isinstance(smiles, str) else Chem.MolToSmiles(mol)}
    if recs is not None:
        flat = mz.ravel()
        order = np.argsort(flat)
        y = np.empty(len(flat)); y[order] = tr._label(flat[order], recs)
        out["y"] = np.sqrt(y.reshape(mz.shape)).astype(np.float32)
    return out


def _job(args):
    try:
        return prep_molecule(*args)
    except Exception:
        return None


# ---------------------------------------------------------------- モデル

class FragNet(torch.nn.Module):
    def __init__(self, nk: int, hand_dim: int, hidden: int = 128, layers: int = 4):
        super().__init__()
        H = hidden
        self.atom_in = torch.nn.Linear(NATOM_F, H)
        self.edge_in = torch.nn.Linear(NBOND_F, H)
        self.msg = torch.nn.ModuleList([torch.nn.Linear(2 * H, H) for _ in range(layers)])
        self.upd = torch.nn.ModuleList([torch.nn.GRUCell(H, H) for _ in range(layers)])
        self.norm = torch.nn.ModuleList([torch.nn.LayerNorm(H) for _ in range(layers)])
        self.hand = torch.nn.Sequential(torch.nn.Linear(hand_dim, 256), torch.nn.ReLU(), torch.nn.Linear(256, 128), torch.nn.ReLU())
        d = H * 2 + H * 6 + H + 128
        self.head = torch.nn.Sequential(torch.nn.Linear(d, 512), torch.nn.ReLU(), torch.nn.Dropout(0.1),
                                        torch.nn.Linear(512, 256), torch.nn.ReLU(), torch.nn.Linear(256, nk))
        self.H = H

    def atom_embed(self, ax, src, dst, ef):
        h = torch.relu(self.atom_in(ax))
        e = torch.relu(self.edge_in(ef))
        for msg, upd, norm in zip(self.msg, self.upd, self.norm):
            m = torch.relu(msg(torch.cat([h[src], e], dim=1)))
            agg = torch.zeros_like(h).index_add_(0, dst, m)
            h = norm(upd(agg, h))
        return h

    def forward(self, ax, src, dst, ef, mol_of_atom, nmol, frag_atoms, frag_mol, cut_ion, cut_neu, hand):
        h = self.atom_embed(ax, src, dst, ef)
        H = self.H
        # 分子全体
        cnt = torch.zeros(nmol, device=h.device).index_add_(0, mol_of_atom, torch.ones(len(h), device=h.device)).clamp(min=1)
        hmol = torch.zeros(nmol, H, device=h.device).index_add_(0, mol_of_atom, h) / cnt[:, None]
        # フラグメント内の原子(平均と和)
        mask = frag_atoms >= 0
        idx = frag_atoms.clamp(min=0).long()
        hf = h[idx] * mask[..., None]
        fsum = hf.sum(1)
        fmean = fsum / mask.sum(1, keepdim=True).clamp(min=1)
        # 切った結合の両側の原子
        def gather(ix):
            m = ix >= 0
            return h[ix.clamp(min=0).long()] * m[..., None]
        ci, cn = gather(cut_ion), gather(cut_neu)
        z = torch.cat([fmean, fsum / 8.0, ci.flatten(1), cn.flatten(1), hmol[frag_mol], self.hand(hand)], dim=1)
        return self.head(z)


def collate(items: list[dict], device):
    """分子を束ねて GPU のテンソルにする。"""
    ax, src, dst, ef, moa = [], [], [], [], []
    fa, fm_, ci, cn, hand, y, valid = [], [], [], [], [], [], []
    off = 0
    for mi, it in enumerate(items):
        ax.append(it["atom_x"]); s, d, e = it["bond"]
        src.append(s + off); dst.append(d + off); ef.append(e)
        moa.append(np.full(it["n_atoms"], mi))
        nf = len(it["atoms"])
        def sh(a):
            b = a.astype(np.int64).copy(); b[b >= 0] += off; return b
        fa.append(sh(it["atoms"])); ci.append(sh(it["cut_ion"])); cn.append(sh(it["cut_neu"]))
        fm_.append(np.full(nf, mi)); hand.append(it["hand"]); valid.append(it["valid"])
        if "y" in it:
            y.append(it["y"])
        off += it["n_atoms"]
    T = lambda a, dt=None: torch.as_tensor(np.concatenate(a) if isinstance(a, list) else a, device=device, dtype=dt)
    out = dict(ax=T(ax, torch.float32), src=T(src, torch.long), dst=T(dst, torch.long), ef=T(ef, torch.float32),
               mol_of_atom=T(moa, torch.long), nmol=len(items), frag_atoms=T(fa, torch.long), frag_mol=T(fm_, torch.long),
               cut_ion=T(ci, torch.long), cut_neu=T(cn, torch.long), hand=T(hand, torch.float32), valid=T(valid, torch.bool))
    if y:
        out["y"] = T(y, torch.float32)
    return out


def _forward(net, b, mu, sd):
    hand = torch.nan_to_num((b["hand"] - mu) / sd, nan=0.0)
    return net(b["ax"], b["src"], b["dst"], b["ef"], b["mol_of_atom"], b["nmol"], b["frag_atoms"], b["frag_mol"], b["cut_ion"], b["cut_neu"], hand)


class Trained:
    """学習済みの GNN(推論用)。"""

    def __init__(self, net, mu, sd, mode, device):
        self.net, self.mu, self.sd, self.mode, self.device = net.eval(), mu, sd, mode, device

    @torch.no_grad()
    def predict_items(self, items: list[dict]) -> list[np.ndarray]:
        out = []
        for i in range(0, len(items), 16):
            chunk = items[i:i + 16]
            b = collate(chunk, self.device)
            p = torch.relu(_forward(self.net, b, self.mu, self.sd)).cpu().numpy() ** 2
            pos = 0
            for it in chunk:
                nf = len(it["atoms"])
                out.append(p[pos:pos + nf]); pos += nf
        return out


def train_net(items: list[dict], mode: str, epochs: int = 40, lr: float = 1.5e-3, bs: int = 12, device="cuda", seed=0, log=print):
    torch.manual_seed(seed); np.random.seed(seed)
    nk = len(fm.KS[mode])
    allhand = np.concatenate([it["hand"] for it in items[: min(len(items), 600)]])
    mu = torch.tensor(np.nanmean(allhand, 0), device=device, dtype=torch.float32)
    sd = torch.tensor(np.nanstd(allhand, 0) + 1e-6, device=device, dtype=torch.float32)
    net = FragNet(nk, allhand.shape[1]).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * ((len(items) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    t0 = time.time()
    for ep in range(epochs):
        net.train()
        perm = np.random.permutation(len(items))
        tot, cnt = 0.0, 0
        for i in range(0, len(items), bs):
            b = collate([items[j] for j in perm[i:i + bs]], device)
            pred = _forward(net, b, mu, sd)
            w = torch.where(b["y"] > 0, 1.0, 0.25) * b["valid"]            # 正例を重く、負例を軽く
            loss = (w * (pred - b["y"]) ** 2).sum() / w.sum().clamp(min=1)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step(); sched.step()
            tot += loss.detach().item(); cnt += 1
        if ep % 5 == 0 or ep == epochs - 1:
            log(f"  epoch {ep + 1:3d}/{epochs}  loss {tot / cnt:.5f}  ({time.time() - t0:.0f} 秒)")
    return Trained(net, mu, sd, mode, device)


# ---------------------------------------------------------------- 保存・読み込み・推論

def save_trained(gnn: Trained, group: str, path: Path | None = None) -> Path:
    path = path or (MODELS / f"{group}.pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": gnn.net.state_dict(), "mu": gnn.mu.cpu(), "sd": gnn.sd.cpu(), "mode": gnn.mode,
                "nk": len(fm.KS[gnn.mode]), "hand_dim": int(gnn.mu.numel()), "hidden": gnn.net.H}, path)
    return path


def load_trained(group: str, device: str | None = None, path: Path | None = None) -> Trained | None:
    """保存した GNN を読む(無ければ None)。"""
    path = path or (MODELS / f"{group}.pt")
    if not path.exists():
        return None
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    d = torch.load(path, map_location=device)
    net = FragNet(d["nk"], d["hand_dim"], hidden=d.get("hidden", 128)).to(device)
    net.load_state_dict(d["state"])
    return Trained(net, d["mu"].to(device), d["sd"].to(device), d["mode"], device)


def predict_scores(gnn: Trained, smiles, enum=None):
    """1 分子(SMILES または Mol)の候補ごとの予測強度。-> (prep_molecule の結果, 強度 nf × nk)。扱えなければ (None, None)。"""
    it = prep_molecule(smiles, gnn.mode, enum=enum)
    if it is None:
        return None, None
    return it, gnn.predict_items([it])[0]


# ---------------------------------------------------------------- 評価

def peaks_from_scores(item: dict, scores: np.ndarray, mode: str, top: int = 15):
    """候補の予測強度(nf × nk)から、m/z ごとの最大を取り、上位のピーク [(m/z, 相対 %)] を返す。"""
    mz, valid = item["mz"], item["valid"]
    s = np.where(valid, scores, 0.0)
    flat = [(float(mz[i, k]), float(s[i, k])) for i in range(s.shape[0]) for k in range(s.shape[1]) if s[i, k] > 0]
    flat.sort(key=lambda t: -t[1])
    tol = 0.3 if mode == "ei" else 0.003
    taken, peaks = [], []
    for m_, v in flat:
        if any(abs(m_ - t) < tol for t in taken):
            continue
        taken.append(m_); peaks.append((m_, v))
        if len(peaks) >= top:
            break
    if not peaks:
        return []
    t0 = peaks[0][1] or 1.0
    return [(m_, 100.0 * v / t0) for m_, v in peaks]


def build_items(group: str, side: str, n: int, procs: int = 11, with_labels: bool = True):
    data = bench.load_compounds(n, side, seed=7 if side == "train" else 0)[group]
    mode = tr.MODE[group]
    jobs = []
    for ik, mol, recs in data:
        arr = tr._recs_arrays(recs)
        if arr:
            jobs.append((Chem.MolToSmiles(mol), mode, arr if with_labels else None))
    with mp.Pool(procs) as pool:
        items = [r for r in pool.imap_unordered(_job, jobs, chunksize=4) if r is not None]
    return items, {j[0]: j[2] for j in jobs}


def evaluate_models(group: str, gnn: Trained, gbm, n: int = 300):
    """検証用の化合物で、規則 / 勾配ブースティング / GNN / 2 つの平均 を比べる。"""
    mode = tr.MODE[group]
    items, arrs = build_items(group, "test", n)
    scores_gnn = gnn.predict_items(items)
    res = {"rules": [], "gbm": [], "gnn": [], "avg": []}
    for it, sg in zip(items, scores_gnn):
        recs = [db.Record("", "", "", "", "", 0.0, "", "", "", "", "", "", [(float(p), float(r)) for p, r in zip(e, rl)], decimals=3 if tol < 0.1 else 0)
                for e, rl, tol in arrs[it["smiles"]]]
        nf, nk = it["mz"].shape
        feats = it["hand"]
        X = np.vstack([np.hstack([np.full((nf, 1), k, dtype=np.float32), feats]) for k in fm.KS[mode]])
        sb = (np.clip(gbm.predict(X), 0, None) ** 2).reshape(nk, nf).T
        mol = Chem.MolFromSmiles(it["smiles"])
        try:
            rl_, _, _ = bench.fr.predict(mol, bench.GROUPS[group], engine="rules")
        except Exception:
            continue
        rp = sorted([(p.mz, p.rel) for p in rl_ if p.kind != "precursor"], key=lambda p: -p[1])[:15]
        variants = {"rules": rp, "gbm": peaks_from_scores(it, sb, mode), "gnn": peaks_from_scores(it, sg, mode),
                    "avg": peaks_from_scores(it, 0.5 * (sb / max(sb.max(), 1e-9) + sg / max(sg.max(), 1e-9)), mode)}
        if not all(variants.values()):
            continue
        for name, pk in variants.items():
            ms_ = [db.compare(pk, r) for r in recs]
            res[name].append((st.mean(m.explained for m in ms_), st.mean(m.precision for m in ms_)))
    f1 = lambda e, p: 2 * e * p / (e + p) if e + p else 0.0
    out = {}
    for name, v in res.items():
        if v:
            e, p = st.mean(x[0] for x in v), st.mean(x[1] for x in v)
            out[name] = (len(v), e, p, f1(e, p))
    return out


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="ESI+", choices=list(tr.MODE))
    ap.add_argument("--ntrain", type=int, default=4000)
    ap.add_argument("--ntest", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--procs", type=int, default=11)
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()
    import joblib
    t0 = time.time()
    items, _ = build_items(a.group, "train", a.ntrain, a.procs)
    nfr = sum(len(i["atoms"]) for i in items)
    print(f"[{a.group}] 学習データ: 分子 {len(items)}、フラグメント {nfr:,}  ({time.time() - t0:.0f} 秒)", flush=True)
    gnn = train_net(items, tr.MODE[a.group], a.epochs, log=lambda s: print(f"[{a.group}]" + s, flush=True))
    gbm = joblib.load(tr.MODELS / f"{a.group}.joblib")
    r = evaluate_models(a.group, gnn, gbm, a.ntest)
    print(f"[{a.group}] 検証用の比較(上位 15 ピーク):")
    for name, (n, e, p, f) in r.items():
        print(f"   {name:6s} n={n}  説明率 {e:.0%}  適合率 {p:.0%}  F1 {f:.3f}")
    if a.save:
        print("保存:", save_trained(gnn, a.group))
    print(f"所要 {time.time() - t0:.0f} 秒")
