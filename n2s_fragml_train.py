"""MassBank から、フラグメントの出やすさ(強度)を学習する(CFM-ID 型)。

  python n2s_fragml_train.py --group ESI+ --ntrain 2500 --ntest 300      学習 → 検証用の化合物で、今の規則と比べる
  群: ESI+ ([M+H]+ 高分解能) / ESI- ([M-H]- 高分解能) / EI (M+ 低分解能)
化合物は InChIKey の骨格のハッシュで、調整用 / 検証用に分ける(msdb_bench.py と同じ分け方。同じ化合物は両方に入らない)。
正解: 候補(フラグメント, 水素の増減 k)の m/z に実測ピークがあるか。複数のスペクトルの平均(強度 / 100)。
モデル: 勾配ブースティング(HistGradientBoostingRegressor)で sqrt(強度) を回帰。negative は間引いて重みで補う。
保存: data/fragml/<群>.joblib(MassBank から作ったローカルの学習結果。公開版には入れない)。
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import statistics as st
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from rdkit import Chem, RDLogger

import msdb_bench as bench
import n2s_fragml as fm
import n2s_msdb as db

RDLogger.DisableLog("rdApp.*")
MODELS = Path(__file__).resolve().parent / "data" / "fragml"
MODE = {"ESI+": "pos", "ESI-": "neg", "EI": "ei"}
DEPTH = 3
TOPN = 15


def _label(cmz: np.ndarray, recs: list[tuple[np.ndarray, np.ndarray, float]]) -> np.ndarray:
    """候補の m/z(昇順)に対する、実測ピークの強度(0〜1、スペクトルの平均)。"""
    y = np.zeros(len(cmz))
    for emz, erel, tol in recs:
        lab = np.zeros(len(cmz))
        for p, r in zip(emz, erel):
            lo, hi = np.searchsorted(cmz, [p - tol, p + tol])
            if hi > lo:
                lab[lo:hi] = np.maximum(lab[lo:hi], r / 100.0)
        y += lab
    return y / max(len(recs), 1)


def _rows(smiles: str, recs, mode: str, rneg: float, seed: int):
    """1 化合物の学習データ (X, y, w) と、全候補の情報。"""
    mol = Chem.MolFromSmiles(smiles)
    M, frs = fm.enumerate_fragments(mol, depth=DEPTH)
    if not frs:
        return None
    glob, tc, tm = fm.global_features(M)
    ks = fm.KS[mode]
    feats = np.array([fm.frag_features(M, f, tc, tm, M.n, glob) for f in frs], dtype=np.float32)
    nf = len(frs)
    mz = np.array([[fm.ion_mz(f.mass, k, mode) for k in ks] for f in frs])          # (nf, nk)
    prec = tm + (db_proton(mode)) if mode != "ei" else tm - fm.E
    valid = (mz > 12) & (mz < prec + 0.5)
    flat_mz = mz.ravel()
    order = np.argsort(flat_mz)
    y_sorted = _label(flat_mz[order], recs)
    y = np.empty(len(flat_mz)); y[order] = y_sorted
    y = y.reshape(nf, len(ks))
    rng = np.random.default_rng(seed)
    keep = valid & ((y > 0) | (rng.random(y.shape) < rneg))
    fi, ki = np.nonzero(keep)
    X = np.hstack([np.array(ks, dtype=np.float32)[ki][:, None], feats[fi]])
    w = np.where(y[fi, ki] > 0, 1.0, 1.0 / rneg).astype(np.float32)
    return X, np.sqrt(y[fi, ki]).astype(np.float32), w


def db_proton(mode: str) -> float:
    return -fm.ms.PROTON if mode == "neg" else fm.ms.PROTON


def _job(args):
    smiles, recs, mode, rneg, seed = args
    try:
        return _rows(smiles, recs, mode, rneg, seed)
    except Exception:
        return None


def _recs_arrays(recs):
    return [(np.array([p[0] for p in r.peaks]), np.array([p[1] for p in r.peaks]), db.tolerance(r)) for r in recs if len(r.peaks) >= 3]


def build_train(group: str, n: int, procs: int = 10, rneg: float = 0.06):
    data = bench.load_compounds(n, "train", seed=7)[group]
    jobs = []
    for i, (ik, mol, recs) in enumerate(data):
        arr = _recs_arrays(recs)
        if arr:
            jobs.append((Chem.MolToSmiles(mol), arr, MODE[group], rneg, i))
    with mp.Pool(procs) as pool:
        out = [r for r in pool.imap_unordered(_job, jobs, chunksize=8) if r is not None]
    X = np.vstack([o[0] for o in out]); y = np.concatenate([o[1] for o in out]); w = np.concatenate([o[2] for o in out])
    return X, y, w, len(out)


def predict_peaks(model, mol: Chem.Mol, mode: str, top: int = TOPN, with_info: bool = False):
    """学習したモデルで、フラグメントのピークを予測する。-> [(m/z, 相対強度 %)] 強度の高い順(m/z が同じものは最大)。"""
    M, frs = fm.enumerate_fragments(mol, depth=DEPTH)
    if not frs:
        return []
    glob, tc, tm = fm.global_features(M)
    ks = fm.KS[mode]
    feats = np.array([fm.frag_features(M, f, tc, tm, M.n, glob) for f in frs], dtype=np.float32)
    prec = tm + db_proton(mode) if mode != "ei" else tm - fm.E
    X = np.vstack([np.hstack([np.full((len(frs), 1), k, dtype=np.float32), feats]) for k in ks])
    pred = np.clip(model.predict(X), 0, None) ** 2
    mz = np.concatenate([[fm.ion_mz(f.mass, k, mode) for f in frs] for k in ks])
    fidx = np.tile(np.arange(len(frs)), len(ks))
    kk = np.repeat(np.array(ks), len(frs))
    ok = (mz > 12) & (mz < prec + 0.5)
    order = np.argsort(-pred)
    tol = 0.3 if mode == "ei" else 0.003
    taken: list[float] = []
    peaks = []
    for i in order:
        if not ok[i]:
            continue
        if any(abs(mz[i] - t) < tol for t in taken):
            continue
        taken.append(mz[i])
        peaks.append((float(mz[i]), float(pred[i]), int(fidx[i]), int(kk[i])))
        if len(peaks) >= top:
            break
    if not peaks:
        return []
    top_v = peaks[0][1] or 1.0
    res = [(m, 100.0 * v / top_v) for m, v, _, _ in peaks]
    return (res, [(frs[fi], k) for _, _, fi, k in peaks]) if with_info else res


_W = {}


def _winit(model_path: str, group: str):
    import joblib
    _W["model"] = joblib.load(model_path)
    _W["group"] = group


def _eval_one(args):
    """1 化合物を、学習した予測と規則の両方で評価する(並列に動く)。"""
    smiles, arr = args
    mol = Chem.MolFromSmiles(smiles)
    group = _W["group"]
    recs = [db.Record("", "", "", "", "", 0.0, "", "", "", "", "", "", [(float(p), float(r)) for p, r in zip(e, rl)], decimals=3 if tol < 0.1 else 0)
            for e, rl, tol in arr]
    pred = predict_peaks(_W["model"], mol, MODE[group])
    try:
        rules, _, _ = bench.fr.predict(mol, bench.GROUPS[group], engine="rules")
    except Exception:
        return None
    rule_pred = sorted([(p.mz, p.rel) for p in rules if p.kind != "precursor"], key=lambda p: -p[1])[:TOPN]
    if not pred or not rule_pred:
        return None
    a = [db.compare(pred, r) for r in recs]
    b = [db.compare(rule_pred, r) for r in recs]
    return (st.mean(m.explained for m in a), st.mean(m.precision for m in a), st.mean(m.explained for m in b), st.mean(m.precision for m in b))


def evaluate(group: str, model, n: int, procs: int = 12):
    """検証用の化合物で、学習した予測と今の規則を比べる。化合物ごとに並列(モデルは一度保存して各プロセスが読む)。"""
    import tempfile
    data = bench.load_compounds(n, "test")[group]
    jobs = [(Chem.MolToSmiles(mol), _recs_arrays(recs)) for _, mol, recs in data]
    jobs = [j for j in jobs if j[1]]
    with tempfile.TemporaryDirectory(dir=str(MODELS.parent)) as td:
        path = str(Path(td) / "m.joblib")
        joblib.dump(model, path)
        with mp.Pool(procs, initializer=_winit, initargs=(path, group)) as pool:
            res = [r for r in pool.imap_unordered(_eval_one, jobs, chunksize=4) if r]
    fe, fp, re_, rp = (st.mean(x) for x in zip(*res))
    f1 = lambda e, p: 2 * e * p / (e + p) if e + p else 0.0
    return {"n": len(res), "ml_explained": fe, "ml_precision": fp, "ml_f1": f1(fe, fp),
            "rule_explained": re_, "rule_precision": rp, "rule_f1": f1(re_, rp)}


class TorchMLP:
    """GPU で学習するニューラルネット(sklearn と同じ fit / predict の口)。NaN は 0 にして、NaN かどうかの印を足す。"""

    def __init__(self, hidden=(512, 256, 128), epochs=40, lr=2e-3, batch=4096, seed=0, device=None):
        self.hidden, self.epochs, self.lr, self.batch, self.seed = hidden, epochs, lr, batch, seed
        self.device = device
        self.n_iter_ = epochs

    def _prep(self, X):
        X = np.asarray(X, dtype=np.float32)
        nan = np.isnan(X)
        Z = np.where(nan, 0.0, X)
        Z = (Z - self.mu) / self.sd
        return np.hstack([Z, nan[:, self.nan_cols].astype(np.float32)])

    def fit(self, X, y, sample_weight=None):
        import torch
        torch.manual_seed(self.seed)
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        X = np.asarray(X, dtype=np.float32)
        nan = np.isnan(X)
        self.nan_cols = np.nonzero(nan.any(axis=0))[0]
        Z = np.where(nan, 0.0, X)
        self.mu = Z.mean(0); self.sd = Z.std(0) + 1e-6
        Xt = torch.tensor(self._prep(X), device=self.device)
        yt = torch.tensor(y, dtype=torch.float32, device=self.device)
        wt = torch.tensor(sample_weight if sample_weight is not None else np.ones(len(y)), dtype=torch.float32, device=self.device)
        dims = [Xt.shape[1], *self.hidden]
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [torch.nn.Linear(a, b), torch.nn.BatchNorm1d(b), torch.nn.ReLU(), torch.nn.Dropout(0.1)]
        layers.append(torch.nn.Linear(dims[-1], 1))
        self.net = torch.nn.Sequential(*layers).to(self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=self.epochs * ((len(y) + self.batch - 1) // self.batch))
        n = len(y)
        for ep in range(self.epochs):
            self.net.train()
            perm = torch.randperm(n, device=self.device)
            tot = 0.0
            for i in range(0, n, self.batch):
                idx = perm[i:i + self.batch]
                pred = self.net(Xt[idx]).squeeze(1)
                loss = (wt[idx] * (pred - yt[idx]) ** 2).sum() / wt[idx].sum()
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
                tot += float(loss) * len(idx)
        return self

    def predict(self, X):
        import torch
        self.net.eval()
        with torch.no_grad():
            Xt = torch.tensor(self._prep(X), device=self.device)
            return np.concatenate([self.net(Xt[i:i + 65536]).squeeze(1).cpu().numpy() for i in range(0, len(Xt), 65536)])


def train(group: str, ntrain: int, procs: int = 10, kind: str = "gbm", **hp):
    from sklearn.ensemble import HistGradientBoostingRegressor
    t0 = time.time()
    X, y, w, nc = build_train(group, ntrain, procs)
    print(f"[{group}] 学習データ: 化合物 {nc}、行 {len(y):,}、特徴量 {X.shape[1]}、正例 {int((y > 0).sum()):,}  ({time.time() - t0:.0f} 秒)", flush=True)
    t1 = time.time()
    if kind == "mlp":
        model = TorchMLP(**hp)
        model.fit(X, y, sample_weight=w)
    else:
        params = dict(max_iter=400, learning_rate=0.06, max_leaf_nodes=63, min_samples_leaf=40, l2_regularization=1.0,
                      early_stopping=True, validation_fraction=0.1, n_iter_no_change=25, random_state=0)
        params.update(hp)
        model = HistGradientBoostingRegressor(**params)
        model.fit(X, y, sample_weight=w)
    print(f"[{group}] 学習 {time.time() - t1:.0f} 秒、反復 {model.n_iter_}", flush=True)
    return model


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="ESI+", choices=list(MODE))
    ap.add_argument("--ntrain", type=int, default=1500)
    ap.add_argument("--ntest", type=int, default=300)
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--kind", default="gbm", choices=("gbm", "mlp"))
    ap.add_argument("--rneg", type=float, default=0.06)
    ap.add_argument("--eval-only", action="store_true", help="保存済みのモデルを検証用の化合物で評価するだけ")
    ap.add_argument("--hp", nargs="*", default=[], help="key=value(モデルのハイパーパラメータ)")
    a = ap.parse_args()
    hp = {k: (int(v) if v.lstrip("-").isdigit() else float(v)) for k, v in (x.split("=") for x in a.hp)}
    build_train.__defaults__ = (10, a.rneg)
    if a.eval_only:
        model = joblib.load(MODELS / f"{a.group}.joblib")
    else:
        model = train(a.group, a.ntrain, a.procs, kind=a.kind, **hp)
    r = evaluate(a.group, model, a.ntest)
    print(f"[{a.group}] 検証用 n={r['n']}")
    print(f"   学習した予測: 説明率 {r['ml_explained']:.0%}  適合率 {r['ml_precision']:.0%}  F1 {r['ml_f1']:.3f}")
    print(f"   今の規則    : 説明率 {r['rule_explained']:.0%}  適合率 {r['rule_precision']:.0%}  F1 {r['rule_f1']:.3f}")
    if a.save:
        MODELS.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODELS / f"{a.group}.joblib")
        print("保存:", MODELS / f"{a.group}.joblib")
