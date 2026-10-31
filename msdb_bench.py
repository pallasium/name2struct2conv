"""MassBank の実測 MS/MS・EI スペクトルと、n2s_frag の予測を、数百の化合物で照合して、予測の質を測る。

  python msdb_bench.py                 現在の重み(n2s_frag.W)で、検証用の化合物を測る
  python msdb_bench.py --ablate        第二段・環開裂・各規則を止めたときの変化も見る
  python msdb_bench.py --tune          調整用の化合物で重みを調整(座標降下)し、検証用で確かめる

群: ESI+ = [M+H]+(高分解能)/ ESI- = [M-H]-(高分解能)/ EI = M+(低分解能)。
化合物は InChIKey の骨格(最初の 14 文字)のハッシュで、調整用と検証用に分ける(同じ化合物が両方に入らない)。
指標(予測の上位 15 ピークだけを使う): 説明率 = 実測の主要ピーク(≥5%)の強度のうち、予測と一致したものの割合 /
適合率 = 予測の 15 ピークのうち実測(≥1%)にあるものの割合 / F1 = その調和平均。ランダムに m/z を置いた場合(基準)と比べる。
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import random
import sqlite3
import statistics as st
import sys
import time

from rdkit import Chem, RDLogger

import n2s_frag as fr
import n2s_msdb as db

RDLogger.DisableLog("rdApp.*")
TOPN = 15
ENGINE = "auto"      # "auto"(学習したモデルがあれば使う)/ "rules"(規則だけ)
GROUPS = {"ESI+": "[M+H]+", "ESI-": "[M-H]-", "EI": "M+"}


def _h(ik14: str) -> int:
    return int(hashlib.md5(ik14.encode()).hexdigest(), 16)


def load_compounds(per_group: int, side: str, seed: int = 0):
    """-> {群: [(ik14, mol, [Record, ...])]}。side = 'train' | 'test'。"""
    con = db.sqlite3.connect(db.DB)
    out = {}
    for g, ion in GROUPS.items():
        if g == "EI":
            where = "itype LIKE '%EI%'"
        else:
            where = f"ptype='{ion}' AND decimals>=3"
        rows = con.execute(db._SELECT + f" WHERE {where} AND smiles!='' AND smiles NOT LIKE '%.%'").fetchall()
        by: dict[str, list] = {}
        for r in rows:
            by.setdefault(r[3], []).append(r)
        keys = sorted(k for k in by if (_h(k) % 2 == 0) == (side == "train"))
        random.Random(seed).shuffle(keys)
        sel = []
        for k in keys:
            rs = by[k]
            mol = Chem.MolFromSmiles(rs[0][4])
            if mol is None or Chem.GetFormalCharge(mol) or mol.GetNumHeavyAtoms() > 45 or mol.GetNumHeavyAtoms() < 4:
                continue
            sel.append((k, mol, [db._row(r) for r in rs[:6]]))
            if len(sel) >= per_group:
                break
        out[g] = sel
    con.close()
    return out


def score_one(mol, recs, ion, rng):
    try:
        peaks, _, prec = fr.predict(mol, ion, engine=ENGINE)
    except Exception:
        return None
    pred = sorted([(p.mz, p.rel) for p in peaks if p.kind != "precursor"], key=lambda p: -p[1])[:TOPN]
    if not pred:
        return None
    res, base = [], []
    for rec in recs:
        m = db.compare(pred, rec)
        res.append((m.explained, m.precision))
        lo, hi = 14.0, max(prec, max(mz for mz, _ in rec.peaks))
        rnd = [(rng.uniform(lo, hi), 50.0) for _ in range(len(pred))]
        mb = db.compare(rnd, rec)
        base.append((mb.explained, mb.precision))
    return [st.mean(x) for x in zip(*res)], [st.mean(x) for x in zip(*base)]


def f1(a, b):
    return 2 * a * b / (a + b) if a + b else 0.0


def evaluate(data, weights=None, label=""):
    if weights:
        fr.W.update(weights)
    rng = random.Random(1)
    out = {}
    for g, comps in data.items():
        ion = GROUPS[g]
        e, p, be, bp = [], [], [], []
        for _, mol, recs in comps:
            r = score_one(mol, recs, ion, rng)
            if r:
                e.append(r[0][0]); p.append(r[0][1]); be.append(r[1][0]); bp.append(r[1][1])
        if e:
            out[g] = {"n": len(e), "explained": st.mean(e), "precision": st.mean(p), "f1": f1(st.mean(e), st.mean(p)),
                      "random_f1": f1(st.mean(be), st.mean(bp))}
    return out


def show(res, title):
    print(f"\n== {title}")
    for g, r in res.items():
        print(f"  {g:5s} n={r['n']:3d}  説明率 {r['explained']:.0%}  適合率 {r['precision']:.0%}  F1 {r['f1']:.3f}   (ランダム F1 {r['random_f1']:.3f})")
    if res:
        print(f"  平均 F1 {st.mean(r['f1'] for r in res.values()):.3f}")


def _worker(args):
    weights, data = args
    saved = dict(fr.W)
    try:
        r = evaluate(data, weights)
    finally:
        fr.W.clear(); fr.W.update(saved)
    return st.mean(v["f1"] for v in r.values()) if r else 0.0


def tune(train, test, procs: int = 8, passes: int = 2):
    base = dict(fr.W)
    cur = dict(base)
    best = _worker((cur, train))
    print(f"調整前 F1(調整用) {best:.4f}")
    keys = list(base)
    levels = (0.0, 0.5, 0.75, 1.25, 1.6, 2.2)
    with mp.Pool(procs) as pool:
        for ps in range(passes):
            for k in keys:
                cands = [dict(cur, **{k: round(max(base[k] * lv if lv else 0.0, 0.0), 3)}) for lv in levels]
                cands = [c for c in cands if c[k] != cur[k]]
                scores = pool.map(_worker, [(c, train) for c in cands])
                i = max(range(len(scores)), key=lambda j: scores[j])
                if scores[i] > best + 0.002:           # 小さな差は採らない(過適合を避ける)
                    best, cur = scores[i], cands[i]
                    print(f"  pass {ps + 1} {k:9s} -> {cur[k]:<6}  F1 {best:.4f}", flush=True)
    return cur


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if not db.available():
        sys.exit("先に python n2s_msdb.py --build で索引を作ってください")
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 120
    if "--rules" in sys.argv:
        ENGINE = "rules"
    t0 = time.time()
    if "--tune" in sys.argv:
        train, test = load_compounds(max(40, n // 2), "train"), load_compounds(n, "test")
        show(evaluate(train, dict(fr.W)), "調整前(調整用)")
        before = evaluate(test, dict(fr.W))
        show(before, "調整前(検証用)")
        best = tune(train, test)
        print("\n調整後の重み:", json.dumps(best, ensure_ascii=False))
        fr.W.update(best)
        show(evaluate(train), "調整後(調整用)")
        after = evaluate(test)
        show(after, "調整後(検証用)")
    else:
        test = load_compounds(n, "test")
        show(evaluate(test), "現在の重み(検証用)")
        if "--htrans" in sys.argv:
            for v in (0.0, 0.3, 0.6, 1.0):
                saved = dict(fr.W)
                show(evaluate(test, dict(saved, htrans=v)), f"水素移動イオンの倍率 {v}")
                fr.W.clear(); fr.W.update(saved)
        if "--ablate" in sys.argv:
            for name, w in (("第二段なし", {"gen2": 0.0}), ("環開裂なし", {"ring": 0.0}), ("中性脱離の点を半分", {"loss": 0.5}),
                            ("ベンジル位を無効", {"benzylic": 0.0}), ("アシリウムを無効", {"acylium": 0.0}),
                            ("α ヘテロ原子を無効", {"alpha": 0.0}), ("小さな断片の罰を無効", {"small": 0.0})):
                saved = dict(fr.W)
                show(evaluate(test, dict(saved, **w)), name)
                fr.W.clear(); fr.W.update(saved)
    print(f"\n所要 {time.time() - t0:.0f} 秒")
