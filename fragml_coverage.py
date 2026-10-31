"""候補生成器(n2s_fragml)が、実測のピークをどれだけ含められるか(上限)を測る。"""
import sys, statistics as st, time
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import msdb_bench as bench
import n2s_fragml as fm

MODE = {"ESI+": "pos", "ESI-": "neg", "EI": "ei"}
n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
DEPTH = int(sys.argv[2]) if len(sys.argv) > 2 else 2
data = bench.load_compounds(n, "test")
for g, comps in data.items():
    mode, cov_major, cov_top5, ncand, t0 = MODE[g], [], [], [], time.time()
    shifts = {k: [] for k in fm.KS[mode]}
    for ik, mol, recs in comps:
        M, frs = fm.enumerate_fragments(mol, depth=DEPTH)
        by_k = {k: np.array([fm.ion_mz(f.mass, k, mode) for f in frs]) for k in fm.KS[mode]}
        allmz = np.sort(np.concatenate(list(by_k.values())))
        ncand.append(len(allmz))
        for rec in recs:
            tol = bench.db.tolerance(rec)
            exp = rec.peaks
            def hit(mz, arr=allmz):
                i = np.searchsorted(arr, mz - tol); return i < len(arr) and arr[i] <= mz + tol
            major = [(mz, r) for mz, r in exp if r >= 5.0 and mz < max(rec.precursor_mz, 1) - 0.5 or (mz, r) in [] ]
            major = [(mz, r) for mz, r in exp if r >= 5.0]
            if major:
                cov_major.append(sum(r for mz, r in major if hit(mz)) / sum(r for _, r in major))
            top = sorted(exp, key=lambda p: -p[1])[:5]
            cov_top5.append(sum(hit(mz) for mz, _ in top) / len(top))
            for k, arr in by_k.items():
                sa = np.sort(arr)
                shifts[k].append(sum(r for mz, r in major if (lambda i: i < len(sa) and sa[i] <= mz + tol)(np.searchsorted(sa, mz - tol))) / max(sum(r for _, r in major), 1e-9)) if major else None
    print(f"{g:5s} 化合物 {len(comps)}  候補(m/z)の数 中央値 {int(st.median(ncand))}  "
          f"主要ピーク(≥5%)の強度のうち候補に含まれる割合 平均 {st.mean(cov_major):.0%}   上位 5 ピークのうち {st.mean(cov_top5):.0%}   ({time.time()-t0:.0f} 秒)")
    print("      k ごとの寄与(その k だけで含められる割合): " + "  ".join(f"k={k:+d} {st.mean(x for x in v if x is not None):.0%}" for k, v in shifts.items()))
