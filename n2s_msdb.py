"""実測の MS/MS スペクトルのデータベース(MassBank)を、ローカルで引いて、予測したフラグメントを検証する。

  python n2s_msdb.py --build [msp のパス]   MassBank_NISTformat.msp から SQLite の索引 data/massbank/massbank.sqlite を作る
  python n2s_msdb.py --download             最新版の MassBank_NISTformat.msp を GitHub の公式リリースから取ってくる(約 140MB。ネットワークを使う)
  lookup(mol) / compare(予測, 実測)         GUI・評価から使う

MassBank(https://github.com/MassBank/MassBank-data)は CC BY などのオープンデータ。レコードごとにライセンスが付く。
再配布はしない(索引はローカルに作る。公開版のリポジトリには含めない)。引用: MassBank Consortium and its contributors.
実測は装置・衝突エネルギー・イオン化法でばらつくので、「予測と合うか」の目安として使う。
"""
from __future__ import annotations

import sqlite3
import statistics
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from rdkit import Chem

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "massbank"
MSP = DATA / "MassBank_NISTformat.msp"
DB = DATA / "massbank.sqlite"
URL = "https://github.com/MassBank/MassBank-data/releases/latest/download/MassBank_NISTformat.msp"


@dataclass
class Record:
    acc: str
    name: str
    ikey: str
    smiles: str
    precursor_type: str
    precursor_mz: float
    ion_mode: str
    ce: str
    instrument_type: str
    instrument: str
    spectrum_type: str
    formula: str
    peaks: list[tuple[float, float]]      # (m/z, 強度 %) 最大 = 100

    @property
    def hires(self) -> bool:
        """小数 3 桁以上のピークが多ければ高分解能として扱う。"""
        return self.decimals >= 3

    decimals: int = 0

    def title(self) -> str:
        ce = f"  CE {self.ce}" if self.ce else ""
        return f"{self.precursor_type or '?'}  {self.instrument_type or ''}{ce}  ({len(self.peaks)} ピーク)  {self.acc}"


def available() -> bool:
    return DB.exists()


def _parse_msp(path: Path):
    rec: dict[str, str] = {}
    peaks: list[str] = []
    in_peaks = False
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if rec:
                    yield rec, peaks
                rec, peaks, in_peaks = {}, [], False
                continue
            if in_peaks:
                peaks.append(line)
                continue
            if ": " in line:
                k, v = line.split(": ", 1)
                if k == "Num Peaks":
                    in_peaks = True
                rec.setdefault(k, v)
            elif line.startswith("Num Peaks"):
                in_peaks = True
        if rec:
            yield rec, peaks


def build(msp: Path = MSP, db: Path = DB, progress=None) -> int:
    """MassBank の NIST 形式から SQLite の索引を作る(MS2 だけ)。-> 登録したスペクトル数。"""
    db.parent.mkdir(parents=True, exist_ok=True)
    tmp = db.with_suffix(".tmp")
    if tmp.exists():
        tmp.unlink()
    con = sqlite3.connect(tmp)
    try:
        n = _fill(con, msp, progress)
        con.execute("CREATE INDEX idx_ik14 ON spectra(ik14)")
        con.commit()
    except BaseException:
        con.close()
        tmp.unlink(missing_ok=True)
        raise
    con.close()
    tmp.replace(db)
    return n


def _fill(con, msp: Path, progress) -> int:
    con.execute("""CREATE TABLE spectra(acc TEXT PRIMARY KEY, name TEXT, ikey TEXT, ik14 TEXT, smiles TEXT, ptype TEXT,
                   pmz REAL, mode TEXT, ce TEXT, itype TEXT, instrument TEXT, stype TEXT, formula TEXT, decimals INTEGER, peaks TEXT)""")
    n = 0
    rows = []
    for rec, plines in _parse_msp(msp):
        if rec.get("Spectrum_type", "MS2") not in ("MS2", "MS"):
            continue
        ik = rec.get("InChIKey", "")
        if len(ik) < 14 or not plines:
            continue
        pk = []
        dec = []
        for ln in plines:
            parts = ln.split()
            if len(parts) >= 2:
                try:
                    pk.append((float(parts[0]), float(parts[1])))
                    dec.append(len(parts[0].split(".")[1]) if "." in parts[0] else 0)
                except ValueError:
                    pass
        if not pk:
            continue
        top = max(i for _, i in pk) or 1.0
        text = ";".join(f"{m:g}:{100 * i / top:.1f}" for m, i in pk)
        try:
            pmz = float(rec.get("PrecursorMZ", "0") or 0)
        except ValueError:
            pmz = 0.0
        rows.append((rec.get("DB#", ""), rec.get("Name", ""), ik, ik[:14], rec.get("SMILES", ""), rec.get("Precursor_type", ""),
                     pmz, rec.get("Ion_mode", ""), rec.get("Collision_energy", ""), rec.get("Instrument_type", ""),
                     rec.get("Instrument", ""), rec.get("Spectrum_type", ""), rec.get("Formula", ""),
                     int(statistics.median(dec)) if dec else 0, text))
        n += 1
        if len(rows) >= 5000:
            con.executemany("INSERT OR REPLACE INTO spectra VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
            rows = []
            if progress:
                progress(n)
    if rows:
        con.executemany("INSERT OR REPLACE INTO spectra VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    return n


def download(dst: Path = MSP, progress=None) -> Path:
    """MassBank-data の公式リリースから NIST 形式をダウンロードする(ネットワークを使う)。"""
    dst.parent.mkdir(parents=True, exist_ok=True)

    def hook(blocks, size, total):
        if progress and total > 0:
            progress(min(blocks * size / total, 1.0))
    # 途中で切れたファイルを完成品と取り違えないように(MSP.exists() で判定している)、別名で取ってから置き換える
    tmp = dst.with_name(dst.name + ".part")
    try:
        urllib.request.urlretrieve(URL, tmp, hook)
        tmp.replace(dst)
    finally:
        if tmp.exists():
            tmp.unlink()
    return dst


def _row(r) -> Record:
    peaks = [(float(a), float(b)) for a, b in (p.split(":") for p in r[14].split(";"))]
    return Record(r[0], r[1], r[2], r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[11], r[12], peaks, decimals=r[13])


_SELECT = "SELECT acc,name,ikey,ik14,smiles,ptype,pmz,mode,ce,itype,instrument,stype,formula,decimals,peaks FROM spectra"


def lookup(mol: Chem.Mol, ion: str | None = None) -> list[Record]:
    """InChIKey の前半 14 文字(骨格)が同じスペクトルを返す。ion を渡すと、前駆イオンの種類が合うものを先に。"""
    if not available() or mol is None:
        return []
    try:
        ik = Chem.MolToInchiKey(mol)
    except Exception:
        return []
    if not ik:
        return []
    con = sqlite3.connect(DB)
    try:
        rows = con.execute(_SELECT + " WHERE ik14=?", (ik[:14],)).fetchall()
    finally:
        con.close()
    recs = [_row(r) for r in rows]
    want = {"M+": ("[M]+", "[M]+•", "[M]+*", "M+"), "[M+H]+": ("[M+H]+",), "[M+Na]+": ("[M+Na]+",), "[M-H]-": ("[M-H]-",)}.get(ion or "", ())

    def rank(r: Record):
        hit = r.precursor_type in want or (ion == "M+" and "EI" in r.instrument_type)
        return (0 if hit else 1, -len(r.peaks), r.precursor_type, r.ce)   # ピークの多い(情報の多い)スペクトルを先に
    return sorted(recs, key=rank)


def stats() -> dict:
    if not available():
        return {}
    con = sqlite3.connect(DB)
    try:
        return {"spectra": con.execute("SELECT COUNT(*) FROM spectra").fetchone()[0],
                "compounds": con.execute("SELECT COUNT(DISTINCT ik14) FROM spectra").fetchone()[0]}
    finally:
        con.close()


def tolerance(rec: Record) -> float:
    return 0.01 if rec.hires else 0.5


@dataclass
class Match:
    tol: float
    explained: float          # 実測の主要ピーク(強度 ≥ 5%)の強度のうち、予測のどれかと一致したものの割合
    top5: float               # 実測の上位 5 ピークのうち、予測と一致した割合
    precision: float          # 予測の主要ピーク(強度 ≥ 10%)のうち、実測(強度 ≥ 1%)にあるものの割合
    pred_hit: list[float | None]   # 予測ピークごとに、一致した実測ピークの強度(一致しなければ None)
    exp_hit: list[bool]            # 実測ピークごとに、予測と一致したか


def compare(pred: list[tuple[float, float]], rec: Record, tol: float | None = None) -> Match:
    tol = tol or tolerance(rec)
    exp = rec.peaks
    pred_hit: list[float | None] = []
    for mz, rel in pred:
        best = None
        for em, er in exp:
            if abs(em - mz) <= tol and er >= 1.0 and (best is None or er > best):
                best = er
        pred_hit.append(best)
    exp_hit = [any(abs(em - mz) <= tol for mz, _ in pred) for em, _ in exp]
    major = [(i, er) for i, (_, er) in enumerate(exp) if er >= 5.0]
    explained = (sum(er for i, er in major if exp_hit[i]) / sum(er for _, er in major)) if major else 0.0
    top = sorted(range(len(exp)), key=lambda i: -exp[i][1])[:5]
    top5 = sum(exp_hit[i] for i in top) / len(top) if top else 0.0
    pm = [i for i, (_, rel) in enumerate(pred) if rel >= 10.0]
    precision = (sum(pred_hit[i] is not None for i in pm) / len(pm)) if pm else 0.0
    return Match(tol, explained, top5, precision, pred_hit, exp_hit)


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--download" in args:
        print("ダウンロード中…", MSP)
        download(progress=lambda f: print(f"\r{f:.0%}", end="", flush=True))
        print("\n完了")
    if "--build" in args:
        src = Path(args[args.index("--build") + 1]) if len(args) > args.index("--build") + 1 and not args[args.index("--build") + 1].startswith("--") else MSP
        print("索引を作っています…", src)
        n = build(src, progress=lambda n: print(f"\r{n} 件", end="", flush=True))
        print(f"\n登録: {n} 件 → {DB}")
        print(stats())
