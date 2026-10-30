"""name2struct2: IUPAC 名 ⇄ 構造(完全オフライン)

  引数なし                                 GUI を開く
  name2struct.py "ethanol"                 名前 → 構造(入力が SMILES / InChI なら 構造 → 名前。自動判定)
  name2struct.py "CCO" --to name           方向を指定(name / struct)
  name2struct.py "DMF" --mol a.mol --png a.png --locants   構造を保存(--locants で IUPAC 位置番号を描く)
  name2struct.py "2-acetoxy benzoic acid" --rename         名前 → 構造のあと、構造から付け直した名前も出す
  name2struct.py --file list.csv -o out.xlsx                一括変換
       入力: .csv .txt(1 列目)/ .xlsx(1 列目)/ .sdf .mol .cdxml .smi(構造 → 名前)
       出力: .csv / .xlsx / .sdf(拡張子で決める。既定は 入力名_out.csv)
  name2struct.py "CCO" --json              結果を JSON で
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    if _s is not None and hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8")

import n2s_core as core

FIELDS = ["input", "direction", "iupac_name", "check", "smiles", "inchi", "inchikey", "formula", "mw",
          "exact_mass", "aliases", "message"]


# ---------------------------------------------------------------- 一括変換

def read_inputs(src: Path) -> list[tuple[str, object]]:
    """[(表示用の入力, 文字列 or Mol)]"""
    suf = src.suffix.lower()
    if suf in (".sdf", ".sd", ".mol", ".cdxml", ".smi"):
        out = []
        for title, mol in core.read_file(src):
            out.append((title or core.Chem.MolToSmiles(mol), mol))
        return out
    if suf in (".xlsx", ".xlsm"):
        from openpyxl import load_workbook
        ws = load_workbook(src, read_only=True).active
        vals = [str(r[0]).strip() for r in ws.iter_rows(values_only=True) if r and r[0] is not None]
    else:
        with open(src, newline="", encoding="utf-8-sig") as f:
            vals = [r[0].strip() for r in csv.reader(f) if r and r[0].strip()]
    if vals and vals[0].lower() in ("input", "smiles", "name", "iupac", "名前", "入力"):
        vals = vals[1:]
    return [(v, v) for v in vals]


def convert_item(item, to: str) -> core.Result:
    if isinstance(item, core.Chem.Mol):
        return core.smiles_to_name(item)
    return core.convert(item, to)


def write_results(results: list[core.Result], dst: Path):
    suf = dst.suffix.lower()
    rows = [r.row() for r in results]
    if suf == ".xlsx":
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill
        wb = Workbook()
        ws = wb.active
        ws.title = "name2struct2"
        ws.append(FIELDS)
        for c in ws[1]:
            c.font = Font(bold=True)
        colors = {"一致": "E2F0D9", "立体以外一致": "FFF2CC", "未検証": "F8CBAD", "推定": "FFF2CC", "失敗": "F8CBAD"}
        for row in rows:
            ws.append([row.get(f, "") for f in FIELDS])
            fill = colors.get(row.get("check", ""))
            if fill:
                ws.cell(ws.max_row, FIELDS.index("check") + 1).fill = PatternFill("solid", fgColor=fill)
        for col, w in zip("ABCDEFGHIJKL", (28, 10, 50, 10, 40, 40, 30, 14, 10, 12, 20, 40)):
            ws.column_dimensions[col].width = w
        ws.freeze_panes = "A2"
        wb.save(dst)
    elif suf in (".sdf", ".sd"):
        from rdkit.Chem import AllChem
        w = core.Chem.SDWriter(str(dst))
        for r, row in zip(results, rows):
            if r.mol is None:
                continue
            m = core.Chem.Mol(r.mol)
            AllChem.Compute2DCoords(m)
            m.SetProp("_Name", r.name or r.query)
            for f in FIELDS:
                if row.get(f) not in (None, ""):
                    m.SetProp(f, str(row[f]))
            w.write(m)
        w.close()
    else:
        with open(dst, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)


def run_batch(src: Path, dst: Path, to: str = "auto", progress=None) -> Path:
    items = read_inputs(src)
    results = []
    for i, (label, item) in enumerate(items, 1):
        try:
            r = convert_item(item, to)
        except Exception as e:  # 1 件の失敗で全体を止めない
            r = core.Result(label, "", level="失敗", message=f"エラー: {e}")
        r.query = label
        results.append(r)
        msg = f"[{i}/{len(items)}] {label} -> {r.name or r.info.get('smiles') or '失敗'} ({r.level})"
        print(msg, flush=True)
        if progress:
            progress(i, len(items), msg)
    write_results(results, dst)
    print(f"保存: {dst}")
    return dst


# ---------------------------------------------------------------- 1 件

def show(r: core.Result):
    print(f"方向     : {r.direction}")
    print(f"IUPAC 名 : {r.name}")
    if r.aliases:
        print(f"別名     : {' / '.join(r.aliases)}")
    for k, label in (("smiles", "SMILES"), ("inchi", "InChI"), ("inchikey", "InChIKey")):
        print(f"{label:9s}: {r.info.get(k, '')}")
    if r.info.get("formula"):
        print(f"分子式   : {r.info['formula']}  MW {r.info['mw']}  精密質量 {r.info['exact_mass']}")
    print(f"判定     : {r.level}  {r.message}")
    for name, lv in r.candidates[1:6]:
        print(f"  候補: {name}  [{lv}]")


def main():
    ap = argparse.ArgumentParser(description="IUPAC 名 ⇄ 構造(オフライン)",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("text", nargs="?", help="名前 / SMILES / InChI")
    ap.add_argument("--to", choices=["auto", "name", "struct"], default="auto")
    ap.add_argument("--file", "--csv", dest="file", type=Path, help="一括変換する入力ファイル")
    ap.add_argument("-o", "--out", type=Path, help="一括変換の出力(.csv/.xlsx/.sdf)")
    ap.add_argument("--mol", type=Path, help="構造を MOL で保存")
    ap.add_argument("--png", type=Path, help="構造図を PNG で保存")
    ap.add_argument("--svg", type=Path, help="構造図を SVG で保存")
    ap.add_argument("--locants", action="store_true", help="構造図に IUPAC の位置番号を描く")
    ap.add_argument("--rename", action="store_true", help="名前 → 構造のあと、構造から検証済みの名前を付け直す")
    ap.add_argument("--json", action="store_true", help="結果を JSON で出す")
    a = ap.parse_args()

    if a.file:
        run_batch(a.file, a.out or a.file.with_name(a.file.stem + "_out.csv"), a.to)
        return
    if not a.text:
        import n2s_gui
        n2s_gui.main()
        return

    r = core.convert(a.text, a.to)
    if a.rename and r.direction == "名前→構造" and r.mol is not None:
        r2 = core.smiles_to_name(r.mol)
        if r2.ok:
            r.message += f"  構造から付けた名前: {r2.name}({r2.level})"
    if a.json:
        print(json.dumps({**r.row(), "candidates": r.candidates}, ensure_ascii=False, indent=2))
    else:
        show(r)
    if r.mol is not None:
        loc = r.locants if a.locants else None
        if a.mol:
            core.save_mol(r.mol, a.mol, r.name)
            print(f"保存: {a.mol}")
        if a.png:
            a.png.write_bytes(core.mol_to_png(r.mol, (800, 600), loc))
            print(f"保存: {a.png}")
        if a.svg:
            a.svg.write_text(core.mol_to_png(r.mol, (800, 600), loc, svg=True), encoding="utf-8")
            print(f"保存: {a.svg}")
    sys.exit(0 if r.level not in ("失敗", "未検証") else 1)


if __name__ == "__main__":
    main()
