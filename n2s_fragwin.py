"""フラグメント(予測)の別ウィンドウ: フラグメントのスペクトル・ピークリスト・構造(選んだピークの部分を強調)。

予測は n2s_frag の規則ベース。強度は目安で、実測ではない。
スペクトルと構造は、ホイールで拡大縮小・ドラッグで移動・ダブルクリックで元の大きさ(n2s_view)。
"""
from __future__ import annotations

import threading
import tkinter as tk
from tkinter import messagebox, ttk

from rdkit import Chem

import n2s_frag as frag
import n2s_ms as ms
import n2s_msdb as msdb
from n2s_view import SpectrumView, StructureView

_KIND = {"precursor": "前駆イオン", "cleavage": "結合開裂", "loss": "中性脱離", "ring": "環開裂"}


def kind_text(p) -> str:
    return _KIND.get(p.kind, p.kind) + ("(2段)" if getattr(p, "gen", 1) == 2 else "")
_IONS = ("M+", "[M+H]+", "[M+Na]+", "[M-H]-")


class FragmentWindow(tk.Toplevel):
    def __init__(self, root: tk.Misc, mol: Chem.Mol, name: str, ion: str, locants: dict | None = None):
        super().__init__(root)
        self.mol = ms.pick_fragment(mol, ion)
        self.name = name
        self.locants = None
        self.peaks: list[frag.Peak] = []
        self.title(f"フラグメント(予測): {name}")
        self.geometry("1180x700")
        self.minsize(800, 480)

        bar = ttk.Frame(self, padding=(10, 8, 10, 2))
        bar.pack(fill="x")
        ttk.Label(bar, text="イオン").pack(side="left")
        self.ion = tk.StringVar(value=ion if ion in _IONS else "M+")
        box = ttk.Combobox(bar, textvariable=self.ion, values=_IONS, width=9, state="readonly")
        box.pack(side="left", padx=(6, 12))
        box.bind("<<ComboboxSelected>>", lambda e: self.compute())
        ttk.Label(bar, text="予測方式").pack(side="left")
        self.engine = tk.StringVar(value="学習(MassBank)")
        eng = ttk.Combobox(bar, textvariable=self.engine, values=("学習(MassBank)", "学習(勾配ブースティング)", "規則"), width=22, state="readonly")
        eng.pack(side="left", padx=(6, 12))
        eng.bind("<<ComboboxSelected>>", lambda e: self.compute())
        ttk.Button(bar, text="ピークリストをコピー", command=self.copy_list).pack(side="left")
        self.note = ttk.Label(bar, text="", foreground="#7a4b00", wraplength=700)
        self.note.pack(side="left", padx=12)

        dbf = ttk.Frame(self, padding=(10, 0, 10, 2))
        dbf.pack(fill="x")
        ttk.Label(dbf, text="データベース(MassBank)").pack(side="left")
        self.rec_var = tk.StringVar()
        self.rec_box = ttk.Combobox(dbf, textvariable=self.rec_var, state="readonly", width=64)
        self.rec_box.pack(side="left", padx=6)
        self.rec_box.bind("<<ComboboxSelected>>", lambda e: self.show_record())
        self.db_btn = ttk.Button(dbf, text="データベースを準備…", command=self.prepare_db)
        self.db_note = ttk.Label(dbf, text="", foreground="#444", wraplength=620)
        self.db_note.pack(side="left", padx=6)
        self.records: list = []
        self.obs: list = []        # 予測ピークごとの、実測との一致(強度 % か None)

        pan = ttk.Panedwindow(self, orient="horizontal")
        pan.pack(fill="both", expand=True, padx=10, pady=6)

        left = ttk.Panedwindow(pan, orient="vertical")
        top = ttk.Frame(left)
        ttk.Label(top, text="フラグメントピーク(ホイール: 拡大縮小 / ドラッグ: 移動 / ダブルクリック: 元に戻す)").pack(anchor="w")
        self.spec = SpectrumView(top, on_select=self.pick_peak, height=240)
        self.spec.pack(fill="both", expand=True)
        left.add(top, weight=3)

        lst = ttk.Frame(left)
        cols = ("mz", "rel", "obs", "kind", "formula", "label")
        self.tree = ttk.Treeview(lst, columns=cols, show="headings", height=10, selectmode="browse")
        for c, t, w in (("mz", "m/z", 90), ("rel", "強度 %", 60), ("obs", "実測 %", 56), ("kind", "種類", 92),
                        ("formula", "組成式", 100), ("label", "帰属(脱離するもの)", 340)):
            self.tree.heading(c, text=t, command=lambda c=c: self.sort_by(c))
            self.tree.column(c, width=w, stretch=(c == "label"), anchor="e" if c in ("mz", "rel", "obs") else "w")
        sb = ttk.Scrollbar(lst, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="left", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self.pick_row)
        left.add(lst, weight=2)
        pan.add(left, weight=3)

        right = ttk.Frame(pan)
        ttk.Label(right, text="構造(選んだピークの部分を赤で強調)").pack(anchor="w")
        self.struct = StructureView(right)
        self.struct.pack(fill="both", expand=True)
        self.info = ttk.Label(right, text="", wraplength=380, font=("Yu Gothic UI", 10), foreground="#333", justify="left")
        self.info.pack(anchor="w", fill="x", pady=(4, 0))
        pan.add(right, weight=2)

        ttk.Label(self, text="予測方式「学習」は、MassBank の実測 MS/MS から結合の切れやすさを学習したモデル(使えないときは規則)。"
                             "強度は目安で、実測ではありません。",
                  foreground="#666").pack(anchor="w", padx=10, pady=(0, 6))
        self._sort = ("mz", False)
        self.compute()

    # ---------------------------------------------------------- 計算と表示
    def compute(self):
        try:
            peaks, note, prec = frag.predict(self.mol, self.ion.get(), engine={"規則": "rules", "学習(勾配ブースティング)": "gbm"}.get(self.engine.get(), "auto"))
        except ValueError as e:
            self.peaks = []
            self.note.config(text=str(e))
            self.tree.delete(*self.tree.get_children())
            self.spec.set_peaks([])
            self.struct.set_mol(self.mol)
            self.info.config(text="")
            self.rec_box.configure(values=[])
            self.rec_var.set("")
            return
        self.peaks = peaks
        self.note.config(text=note)
        self.spec.set_peaks([(p.mz, p.rel) for p in peaks], [p.kind for p in peaks])
        self.struct.set_mol(self.mol, None, None)
        self.obs = []
        self.fill()
        self.info.config(text=f"前駆イオン {self.ion.get()}  m/z {prec:.4f}")
        self.load_records()

    def fill(self):
        key, rev = self._sort
        idx = list(range(len(self.peaks)))
        getter = {"mz": lambda i: self.peaks[i].mz, "rel": lambda i: self.peaks[i].rel,
                  "obs": lambda i: (self.obs[i] if i < len(self.obs) and self.obs[i] is not None else -1),
                  "kind": lambda i: kind_text(self.peaks[i]), "formula": lambda i: self.peaks[i].formula,
                  "label": lambda i: self.peaks[i].label}[key]
        idx.sort(key=getter, reverse=rev)
        self.tree.delete(*self.tree.get_children())
        for i in idx:
            p = self.peaks[i]
            o = self.obs[i] if i < len(self.obs) else None
            self.tree.insert("", "end", iid=str(i),
                             values=(f"{p.mz:.4f}", f"{p.rel:.1f}", "" if o is None else f"{o:.0f}", kind_text(p), p.formula, p.label))

    def sort_by(self, col: str):
        key, rev = self._sort
        self._sort = (col, (not rev) if col == key else (col in ("rel",)))
        self.fill()

    # ---------------------------------------------------------- 選択の連動
    def pick_row(self, _e=None):
        sel = self.tree.selection()
        if sel:
            self.show_peak(int(sel[0]), from_tree=True)

    def pick_peak(self, index: int):
        iid = str(index)
        if self.tree.exists(iid):
            self.tree.selection_set(iid)
            self.tree.see(iid)
        self.show_peak(index, from_tree=False)

    def show_peak(self, i: int, from_tree: bool):
        p = self.peaks[i]
        self.spec.select(i, center=from_tree)
        self.struct.set_mol(self.mol, None, p.atoms or None, keep_view=True)
        lines = [f"m/z {p.mz:.4f}  強度 {p.rel:.1f}%  {kind_text(p)}", f"組成式 {p.formula}"]
        if p.smiles:
            lines.append(("フラグメント " if p.kind == "cleavage" else "") + p.smiles + ("  (* = 切れた位置)" if "*" in p.smiles else ""))
        lines.append(p.label)
        self.info.config(text="\n".join(lines))

    def copy_list(self):
        rows = ["m/z\t強度%\t種類\t組成式\t帰属"]
        for p in sorted(self.peaks, key=lambda p: p.mz):
            rows.append(f"{p.mz:.4f}\t{p.rel:.1f}\t{kind_text(p)}\t{p.formula}\t{p.label}")
        self.clipboard_clear()
        self.clipboard_append("\n".join(rows))
        self.note.config(text="ピークリストをクリップボードにコピーしました(タブ区切り)。")

    # ---------------------------------------------------------- データベース(MassBank)
    def load_records(self):
        self.spec.set_mirror([], None, "")
        self.records = []
        if not msdb.available():
            self.rec_box.configure(values=[], state="disabled")
            self.rec_var.set("")
            self.db_btn.pack(side="left", before=self.db_note)
            self.db_note.config(text="データベースが未準備です(実測の MS/MS で予測を検証できます。MassBank、約 140MB)。")
            return
        self.db_btn.pack_forget()
        self.records = msdb.lookup(self.mol, self.ion.get())
        if not self.records:
            self.rec_box.configure(values=[], state="disabled")
            self.rec_var.set("")
            self.db_note.config(text="MassBank に、骨格が同じ化合物のスペクトルはありません。")
            return
        self.rec_box.configure(values=[r.title() for r in self.records], state="readonly")
        self.rec_var.set(self.records[0].title())
        self.show_record()

    def show_record(self):
        i = self.rec_box.current()
        if i < 0 or i >= len(self.records) or not self.peaks:
            return
        rec = self.records[i]
        m = msdb.compare([(p.mz, p.rel) for p in self.peaks], rec)
        self.obs = m.pred_hit
        self.spec.set_mirror(rec.peaks, m.exp_hit, rec.title(), m.pred_hit)
        self.fill()
        k = min(5, len(rec.peaks))
        top = round(m.top5 * k)
        self.db_note.config(text=f"{len(self.records)} 件中 {i + 1} 件目: 実測の主要ピークの {m.explained:.0%} を予測が説明 / "
                                 f"実測の上位 {k} つのうち {top} つ / 予測の主要ピークの {m.precision:.0%} が実測にある"
                                 f"(許容 ±{m.tol} Da)。出典: MassBank {rec.acc}")

    def prepare_db(self):
        """索引を作る(MassBank_NISTformat.msp が無ければ、確認してから公式リリースをダウンロードする)。"""
        if not msdb.MSP.exists():
            if not messagebox.askyesno("データベースの準備", "MassBank(GitHub の公式リリース、約 140MB)をダウンロードして、"
                                       f"索引を作ります。\n保存先: {msdb.DATA}\n続けますか?", parent=self):
                return
        self.db_btn.state(["disabled"])

        def work():
            try:
                if not msdb.MSP.exists():
                    msdb.download(progress=lambda f: self.after(0, lambda: self.db_note.config(text=f"ダウンロード中… {f:.0%}")))
                msdb.build(progress=lambda n: self.after(0, lambda: self.db_note.config(text=f"索引を作っています… {n} 件")))
                self.after(0, self.load_records)
            except Exception as e:
                msg = f"失敗しました: {e}"   # e は except を抜けると消えるので、文字列にしてから渡す
                self.after(0, lambda: (self.db_note.config(text=msg), self.db_btn.state(["!disabled"])))
        threading.Thread(target=work, daemon=True).start()
