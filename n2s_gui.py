"""name2struct2 の GUI(tkinter)。name2struct.py を引数なしで実行すると開く。"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from rdkit import Chem

import n2s_core as core
import n2s_ms as ms
import n2s_props as props
from n2s_fragwin import FragmentWindow
from n2s_view import SpectrumView, StructureView
from name2struct import run_batch

LEVEL_COLORS = {"一致": ("#1e7b34", "#e2f0d9"), "OPSIN": ("#1e7b34", "#e2f0d9"), "辞書": ("#1e7b34", "#e2f0d9"),
                "利用者辞書": ("#1e7b34", "#e2f0d9"), "立体以外一致": ("#8a6100", "#fff2cc"), "互変異性体違い": ("#8a6100", "#fff2cc"),
                "推定": ("#8a6100", "#fff2cc"), "名前なし": ("#555", "#e7e6e6"), "未検証": ("#a12b1a", "#f8cbad"), "失敗": ("#a12b1a", "#f8cbad")}
LEVEL_TEXT = {"一致": "検証済み(OPSIN で同じ構造に戻る)", "OPSIN": "OPSIN で読んだ構造", "辞書": "辞書から",
              "利用者辞書": "利用者辞書から", "立体以外一致": "骨格は検証済み・立体は要確認", "互変異性体違い": "互変異性体が違う可能性(要確認)",
              "推定": "推定(要確認)", "名前なし": "名前は未計算(チェックで計算)", "未検証": "未検証(誤りの可能性が高い)", "失敗": "読めませんでした"}
USER_DICT_TEMPLATE = """# 利用者辞書: 別名(| で区切って複数可)<TAB>名前 または SMILES
# 例(先頭の # を消すと有効):
#化合物A|cmpd-A	4-methoxybenzoic acid
#化合物B	COc1ccc(cc1)C(=O)OC
"""


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.q: queue.Queue = queue.Queue()
        self.result: core.Result | None = None
        self.history: list[core.Result] = []
        self.busy = False
        root.title("name2struct2 — IUPAC 名 ⇄ 構造")
        root.geometry("1180x760")
        root.minsize(980, 640)
        style = ttk.Style()
        try:
            style.theme_use("vista")
        except tk.TclError:
            pass
        ui = ("Yu Gothic UI", 10)
        mono = ("Consolas", 11)
        style.configure(".", font=ui)
        style.configure("Big.TButton", font=("Yu Gothic UI", 11, "bold"))

        # ---- 入力
        top = ttk.Frame(root, padding=(10, 8, 10, 4))
        top.pack(fill="x")
        ttk.Label(top, text="名前 / SMILES / InChI / MOL(Enter で変換、Shift+Enter で改行)").pack(anchor="w")
        row = ttk.Frame(top)
        row.pack(fill="x")
        self.entry = tk.Text(row, height=2, font=mono, wrap="word", undo=True)
        self.entry.pack(side="left", fill="x", expand=True)
        self.entry.bind("<Return>", self._enter)
        self.entry.bind("<KP_Enter>", self._enter)
        self.btn = ttk.Button(row, text="変換", style="Big.TButton", command=self.run, width=8)
        self.btn.pack(side="left", padx=(6, 0), fill="y")

        mode = ttk.Frame(top)
        mode.pack(fill="x", pady=(4, 0))
        self.to = tk.StringVar(value="auto")
        for text, val in (("自動判定", "auto"), ("名前 → 構造", "struct"), ("構造 → 名前", "name")):
            ttk.Radiobutton(mode, text=text, value=val, variable=self.to).pack(side="left", padx=(0, 10))
        self.calc_name = tk.BooleanVar(value=True)     # 構造を入れたときに IUPAC 名を計算する(数秒〜十数秒かかる処理)
        ttk.Checkbutton(mode, text="構造 → 名前を計算する", variable=self.calc_name).pack(side="left", padx=(10, 0))
        self.calc_props = tk.BooleanVar(value=False)   # 化学物性(沸点・融点・logP など)を推算して別ウィンドウに出す
        ttk.Checkbutton(mode, text="物性を計算する", variable=self.calc_props, command=self._refresh_props).pack(side="left", padx=(10, 0))
        self.prop_win: tk.Toplevel | None = None
        self.opt_rename = tk.BooleanVar(value=False)   # CLI の --rename
        self.opt_json = tk.BooleanVar(value=False)     # CLI の --json
        self.opt_win: tk.Toplevel | None = None
        ttk.Button(mode, text="その他…", command=self.open_options).pack(side="right")
        ttk.Button(mode, text="利用者辞書…", command=self.open_user_dict).pack(side="right", padx=(0, 4))
        ttk.Button(mode, text="一括変換…", command=self.batch).pack(side="right", padx=4)
        ttk.Button(mode, text="ファイルを開く…", command=self.open_file).pack(side="right")

        # ---- 状態表示
        bottom = ttk.Frame(root, padding=(10, 2))
        bottom.pack(fill="x", side="bottom")
        self.status = ttk.Label(bottom, text="準備中…(OPSIN と翻訳モデルを読み込んでいます)")
        self.status.pack(side="left")
        self.pbar = ttk.Progressbar(bottom, length=220, mode="determinate")
        self.pbar.pack(side="right")

        # ---- 本体: 左 = 履歴、中 = 構造図、右 = 結果
        body = ttk.Panedwindow(root, orient="horizontal")
        body.pack(fill="both", expand=True, padx=10, pady=4)

        left = ttk.Frame(body)
        ttk.Label(left, text="履歴").pack(anchor="w")
        self.hist = tk.Listbox(left, font=("Consolas", 9), activestyle="none", width=24)
        self.hist.pack(fill="both", expand=True)
        self.hist.bind("<<ListboxSelect>>", self._pick_history)
        body.add(left, weight=1)

        mid = ttk.Frame(body)
        self.canvas = StructureView(mid)   # ホイールで拡大縮小 / ドラッグで移動 / ダブルクリックで元の大きさ
        self.canvas.pack(fill="both", expand=True)
        opts = ttk.Frame(mid)
        opts.pack(fill="x", pady=(4, 0))
        self.show_loc = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts, text="位置番号", variable=self.show_loc, command=lambda: self.redraw(True)).pack(side="left")
        for text, cmd in (("SVG", self.save_svg), ("PNG", self.save_png), ("SDF", self.save_sdf), ("MOL", self.save_mol)):
            ttk.Button(opts, text=text, command=cmd, width=5).pack(side="right", padx=1)
        ttk.Label(opts, text="保存:").pack(side="right")
        body.add(mid, weight=3)

        right = ttk.Frame(body, padding=(8, 0, 0, 0))
        self.badge = tk.Label(right, text="", font=("Yu Gothic UI", 10, "bold"), anchor="w", padx=8, pady=3)
        self.badge.pack(fill="x")
        self.msg = ttk.Label(right, text="", wraplength=440, foreground="#444")
        self.msg.pack(anchor="w", pady=(2, 6), fill="x")
        ttk.Label(right, text="IUPAC 名").pack(anchor="w")
        r = ttk.Frame(right)
        r.pack(fill="x", pady=(0, 3))
        self.name_box = tk.Text(r, height=3, font=("Consolas", 10), wrap="char", relief="solid", borderwidth=1,
                                background="#fafafa")
        ttk.Button(r, text="コピー", width=6,
                   command=lambda: self.copy(self.name_box.get("1.0", "end").strip())).pack(side="right", padx=(2, 0), anchor="n")
        self.name_box.pack(side="left", fill="x", expand=True)
        self.vars: dict[str, tk.StringVar] = {}
        def field(parent, key, label, **grid):
            """ラベル + 読み取り専用の欄 + コピーボタン。"""
            box = ttk.Frame(parent)
            box.grid(**grid)
            ttk.Label(box, text=label).pack(anchor="w")
            row = ttk.Frame(box)
            row.pack(fill="x", pady=(0, 3))
            v = tk.StringVar()
            ttk.Entry(row, textvariable=v, state="readonly", font=("Consolas", 10)).pack(side="left", fill="x", expand=True)
            ttk.Button(row, text="コピー", width=6, command=lambda v=v: self.copy(v.get())).pack(side="left", padx=(2, 0))
            self.vars[key] = v

        one = ttk.Frame(right)
        one.pack(fill="x")
        one.columnconfigure(0, weight=1)
        field(one, "smiles", "SMILES", row=0, column=0, sticky="ew")
        two = ttk.Frame(right)   # 分子式と分子量は横並び
        two.pack(fill="x")
        two.columnconfigure(0, weight=2)
        two.columnconfigure(1, weight=3)
        field(two, "formula", "分子式", row=0, column=0, sticky="ew", padx=(0, 6))
        field(two, "mass", "分子量 / 精密質量", row=0, column=1, sticky="ew")
        three = ttk.Frame(right)
        three.pack(fill="x")
        three.columnconfigure(0, weight=1)
        field(three, "ea", "元素分析(計算値 %)", row=0, column=0, sticky="ew")
        msr = ttk.Frame(right)
        msr.pack(fill="x", pady=(2, 0))
        ttk.Label(msr, text="マススペクトル(計算値)").pack(side="left")
        self.ion = tk.StringVar(value="M+")
        self.ion_box = ttk.Combobox(msr, textvariable=self.ion, width=9, state="readonly",
                                    values=("M+", "[M+H]+", "[M+Na]+", "[M-H]-"))
        self.ion_box.pack(side="left", padx=(8, 0))
        self.ion_box.bind("<<ComboboxSelected>>", lambda e: self.draw_ms())
        ttk.Button(msr, text="フラグメント…", command=self.open_fragments).pack(side="left", padx=(8, 0))
        self.ms_canvas = SpectrumView(right, height=110)   # ホイール: m/z 軸の拡大縮小 / ダブルクリック: 元に戻す
        self.ms_canvas.pack(fill="x", pady=(2, 0))
        self.ms_note = ttk.Label(right, text="", foreground="#444", wraplength=440, font=("Consolas", 9))
        self.ms_note.pack(anchor="w", pady=(0, 3), fill="x")
        self.rename_btn = ttk.Button(right, text="この構造に名前を付け直す(検証済みの名前)", command=self.rename)
        self.rename_btn.pack(anchor="w", pady=(4, 4))
        ttk.Label(right, text="ほかの候補(ダブルクリックでコピー)").pack(anchor="w")
        tv = ttk.Frame(right)
        tv.pack(fill="both", expand=True)
        self.cands = ttk.Treeview(tv, columns=("lv", "name"), show="headings", height=7)
        self.cands.heading("lv", text="判定")
        self.cands.heading("name", text="名前 / SMILES")
        self.cands.column("lv", width=90, stretch=False)
        self.cands.column("name", width=360)
        sb = ttk.Scrollbar(tv, orient="vertical", command=self.cands.yview)
        self.cands.configure(yscrollcommand=sb.set)
        self.cands.pack(side="left", fill="both", expand=True)
        sb.pack(side="left", fill="y")
        self.cands.bind("<Double-Button-1>", self._copy_cand)
        body.add(right, weight=4)

        self.entry.focus_set()
        threading.Thread(target=self._warmup, daemon=True).start()
        root.after(100, self.poll)

    # ------------------------------------------------------------ 背景処理
    def _warmup(self):
        try:
            core.warmup()
            self.q.put(("status", "準備完了"))
        except Exception as e:
            self.q.put(("status", f"読み込みに失敗: {e}"))

    def _enter(self, e):
        if e.state & 0x1:  # Shift+Enter は改行
            return None
        self.run()
        return "break"

    def _work(self, fn, *args):
        if self.busy:
            return
        self.busy = True
        self.btn.state(["disabled"])
        self.status.config(text="変換中…")

        def target():
            try:
                self.q.put(("result", fn(*args)))
            except Exception as e:
                self.q.put(("error", str(e)))
        threading.Thread(target=target, daemon=True).start()

    def run(self):
        text = self.entry.get("1.0", "end").strip()
        if text:
            # Tk の変数は別スレッドから読まない(tkinter はスレッド安全でない)。ここで値にしてから渡す
            self._work(self._convert, text, self.to.get(), self.calc_name.get(), self.opt_rename.get())

    @staticmethod
    def _convert(text: str, to: str, name_it: bool, rename: bool) -> core.Result:
        r = core.convert(text, to, name_it)
        if rename and r.direction == "名前→構造" and r.mol is not None:   # --rename: 構造から検証済みの名前も付け直す
            r2 = core.smiles_to_name(r.mol)
            if r2.ok:
                r.message += f"  構造から付けた名前: {r2.name}({r2.level})"
        return r

    def open_options(self):
        """コマンドの引数(--locants / --rename / --json)に当たる設定を、チェックボックスで切り替える別ウィンドウ。"""
        if self.opt_win is not None and self.opt_win.winfo_exists():
            self.opt_win.lift()
            return
        w = self.opt_win = tk.Toplevel(self.root)
        w.title("その他の設定")
        w.geometry("560x420")
        w.transient(self.root)
        f = ttk.Frame(w, padding=10)
        f.pack(fill="both", expand=True)
        ttk.Checkbutton(f, text="構造図に IUPAC の位置番号を描く(--locants)", variable=self.show_loc,
                        command=lambda: self.redraw(True)).pack(anchor="w", pady=2)
        ttk.Checkbutton(f, text="名前 → 構造のあと、構造から検証済みの名前も付け直す(--rename)  ※次の変換から",
                        variable=self.opt_rename).pack(anchor="w", pady=2)
        ttk.Checkbutton(f, text="結果を JSON でも表示する(--json)", variable=self.opt_json,
                        command=self._refresh_json).pack(anchor="w", pady=2)
        self.json_box = tk.Text(f, height=14, font=("Consolas", 10), wrap="none", state="disabled")
        self.json_box.pack(fill="both", expand=True, pady=(8, 4))
        bar = ttk.Frame(f)
        bar.pack(fill="x")
        ttk.Button(bar, text="JSON をコピー", command=lambda: self.copy(self._json_text())).pack(side="left")
        ttk.Button(bar, text="閉じる", command=w.destroy).pack(side="right")
        self._refresh_json()

    def _refresh_props(self):
        """「物性を計算する」が入っていれば、いまの構造の物性(Joback 法ほか。推算)を別ウィンドウに出す。切れば閉じる。"""
        r = self.result
        if not self.calc_props.get() or r is None or r.mol is None:
            if self.prop_win is not None and self.prop_win.winfo_exists():
                self.prop_win.destroy()
            return
        rows = props.properties(r.mol)
        if self.prop_win is None or not self.prop_win.winfo_exists():
            w = self.prop_win = tk.Toplevel(self.root)
            w.title("物性(推算)")
            w.transient(self.root)
            self.prop_frame = ttk.Frame(w, padding=8)
            self.prop_frame.pack(fill="both", expand=True)
            self.prop_checks: dict[str, tk.BooleanVar] = {}
        for c in self.prop_frame.winfo_children():
            c.destroy()
        for i, (key, label, val, unit, approx) in enumerate(rows):
            var = self.prop_checks.setdefault(key, tk.BooleanVar(value=True))
            ttk.Checkbutton(self.prop_frame, text=label, variable=var).grid(row=i, column=0, sticky="w")
            ttk.Label(self.prop_frame, text=val + (f" [{unit}]" if unit else "") + (" ≈" if approx else ""),
                      font=("Consolas", 10)).grid(row=i, column=1, sticky="e", padx=(14, 0))
        n = len(rows)
        ttk.Label(self.prop_frame, wraplength=300, foreground="#666",
                  text="すべて構造からの推算値(実測ではない)。沸点〜生成熱は Joback 法、logP・MR は Crippen、tPSA は Ertl。≈ は代用(CLogP・CMR は Crippen、LogS は ESOL)。N/A は未対応。"
                  ).grid(row=n, column=0, columnspan=2, sticky="w", pady=(8, 4))
        bar = ttk.Frame(self.prop_frame)
        bar.grid(row=n + 1, column=0, columnspan=2, sticky="ew")
        ttk.Button(bar, text="コピー(チェックした行)", command=lambda: self.copy("\n".join(
            f"{lab}: {val}" + (f" [{u}]" if u else "") for k, lab, val, u, _ in props.properties(self.result.mol)
            if self.prop_checks[k].get()))).pack(side="left")
        ttk.Button(bar, text="閉じる", command=lambda: self.calc_props.set(False) or self._refresh_props()).pack(side="right")

    def _json_text(self) -> str:
        r = self.result
        return json.dumps({**r.row(), "candidates": r.candidates}, ensure_ascii=False, indent=2) if r is not None else ""

    def _refresh_json(self):
        if self.opt_win is None or not self.opt_win.winfo_exists():
            return
        self.json_box.config(state="normal")
        self.json_box.delete("1.0", "end")
        if self.opt_json.get():
            self.json_box.insert("1.0", self._json_text() or "(まだ変換していません)")
        self.json_box.config(state="disabled")

    def rename(self):
        if self.result is not None and self.result.mol is not None:
            self._work(core.smiles_to_name, self.result.mol)

    def poll(self):
        try:
            while True:
                kind, data = self.q.get_nowait()
                if kind == "status":
                    self.status.config(text=data)
                elif kind == "progress":
                    i, n, msg = data
                    self.pbar.config(maximum=n, value=i)
                    self.status.config(text=msg)
                else:
                    self.busy = False
                    self.btn.state(["!disabled"])
                    if kind == "result":
                        self.show(data, remember=True)
                        self.status.config(text="完了")
                    elif kind == "batch":
                        self.status.config(text=f"保存しました: {data}")
                        if hasattr(os, "startfile") and messagebox.askyesno("一括変換", f"保存しました:\n{data}\n\n開きますか?"):
                            try:
                                os.startfile(data)
                            except OSError as e:
                                messagebox.showerror("開けません", str(e))
                        elif not hasattr(os, "startfile"):
                            messagebox.showinfo("一括変換", f"保存しました:\n{data}")
                    elif kind == "error":
                        self.status.config(text="エラー")
                        messagebox.showerror("エラー", data)
        except queue.Empty:
            pass
        self.root.after(100, self.poll)

    # ------------------------------------------------------------ 表示
    def show(self, r: core.Result, remember=False):
        self.result = r
        info = r.info
        vals = {"iupac_name": r.name, "smiles": info.get("smiles", ""),
                "formula": info.get("formula", ""),
                "mass": f"{info['mw']}  /  {info['exact_mass']}" if "mw" in info else "",
                "ea": ms.ea_text(r.mol) if r.mol is not None else ""}
        for k, v in self.vars.items():
            v.set(vals.get(k, ""))
        self.name_box.config(state="normal")
        self.name_box.delete("1.0", "end")
        self.name_box.insert("1.0", r.name)
        self.name_box.config(state="disabled")
        fg, bg = LEVEL_COLORS.get(r.level, ("#333", "#eee"))
        self.badge.config(text=f"{r.direction}:{LEVEL_TEXT.get(r.level, r.level)}", fg=fg, bg=bg)
        self.msg.config(text=r.message)
        self.cands.delete(*self.cands.get_children())
        for name, lv in r.candidates:
            self.cands.insert("", "end", values=(lv, name))
        self.rename_btn.state(["!disabled"] if r.mol is not None and (r.direction == "名前→構造" or r.level == "名前なし") else ["disabled"])
        self.redraw()
        self.draw_ms()
        self._refresh_json()
        self._refresh_props()
        if remember:
            self.history.insert(0, r)
            label = (r.query.splitlines()[0] if r.query else "")[:40]
            self.hist.insert(0, f"{'✓' if r.ok else '·'} {label}")

    def draw_ms(self):
        """計算で求めた同位体パターンを棒グラフにする(実測ではない)。電荷をもつ分子は、そのイオンの m/z。"""
        r = self.result
        if r is None or r.mol is None:
            self.ms_canvas.set_peaks([])
            self.ms_note.config(text="")
            return
        try:
            pts = ms.spectrum(r.mol, self.ion.get())
        except Exception as e:
            self.ms_canvas.set_peaks([])
            self.ms_note.config(text=f"計算できませんでした: {e}")
            return
        self.ms_canvas.set_peaks(pts)
        note = ms.ms_text(r.mol, self.ion.get(), top=6)
        if len(Chem.GetMolFrags(r.mol)) > 1:
            note += "  (塩・混合物: このイオンに合う成分。元素分析は全成分の合算)"
        if ms.approx(r.mol, self.ion.get()):
            note += "  (同位体の表にない元素を含む: パターンは概算)"
        self.ms_note.config(text=note)

    def open_fragments(self):
        r = self.result
        if r is None or r.mol is None:
            self.status.config(text="先に構造を出してください")
            return
        FragmentWindow(self.root, r.mol, r.name or r.query, self.ion.get())

    def redraw(self, keep_view: bool = False):
        r = self.result
        if r is None or r.mol is None:
            self.canvas.set_mol(None)
            return
        self.canvas.set_mol(r.mol, r.locants if self.show_loc.get() else None, keep_view=keep_view)

    def _pick_history(self, _e):
        sel = self.hist.curselection()
        if sel:
            self.show(self.history[sel[0]])

    def copy(self, text: str):
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.status.config(text=f"コピー: {text[:80]}")

    def _copy_cand(self, _e):
        sel = self.cands.selection()
        if sel:
            self.copy(self.cands.item(sel[0], "values")[1])

    # ------------------------------------------------------------ ファイル
    def _save(self, ext, writer):
        r = self.result
        if r is None or r.mol is None:
            return
        p = filedialog.asksaveasfilename(defaultextension=ext, filetypes=[(ext[1:].upper(), "*" + ext)])
        if p:
            try:
                writer(Path(p), r)
            except Exception as e:   # 書き込めない場所など。pythonw ではコンソールに出ないので必ず知らせる
                messagebox.showerror("保存できません", f"{p}\n{e}")
                return
            self.status.config(text=f"保存: {p}")

    def save_mol(self):
        self._save(".mol", lambda p, r: core.save_mol(r.mol, p, r.name))

    def save_sdf(self):
        from name2struct import write_results
        self._save(".sdf", lambda p, r: write_results([r], p))

    def save_png(self):
        self._save(".png", lambda p, r: p.write_bytes(
            core.mol_to_png(r.mol, (1000, 750), r.locants if self.show_loc.get() else None)))

    def save_svg(self):
        self._save(".svg", lambda p, r: p.write_text(
            core.mol_to_png(r.mol, (1000, 750), r.locants if self.show_loc.get() else None, svg=True),
            encoding="utf-8"))

    def open_file(self):
        p = filedialog.askopenfilename(title="構造ファイル", filetypes=[
            ("構造", "*.mol *.sdf *.cdxml *.smi"), ("すべて", "*.*")])
        if not p:
            return
        try:
            mols = core.read_file(Path(p))
        except Exception as e:
            messagebox.showerror("読めません", str(e))
            return
        if not mols:
            messagebox.showerror("読めません", "構造が見つかりませんでした。")
        elif len(mols) == 1:
            self.entry.delete("1.0", "end")
            self.entry.insert("1.0", core.Chem.MolToSmiles(mols[0][1]))
            self._work(core.smiles_to_name, mols[0][1])
        elif messagebox.askyesno("一括変換", f"{len(mols)} 個の構造があります。まとめて名前を付けて保存しますか?"):
            self._batch(Path(p))

    def batch(self):
        p = filedialog.askopenfilename(title="一括変換する入力", filetypes=[
            ("表・構造", "*.csv *.txt *.xlsx *.sdf *.mol *.cdxml *.smi"), ("すべて", "*.*")])
        if p:
            self._batch(Path(p))

    def _batch(self, src: Path):
        if self.busy:
            return
        dst = filedialog.asksaveasfilename(
            title="保存先", initialfile=src.stem + "_out.xlsx", initialdir=str(src.parent),
            defaultextension=".xlsx", filetypes=[("Excel", "*.xlsx"), ("CSV", "*.csv"), ("SDF", "*.sdf")])
        if not dst:
            return
        self.busy = True
        self.btn.state(["disabled"])
        to = self.to.get()   # Tk の変数は別スレッドから読まない

        def target():
            try:
                out = run_batch(src, Path(dst), to,
                                progress=lambda i, n, m: self.q.put(("progress", (i, n, m))))
                self.q.put(("batch", str(out)))
            except Exception as e:
                self.q.put(("error", str(e)))
        threading.Thread(target=target, daemon=True).start()

    def open_user_dict(self):
        p = core.HERE / "user_dict.tsv"
        try:
            if not p.exists():
                p.write_text(USER_DICT_TEMPLATE, encoding="utf-8")
        except OSError as e:
            messagebox.showerror("利用者辞書", f"作れません: {p}\n{e}")
            return
        try:
            subprocess.Popen(["notepad.exe", str(p)])
        except Exception:
            messagebox.showinfo("利用者辞書", str(p))
        if messagebox.askyesno("利用者辞書", "編集して保存したら「はい」を押すと読み直します。"):
            core.reload_dictionary()
            self.status.config(text="利用者辞書を読み直しました")


def main():
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
