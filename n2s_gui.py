"""name2struct2 の GUI(tkinter)。name2struct.py を引数なしで実行すると開く。"""
from __future__ import annotations

import base64
import os
import queue
import subprocess
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import n2s_core as core
from name2struct import run_batch

LEVEL_COLORS = {"一致": ("#1e7b34", "#e2f0d9"), "OPSIN": ("#1e7b34", "#e2f0d9"), "辞書": ("#1e7b34", "#e2f0d9"),
                "利用者辞書": ("#1e7b34", "#e2f0d9"), "立体以外一致": ("#8a6100", "#fff2cc"),
                "推定": ("#8a6100", "#fff2cc"), "未検証": ("#a12b1a", "#f8cbad"), "失敗": ("#a12b1a", "#f8cbad")}
LEVEL_TEXT = {"一致": "検証済み(OPSIN で同じ構造に戻る)", "OPSIN": "OPSIN で読んだ構造", "辞書": "辞書から",
              "利用者辞書": "利用者辞書から", "立体以外一致": "骨格は検証済み・立体は要確認",
              "推定": "推定(要確認)", "未検証": "未検証(誤りの可能性が高い)", "失敗": "読めませんでした"}
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
        self.img = None
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
        ttk.Button(mode, text="利用者辞書…", command=self.open_user_dict).pack(side="right")
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
        self.canvas = tk.Label(mid, background="white", relief="solid", borderwidth=1)
        self.canvas.pack(fill="both", expand=True)
        opts = ttk.Frame(mid)
        opts.pack(fill="x", pady=(4, 0))
        self.show_loc = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts, text="位置番号", variable=self.show_loc, command=self.redraw).pack(side="left")
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
        for key, label in (("aliases", "別名(辞書)"), ("smiles", "SMILES"),
                           ("inchi", "InChI"), ("inchikey", "InChIKey"), ("formula", "分子式"),
                           ("mass", "分子量 / 精密質量")):
            ttk.Label(right, text=label).pack(anchor="w")
            r = ttk.Frame(right)
            r.pack(fill="x", pady=(0, 3))
            v = tk.StringVar()
            ttk.Entry(r, textvariable=v, state="readonly", font=("Consolas", 10)).pack(side="left", fill="x", expand=True)
            ttk.Button(r, text="コピー", width=6, command=lambda v=v: self.copy(v.get())).pack(side="left", padx=(2, 0))
            self.vars[key] = v
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

        self.canvas.bind("<Configure>", lambda e: self.redraw())
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
            self._work(core.convert, text, self.to.get())

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
                        if messagebox.askyesno("一括変換", f"保存しました:\n{data}\n\n開きますか?"):
                            os.startfile(data) if hasattr(os, "startfile") else None
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
        vals = {"iupac_name": r.name, "aliases": " / ".join(r.aliases), "smiles": info.get("smiles", ""),
                "inchi": info.get("inchi", ""), "inchikey": info.get("inchikey", ""),
                "formula": info.get("formula", ""),
                "mass": f"{info['mw']}  /  {info['exact_mass']}" if "mw" in info else ""}
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
        self.rename_btn.state(["!disabled"] if r.mol is not None and r.direction == "名前→構造" else ["disabled"])
        self.redraw()
        if remember:
            self.history.insert(0, r)
            label = (r.query.splitlines()[0] if r.query else "")[:40]
            self.hist.insert(0, f"{'✓' if r.ok else '·'} {label}")

    def redraw(self):
        r = self.result
        if r is None or r.mol is None:
            self.canvas.config(image="")
            self.img = None
            return
        w = max(self.canvas.winfo_width() - 4, 300)
        h = max(self.canvas.winfo_height() - 4, 240)
        png = core.mol_to_png(r.mol, (w, h), r.locants if self.show_loc.get() else None)
        self.img = tk.PhotoImage(data=base64.b64encode(png))
        self.canvas.config(image=self.img)

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
            writer(Path(p), r)
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

        def target():
            try:
                out = run_batch(src, Path(dst), self.to.get(),
                                progress=lambda i, n, m: self.q.put(("progress", (i, n, m))))
                self.q.put(("batch", str(out)))
            except Exception as e:
                self.q.put(("error", str(e)))
        threading.Thread(target=target, daemon=True).start()

    def open_user_dict(self):
        p = core.HERE / "user_dict.tsv"
        if not p.exists():
            p.write_text(USER_DICT_TEMPLATE, encoding="utf-8")
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
