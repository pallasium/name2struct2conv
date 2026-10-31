"""GUI の表示部品: ホイールで拡大縮小・ドラッグで移動・ダブルクリックで元の大きさに戻せる、構造図とスペクトル。

  StructureView  構造図(RDKit で描く)。拡大のたびにその大きさで描き直すので、拡大してもぼやけない
  SpectrumView   棒グラフ(m/z 軸の拡大縮小)。ピークをクリックで選択(コールバック)
操作は共通: ホイール = マウスの位置を中心に拡大縮小 / 左ドラッグ = 移動 / ダブルクリック = 元の大きさ
"""
from __future__ import annotations

import base64
import math
import tkinter as tk

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D


def render_png(mol: Chem.Mol, size: tuple[int, int], locants: dict | None = None, highlight=None) -> bytes:
    """構造図の PNG。highlight は強調する原子番号(フラグメント表示用)。"""
    m = Chem.Mol(mol)
    if locants:
        for i, loc in locants.items():
            if i < m.GetNumAtoms():
                m.GetAtomWithIdx(i).SetProp("atomNote", loc)
    d = rdMolDraw2D.MolDraw2DCairo(*size)
    o = d.drawOptions()
    o.addStereoAnnotation = True
    o.annotationFontScale = 0.6
    o.fixedBondLength = 40 * max(size[0], 460) / 460
    atoms = [i for i in (highlight or ()) if i < m.GetNumAtoms()]
    bonds = [b.GetIdx() for b in m.GetBonds() if b.GetBeginAtomIdx() in atoms and b.GetEndAtomIdx() in atoms]
    if atoms:
        red = (0.93, 0.35, 0.30)
        rdMolDraw2D.PrepareAndDrawMolecule(d, m, highlightAtoms=atoms, highlightBonds=bonds,
                                           highlightAtomColors={i: red for i in atoms},
                                           highlightBondColors={i: red for i in bonds})
    else:
        rdMolDraw2D.PrepareAndDrawMolecule(d, m)
    d.FinishDrawing()
    return d.GetDrawingText()


class _Zoomable(tk.Canvas):
    """ホイール・ドラッグ・ダブルクリックの共通の結びつけ。"""

    MIN, MAX = 0.25, 8.0

    def __init__(self, master, **kw):
        kw.setdefault("highlightthickness", 1)
        kw.setdefault("highlightbackground", "#bbb")
        super().__init__(master, **kw)
        self.bind("<MouseWheel>", lambda e: self._wheel(e.x, e.y, 1 if e.delta > 0 else -1))
        self.bind("<Button-4>", lambda e: self._wheel(e.x, e.y, 1))      # Linux
        self.bind("<Button-5>", lambda e: self._wheel(e.x, e.y, -1))
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<B1-Motion>", self._drag)
        self.bind("<ButtonRelease-1>", self._release)
        self.bind("<Double-Button-1>", lambda e: self.reset())
        self._last = None
        self._moved = False

    def _press(self, e):
        self._last = (e.x, e.y)
        self._moved = False

    def _drag(self, e):
        if self._last is None:
            return
        dx, dy = e.x - self._last[0], e.y - self._last[1]
        if abs(dx) + abs(dy) > 2:
            self._moved = True
        self._last = (e.x, e.y)
        self.pan(dx, dy)

    def _release(self, e):
        if not self._moved:
            self.click(e.x, e.y)
        self._last = None

    # サブクラスが実装する
    def _wheel(self, x, y, direction): ...
    def pan(self, dx, dy): ...
    def reset(self): ...
    def click(self, x, y): ...


class StructureView(_Zoomable):
    def __init__(self, master, **kw):
        kw.setdefault("background", "white")
        super().__init__(master, **kw)
        self.mol: Chem.Mol | None = None
        self.locants = None
        self.highlight = None
        self.scale, self.ox, self.oy = 1.0, 0.0, 0.0
        self._img = None
        self._job = None
        self._size = (0, 0)
        self.bind("<Configure>", self._on_configure)

    def set_mol(self, mol: Chem.Mol | None, locants=None, highlight=None, keep_view: bool = False):
        self.mol, self.locants, self.highlight = mol, locants, highlight
        if not keep_view:
            self.scale, self.ox, self.oy = 1.0, 0.0, 0.0
        self._schedule()

    def _on_configure(self, e):
        if (e.width, e.height) != self._size:
            self._size = (e.width, e.height)
            self._schedule()

    def _schedule(self):
        if self._job is None:
            self._job = self.after(30, self._render)

    def _render(self):
        self._job = None
        self.delete("all")
        self._img = None
        if self.mol is None:
            return
        w, h = max(self.winfo_width() - 2, 120), max(self.winfo_height() - 2, 120)
        size = (min(int(w * self.scale), 4000), min(int(h * self.scale), 4000))
        png = render_png(self.mol, size, self.locants, self.highlight)
        self._img = tk.PhotoImage(data=base64.b64encode(png))
        self.create_image(self.ox, self.oy, anchor="nw", image=self._img)

    def _wheel(self, x, y, direction):
        new = min(max(self.scale * (1.18 if direction > 0 else 1 / 1.18), self.MIN), self.MAX)
        k = new / self.scale
        if abs(k - 1) < 1e-9:
            return
        self.ox = x - (x - self.ox) * k          # マウスの下の点を動かさない
        self.oy = y - (y - self.oy) * k
        self.scale = new
        self._schedule()

    def pan(self, dx, dy):
        self.ox += dx
        self.oy += dy
        self.move("all", dx, dy)

    def reset(self):
        self.scale, self.ox, self.oy = 1.0, 0.0, 0.0
        self._schedule()


def _nice_step(span: float, target: int = 8) -> float:
    raw = span / target
    mag = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1
    for m in (1, 2, 5, 10):
        if raw <= m * mag:
            return m * mag
    return 10 * mag


class SpectrumView(_Zoomable):
    """棒グラフ。peaks = [(m/z, 相対強度%)]。m/z 軸を拡大縮小する(強度は 0〜100% 固定)。"""

    def __init__(self, master, on_select=None, **kw):
        kw.setdefault("background", "#fafafa")
        super().__init__(master, **kw)
        self.peaks: list[tuple[float, float]] = []
        self.kinds: list[str] = []
        self.hits: list = []                       # 予測ピークごとの、実測との一致(強度 % か None)。空なら色分けしない
        self.mirror: list[tuple[float, float]] = []   # 実測(下向きに描く)
        self.mirror_hit: list[bool] = []
        self.mirror_title = ""
        self.on_select = on_select
        self.selected: int | None = None
        self.lo = self.hi = 0.0
        self._home = (0.0, 1.0)
        self.bind("<Configure>", lambda e: self.draw())

    def set_peaks(self, peaks, kinds=None, selected=None, keep_view: bool = False):
        self.peaks = list(peaks)
        self.kinds = list(kinds) if kinds else ["peak"] * len(self.peaks)
        self.hits = []                      # ピークが変わったら、前の照合結果(一致の印)は捨てる
        self.selected = selected
        if self.peaks:
            lo, hi = min(m for m, _ in self.peaks), max(m for m, _ in self.peaks)
            pad = max((hi - lo) * 0.06, 1.5)
            self._home = (lo - pad, hi + pad)
        if not keep_view:
            self.lo, self.hi = self._home
        self.draw()

    def set_mirror(self, peaks, hit=None, title: str = "", pred_hits=None):
        """実測のスペクトルを、予測の下に鏡像で描く。hit は実測ピークごとの「予測と一致したか」。"""
        self.mirror = list(peaks or [])
        self.mirror_hit = list(hit) if hit else [False] * len(self.mirror)
        self.mirror_title = title
        self.hits = list(pred_hits) if pred_hits is not None else []
        if self.mirror:
            lo = min([m for m, _ in self.peaks] + [m for m, _ in self.mirror])
            hi = max([m for m, _ in self.peaks] + [m for m, _ in self.mirror])
            pad = max((hi - lo) * 0.06, 1.5)
            self._home = (lo - pad, hi + pad)
            self.lo, self.hi = self._home
        self.draw()

    def select(self, index: int | None, center: bool = False):
        self.selected = index
        if center and index is not None:
            mz = self.peaks[index][0]
            if not (self.lo < mz < self.hi):
                half = (self.hi - self.lo) / 2
                self.lo, self.hi = mz - half, mz + half
        self.draw()

    # ---- 座標
    def _geom(self):
        w, h = max(self.winfo_width(), 200), max(self.winfo_height(), 80)
        return w, h, 40, 22, 18     # 幅・高さ・左余白・上余白(強いピークのラベルが見切れない高さ)・下余白

    def _x(self, mz):
        w, _, left, _, _ = self._geom()
        return left + (w - left - 10) * (mz - self.lo) / (self.hi - self.lo)

    def _mz(self, x):
        w, _, left, _, _ = self._geom()
        return self.lo + (x - left) / (w - left - 10) * (self.hi - self.lo)

    def draw(self):
        self.delete("all")
        if not self.peaks or self.hi <= self.lo:
            return
        w, h, left, top, bot = self._geom()
        bottom = h - bot
        mir = bool(self.mirror)
        mid = (top + bottom) / 2 if mir else bottom          # 鏡像のときは中央が 0
        up = (mid - top)                                      # 上向き(予測)の高さ
        down = (bottom - mid)                                 # 下向き(実測)の高さ
        self.create_line(left, mid, w - 10, mid, fill="#888")
        self.create_line(left, top, left, bottom, fill="#888")
        ticks = (0, 50, 100) if not mir else (100, 50, 0, -50, -100)
        for v in ticks:
            y = mid - (up if v >= 0 else down) * abs(v) / 100 * (1 if v >= 0 else -1)
            self.create_line(left - 3, y, left, y, fill="#888")
            self.create_text(left - 5, y, text=str(abs(v)), anchor="e", font=("Consolas", 8), fill="#666")
        step = _nice_step(self.hi - self.lo)
        t = math.ceil(self.lo / step) * step
        while t <= self.hi:
            x = self._x(t)
            self.create_line(x, bottom, x, bottom + 3, fill="#888")
            self.create_text(x, bottom + 4, text=f"{t:g}", anchor="n", font=("Consolas", 8), fill="#666")
            t += step
        if mir:
            self.create_text(left + 4, top - 2, anchor="nw", text="予測", font=("Yu Gothic UI", 8), fill="#1f6fb4")
            self.create_text(left + 4, bottom - 2, anchor="sw", text="実測 " + self.mirror_title, font=("Yu Gothic UI", 8), fill="#b86a00")
        colors = {"precursor": "#8a8a8a", "cleavage": "#1f6fb4", "loss": "#2a9d55", "ring": "#7b4fb3", "peak": "#1f6fb4"}
        px_per_mz = (w - left - 10) / (self.hi - self.lo)
        # 実測(下向き)
        for (mz, rel), ok in zip(self.mirror, self.mirror_hit):
            if self.lo <= mz <= self.hi:
                x = self._x(mz)
                self.create_line(x, mid, x, mid + down * min(rel, 100) / 100, fill="#2a9d55" if ok else "#d28c1f", width=2)
        # 予測(上向き)
        order = sorted(range(len(self.peaks)), key=lambda i: -self.peaks[i][1])
        labelled: list[float] = []
        for i in order:
            mz, rel = self.peaks[i]
            if not (self.lo <= mz <= self.hi):
                continue
            x = self._x(mz)
            y = mid - up * min(rel, 100) / 100
            sel = i == self.selected
            matched = i < len(self.hits) and self.hits[i] is not None
            col = "#d9382c" if sel else ("#2a9d55" if matched else colors.get(self.kinds[i], "#1f6fb4"))
            self.create_line(x, mid, x, y, fill=col, width=3 if sel else 2)
            if (rel >= 4 or sel) and all(abs(x - lx) > 30 for lx in labelled):
                self.create_text(x, y - 8, text=f"{mz:.{4 if px_per_mz > 40 else 2}f}", font=("Consolas", 8),
                                 fill="#d9382c" if sel else "#333")
                labelled.append(x)

    # ---- 操作
    def _wheel(self, x, y, direction):
        w, _, left, _, _ = self._geom()
        mz = self._mz(max(x, left))
        k = (1 / 1.2) if direction > 0 else 1.2           # ホイール上 = 拡大(範囲を狭める)
        home_span = self._home[1] - self._home[0]
        span = min(max((self.hi - self.lo) * k, home_span / 200), home_span * 6)
        f = (mz - self.lo) / (self.hi - self.lo)
        self.lo, self.hi = mz - f * span, mz + (1 - f) * span
        self.draw()

    def pan(self, dx, dy):
        w, _, left, _, _ = self._geom()
        d = -dx * (self.hi - self.lo) / (w - left - 10)
        self.lo += d
        self.hi += d
        self.draw()

    def reset(self):
        self.lo, self.hi = self._home
        self.draw()

    def click(self, x, y):
        if not self.peaks:
            return
        best, dist = None, 9
        for i, (mz, _) in enumerate(self.peaks):
            d = abs(self._x(mz) - x)
            if d < dist:
                best, dist = i, d
        if best is not None:
            self.select(best)
            if self.on_select:
                self.on_select(best)
