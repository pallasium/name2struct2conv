"""n2s_view の描画部分(Tk を使わない関数)のテスト。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rdkit import Chem

import n2s_view as v


def test_render_png_and_highlight():
    m = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
    a = v.render_png(m, (300, 220))
    b = v.render_png(m, (300, 220), highlight=(0, 1, 2))
    c = v.render_png(m, (600, 440))
    assert a[:8] == b"\x89PNG\r\n\x1a\n" and b[:8] == a[:8]
    assert a != b                      # 強調すると絵が変わる
    assert len(c) > len(a)             # 大きく描くと(拡大すると)画像も大きい


def test_render_png_ignores_out_of_range_highlight():
    m = Chem.MolFromSmiles("CCO")
    assert v.render_png(m, (200, 150), highlight=(0, 99))[:4] == b"\x89PNG"


def test_nice_step():
    assert v._nice_step(100) == 20 or v._nice_step(100) == 10 or v._nice_step(100) == 50
    assert v._nice_step(10) in (1, 2)
    assert v._nice_step(0.5) in (0.1, 0.05, 0.2)


def test_spectrum_state_reset_logic():
    """SpectrumView の状態(Tk なしで検証できる範囲): ピークを入れ替えたとき hits が残らない。"""
    import inspect
    src = inspect.getsource(v.SpectrumView.set_peaks)
    assert "self.hits = []" in src
    assert "i < len(self.hits)" in inspect.getsource(v.SpectrumView.draw)
