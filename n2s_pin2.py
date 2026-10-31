"""PIN の書き方への書き換え、構造化版(n2s_pin.py の作り直し)。

n2s_pin.py は正規表現と文字列の切り貼りで、括弧の入れ子・英数字順・di / bis・位置番号をそれぞれ別の場所で
直していて、互いの結果を見ていない。ここでは名前を一度木にして、すべての規則を木の上で一度に適用する。

  parse(name, parent_ok)   名前の断片 -> Frag(立体記号, [Sub...], 主骨格)。括弧は実際に対応を取って読む
  render(frag)             木 -> 名前。規則(括弧に入れるか・入れ子の括弧の種類・英数字順・同じ置換基の束ね・
                            di と bis の使い分け・位置番号の並び)はここだけに書く
  normalize(name, ...)     parse -> 葉の言い換え -> render。何度かけても同じ結果になる(冪等)

葉の言い換え(benzenize・N 位置番号・親の言い換え)は n2s_pin のものをそのまま使う。
呼び出し側(n2s_core)が OPSIN で構造を確かめてから採る点は n2s_pin と同じ。
"""
from __future__ import annotations

import re
from contextlib import contextmanager
from dataclasses import dataclass, field

import n2s_pin as pin

OPEN, CLOSE = "([{", ")]}"

# ---------------------------------------------------------------- 規則表(ID・条文・説明)
# G00〜 は n2s_pin._GLOBAL の各行(名前全体の言い換え)。`disabled(...)` で 1 つずつ止めて寄与を測れる(pin_eval/ablate.py)。
RULES = {
    "R-NORM": ("P-16 / P-14.5", "語ごとに名前を木にして描き直す(括弧・英数字順・di/bis・位置番号の並び)"),
    "R-ORDER": ("P-14.5", "置換基を英数字順に並べる"),
    "R-MERGE": ("P-16.3", "同じ置換基を束ねて di / bis / tris にする"),
    "R-BENZ": ("P-22.1 / P-14.3", "シクロヘキサトリエン -> benzene / phenyl、位置番号を最小に付け直す"),
    "R-NLOC": ("P-62 / P-66", "尿素 N,N' / ジアミン N1,N4 などの N の位置記号"),
    "R-SUBREN": ("P-32", "置換基の言い換え(methylsulfinyl -> methanesulfinyl、phenylphenyl -> biphenyl など)"),
    "R-PARENT": ("P-31.1.4", "主骨格の位置番号(ethanone -> ethan-1-one、N だけ置換なら ethanamine)"),
    "R-UNWRAP": ("P-65.6", "エステルの最初の語の外側の括弧を外す"),
    "R-DROPLOC": ("P-14.3.4", "全置換の ethene は位置番号を省く(tetrachloroethene)"),
    "R-NEST": ("P-16.5", "括弧の入れ子を ( [ { の順にする"),
    "R-SUFFIX": ("P-41 / P-44.1.1", "他に優先する基が無い環では、hydroxy / amino を接頭語でなく接尾語にする(8-hydroxyquinoline -> quinolin-8-ol)"),
}
_OFF: set[str] = set()


def on(rule_id: str) -> bool:
    return rule_id not in _OFF


@contextmanager
def disabled(*ids: str):
    """with disabled('R-ORDER'): ... の間だけ規則を止める(ablation 用)。"""
    old = set(_OFF)
    _OFF.update(ids)
    try:
        yield
    finally:
        _OFF.clear()
        _OFF.update(old)


def global_rule_ids() -> list[str]:
    return [f"G{i:02d}" for i in range(len(pin._GLOBAL))]

_LOC = pin._LOC
_LOCS = pin._LOCS


# ---------------------------------------------------------------- 木

@dataclass
class Sub:
    loc: str                      # '2,3' / 'N,N' / ''
    mult: int = 1                 # 同じ置換基の数
    core: "Frag | str" = ""       # 置換基の中身(文字列 = これ以上分けない / Frag = 入れ子)
    bis: bool = False             # bis / tris 形で書かれていた


@dataclass
class Frag:
    stereo: str = ""
    subs: list[Sub] = field(default_factory=list)
    parent: str = ""


# ---------------------------------------------------------------- 括弧の読み取り

def _match(s: str, i: int) -> int:
    """s[i] の開き括弧に対応する閉じ括弧の位置(無ければ -1)。"""
    d = 0
    for j in range(i, len(s)):
        if s[j] in OPEN:
            d += 1
        elif s[j] in CLOSE:
            d -= 1
            if d == 0:
                return j
    return -1


def _depth0(s: str) -> list[bool]:
    d, out = 0, []
    for ch in s:
        if ch in CLOSE:
            d -= 1
        out.append(d == 0)
        if ch in OPEN:
            d += 1
    return out


def _balanced(s: str) -> bool:
    d = 0
    for ch in s:
        d += ch in OPEN
        d -= ch in CLOSE
        if d < 0:
            return False
    return d == 0


# ---------------------------------------------------------------- parse

_MULT_N = {v: k for k, v in pin._MULT.items()}
_MULTC_N = {v: k for k, v in pin._MULT_C.items()}


def parse(frag: str, parent_ok=None, depth: int = 0) -> Frag | None:
    """断片を木にする。分けられなければ None(呼び出し側は元の名前のまま使う)。"""
    if not _balanced(frag):
        return None
    stereo = ""
    m = pin._STEREO.match(frag)
    if m:
        stereo, frag = m.group(0), frag[m.end():]
    top = _depth0(frag)
    cuts = []
    for i, ch in enumerate(frag):
        if ch == "-" and top[i]:
            m = re.match(rf"({_LOCS})-(?=[a-z(\[{{])", frag[i + 1:])
            if m and pin._SUB_END.search(frag[:i]):
                cuts.append(i)
    pieces, start = [], 0
    for c in cuts:
        pieces.append(frag[start:c])
        start = c + 1
    pieces.append(frag[start:])
    raw = []
    for p in pieces:
        m = re.match(rf"({_LOCS})-(.*)$", p, re.S)
        raw.append((m.group(1), m.group(2)) if m else ("", p))
    loc, last = raw[-1]
    parent, body = pin._split_parent(last, parent_ok)
    if parent is None:
        return None
    if not body and loc:   # 主骨格の頭の位置番号は、環名の一部(1,3-benzodioxole・1-benzofuran)のときだけ主骨格に含める
        if parent_ok is not None and parent_ok(f"{loc}-{parent}"):
            subs = [_sub(l, b, parent_ok, depth) for l, b in raw[:-1]]
            return Frag(stereo, subs, f"{loc}-{parent}")
        return None
    raw = raw[:-1] + ([(loc, body)] if body else [])
    subs = [_sub(l, b, parent_ok, depth) for l, b in raw]
    return Frag(stereo, subs, parent)


def _sub(loc: str, body: str, parent_ok, depth: int) -> Sub:
    mult, bis = 1, False
    m = pin._MULTC_RE.match(body)
    if m:
        mult, bis, body = _MULTC_N[m.group(1)], True, body[m.end():]
    else:
        m = pin._MULT_RE.match(body)
        if m:
            rest = body[m.end():]
            if rest[:1] in OPEN and _match(rest, 0) == len(rest) - 1 or pin._SIMPLE.match(rest):
                mult, body = _MULT_N[m.group(1)], rest
    core: Frag | str = body
    if body[:1] in OPEN and _match(body, 0) == len(body) - 1:
        inner = body[1:-1]
        sub = parse(inner, parent_ok, depth + 1) if depth < 6 else None
        core = sub if sub is not None and sub.subs else inner
    elif depth < 6:
        sub = parse(body, parent_ok, depth + 1) if re.search(rf"({_LOC})-", body) else None
        core = sub if sub is not None and sub.subs else body
    if isinstance(core, str) and depth < 6:
        core = _inner_groups(core, parent_ok, depth)
    return Sub(loc, mult, core, bis)


def _inner_groups(s: str, parent_ok, depth: int) -> "Frag | str":
    """'[2,4,5-tris(chloro)phenyl]oxy' のように、括弧のかたまりが中に入った葉: かたまりごとに木にして描き直す。"""
    out, last, changed = [], 0, False
    i = 0
    while i < len(s):
        if s[i] in OPEN:
            j = _match(s, i)
            if j < 0:
                return s
            sub = parse(s[i + 1:j], parent_ok, depth + 1)
            if sub is not None and sub.subs:
                out.append(s[last:i])
                out.append("(" + _text(_leaf(sub, parent_ok)) + ")")
                last, changed = j + 1, True
            i = j + 1
        else:
            i += 1
    if not changed:
        return s
    out.append(s[last:])
    return "".join(out)


# ---------------------------------------------------------------- render

def _text(core: "Frag | str") -> str:
    """入れ子の括弧の種類を決める前の、中身の文字列(括弧は ( で仮置き)。"""
    return core if isinstance(core, str) else _render(core, True)[0]


def _is_compound(core: "Frag | str") -> bool:
    if isinstance(core, Frag):
        return True
    return not pin._is_simple(core)


def _render_sub(s: Sub) -> tuple[str, str]:
    """-> (置換基の名前(括弧つき)、英数字順の鍵)。位置番号と倍数は呼び出し側で付ける。"""
    inner = _text(s.core)
    compound = _is_compound(s.core)
    key = pin.alpha_key(inner if not compound else f"({inner})")
    if compound:
        inner = "(" + inner + ")"
    return inner, key


_HYDRO_CORE = re.compile(r"^(?:di|tri|tetra|penta|hexa|hepta|octa|nona|deca)?hydro$")


def _render(f: Frag, flat: bool = False) -> tuple[str, int]:   # flat は互換用(括弧は最後に render が決める)
    groups: dict[str, dict] = {}
    hydro = [s for s in f.subs if isinstance(s.core, str) and _HYDRO_CORE.match(s.core)]   # hydro 接頭語は英数字順に並べず、母体の直前に置く
    for s in [x for x in f.subs if x not in hydro]:
        name, key = _render_sub(s)
        gid = name if on("R-MERGE") else f"{name}\x00{len(groups)}"
        g = groups.setdefault(gid, {"key": key, "locs": [], "n": 0, "compound": name.startswith("("),
                                     "first": len(groups)})
        g["locs"] += [t for t in s.loc.split(",") if t]
        g["n"] += s.mult
    parts = []
    ordered = sorted(groups.items(), key=lambda kv: (kv[1]["key"] if on("R-ORDER") else "", kv[1]["first"]))
    for name, g in ordered:
        name = name.split("\x00")[0]
        n = g["n"]
        locs = sorted(g["locs"], key=_loc_sort)
        pre = ""
        if n > 1:
            pre = pin._MULT_C.get(n, "") if g["compound"] else pin._MULT.get(n, "")
        parts.append((",".join(locs), pre + name))
    out = ""
    for loc, body in parts:
        piece = f"{loc}-{body}" if loc else body
        out = piece if not out else (out + "-" + piece if re.match(_LOC, piece) else out + piece)
    if hydro:
        htxt = "-".join(f"{','.join(sorted((x for x in s.loc.split(',') if x), key=_loc_sort))}-{s.core}" if s.loc else s.core for s in hydro)
        out = (out + "-" if out else "") + htxt
        if f.parent and f.parent[0].isdigit():
            out += "-"
    elif out and f.parent and f.parent[0].isdigit():
        out += "-"
    return f.stereo + out + f.parent, 0


def _loc_sort(t: str):
    m = re.match(r"([A-Za-z]*)(\d*)([a-z]?)('*)", t)
    return (0 if m.group(1) else 1, m.group(1), int(m.group(2) or 0), m.group(3), len(m.group(4)))


def _brackets(s: str) -> str:
    """仮置きの ( を、深い方から ( [ { ( ... の順の本来の括弧にする。付加水素 (2H) は数えない。"""
    protected = re.sub(r"\((\d+[a-z]?H(?:,\d+[a-z]?H)*)\)", "\x02\\1\x03", s)
    return _brackets_core(protected).replace("\x02", "(").replace("\x03", ")")


def _brackets_core(s: str) -> str:
    out = list(s)
    stack = []
    pairs = []
    for i, ch in enumerate(s):
        if ch == "(":
            stack.append(i)
        elif ch == ")" and stack:
            pairs.append((stack.pop(), i))
    # 各ペアの高さ = 中にある一番深い入れ子の段数
    height = {}
    for a, b in sorted(pairs, key=lambda p: p[1] - p[0]):
        inner = [height[q] for q in height if a < q[0] and q[1] < b]
        height[(a, b)] = (max(inner) + 1) if inner else 0
    for (a, b), h in height.items():
        out[a], out[b] = OPEN[h % 3], CLOSE[h % 3]
    return "".join(out)


def render(f: Frag) -> str:
    return _brackets(_render(f, True)[0])


# ---------------------------------------------------------------- 葉の規則

def _leaf(f: Frag, parent_ok) -> Frag:
    # benzenize / N 位置番号は (loc, body) の形で動く
    flat = [(s.loc, _text(s.core)) for s in f.subs]
    parent = f.parent
    if on("R-BENZ"):
        flat, parent = pin.benzenize(flat, parent)
    if on("R-NLOC"):
        flat = pin.n_locants(flat, parent)
    new_subs = []
    for (loc, body), s in zip(flat, f.subs):
        core = s.core
        if isinstance(core, str):
            core = _strip_wrap(pin.rename_sub(core)) if on("R-SUBREN") else core
        new_subs.append(Sub(loc, s.mult, core, s.bis))
    if on("R-SUFFIX"):
        new_subs, parent = promote_suffix(new_subs, parent)
    if on("R-PARENT"):
        parent = pin.rename_parent(parent, pin.c_substituted([(s.loc, "") for s in new_subs]))
    for s in new_subs:
        if isinstance(s.core, Frag):
            flat_txt = _text(s.core)
            renamed = pin.rename_sub(flat_txt) if on("R-SUBREN") else flat_txt
            s.core = renamed if renamed != flat_txt else _leaf(s.core, parent_ok)
    return Frag(f.stereo, new_subs, parent)


_LOWER = re.compile(r"^(?:fluoro|chloro|bromo|iodo|nitro|methyl|ethyl|propyl|butyl|methoxy|ethoxy|propoxy|butoxy|phenyl|propan-2-yl|tert-butyl)$")


def promote_suffix(subs: list, parent: str):
    """環の母体(…ine / …ene / …ole など、接尾語の無い名前)に hydroxy か amino が 1 つだけあり、他が優先順位の低い基だけなら、
    その基を接尾語にする。hydroxy が優先(OH > NH2)。優先する基(酸・エステル・アミド・ニトリル・アルデヒド・ケトン)があるときは何もしない。"""
    # 接尾語の無い素の環名だけ(位置番号つきの接尾語 -1,4-dione などがあれば対象外。ケトン・酸・アミドなどの優先する基を壊さない)
    if not re.fullmatch(r"(?:\d+(?:,\d+)*-)?[a-z\[\],\d']*[a-z]e", parent) or "-" in parent.split("-", 1)[-1] \
            or re.search(r"(?:yl|ane|ene|yne|one|al|ide|ate|ole?)$", parent) and not re.search(r"(?:azole|oxole|ole)$", parent):
        return subs, parent
    cores = [(i, s) for i, s in enumerate(subs) if isinstance(s.core, str)]
    for group, suffix in (("hydroxy", "ol"), ("amino", "amine")):
        hit = [(i, s) for i, s in cores if s.core == group and s.mult == 1 and re.fullmatch(r"\d+[a-z]?", s.loc)]
        if len(hit) != 1:
            continue
        others = [s for i, s in enumerate(subs) if i != hit[0][0]]
        if group == "amino" and any(isinstance(s.core, str) and s.core == "hydroxy" for s in others):
            continue
        if all(isinstance(s.core, Frag) or bool(_LOWER.match(s.core)) or not re.search(r"oxo|carboxy|formyl|cyano|carbamoyl|sulfo|hydroxy|amino|imino", s.core) for s in others):
            i, s = hit[0]
            return [x for j, x in enumerate(subs) if j != i], f"{parent[:-1]}-{s.loc}-{suffix}"
    return subs, parent


def _strip_wrap(core: str) -> str:
    return core[1:-1] if core[:1] in OPEN and _match(core, 0) == len(core) - 1 else core


def normalize_fragment(frag: str, parent_ok=None) -> str:
    tree = parse(frag, parent_ok)
    if tree is None or not tree.subs:
        return frag
    return render(_leaf(tree, parent_ok))


def normalize(name: str, parent_ok=None) -> str:
    """空白で分かれた語ごとに normalize_fragment を適用する(酸・オキシドは前の語とひとまとめ)。"""
    words = re.sub(r" (acid|oxide)\b", r"~\1", name).split(" ")
    return " ".join(normalize_fragment(w.replace("~", " "), parent_ok).replace("~", " ") for w in words)


# ---------------------------------------------------------------- 候補の作り方(n2s_pin.rewrites と同じ口)

def rewrites(name: str, parent_ok=None) -> list[str]:
    """PIN に近づけた候補(強い順)。名前全体の言い換え(n2s_pin._GLOBAL)→ 語ごとの木の正規化 → 括弧の入れ子の整形。"""
    g = pin._unwrap_first(name) if on("R-UNWRAP") else name
    for i, (pat, rep) in enumerate(pin._GLOBAL):
        if on(f"G{i:02d}"):
            g = re.sub(pat, rep, g)
    outs = []
    for base in dict.fromkeys([g, name]):
        if on("R-NORM"):
            norm = normalize(base, parent_ok)
            outs.append(norm)
            if on("R-DROPLOC"):
                outs.append(" ".join(pin._drop_locants(w) for w in norm.split(" ")))
        outs.append(base)
    if on("R-NEST"):
        outs = [pin.tidy_nesting(o) for o in outs] + outs
    return [o for o in dict.fromkeys(outs) if o != name]


def applied_rules(before: str, after: str, parent_ok=None) -> list[str]:
    """before -> after で働いた規則の ID の目安(名前全体の言い換えは before に当たったものを数える)。"""
    if before == after:
        return []
    ids = [f"G{i:02d}" for i, (pat, _) in enumerate(pin._GLOBAL) if re.search(pat, before)]
    if normalize(before, parent_ok) != before:
        ids.append("R-NORM")
    return ids
