"""名前を IUPAC 推奨名(PIN, 2013 勧告)の書き方に近づける書き換え。

どの書き換えも文字列の操作だけで、構造は変えないつもりで作ってあるが、保証はしない。
呼び出し側(n2s_core)が、書き換えた名前を OPSIN で構造に戻し、元の構造と
(互変異性体まで含めて)同じになるときだけ採用する。

  split(frag)        名前の断片を [立体記号] + [(位置番号, 置換基), ...] + 主骨格 に分ける
  rewrites(name)     書き換えた候補を、強いものから順に返す
  pin_penalty(name)  PIN の規則に合わない書き方の数(小さいほど PIN らしい)
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------- 字句

_LOC = r"(?:\d+[a-z]?'*|[NOSP]\d*'*|[NOSP]|\d+H)"
_LOCS = rf"{_LOC}(?:,{_LOC})*"
_STEREO = re.compile(r"\((?:(?:\d+[a-z]?'*|[A-Z]\d*'*)?(?:[RSEZrs]\*?|RS|SR))(?:,(?:(?:\d+[a-z]?'*|[A-Z]\d*'*)?(?:[RSEZrs]\*?|RS|SR)))*\)-")
# 置換基の名前の終わり方(ここで区切ってよい)
_SUB_END = re.compile(r"(?:yl|ylidene|ylidyne|oxy|o|thio|amino|imino|amido|sulfanyl|sulfinyl|sulfonyl|"
                      r"carbonyl|oyl|io|ido|ato|ylene|diyl|triyl|idene|ylium)$|[)\]}]$")
_MULT = {2: "di", 3: "tri", 4: "tetra", 5: "penta", 6: "hexa", 7: "hepta", 8: "octa", 9: "nona", 10: "deca"}
_MULT_C = {2: "bis", 3: "tris", 4: "tetrakis", 5: "pentakis", 6: "hexakis"}
_MULT_RE = re.compile(r"^(di|tri|tetra|penta|hexa|hepta|octa|nona|deca)(?=[a-z])")
_MULTC_RE = re.compile(r"^(bis|tris|tetrakis|pentakis|hexakis)(?=[(\[{])")

# 括弧が要らない「単純な」置換基(P-16.5.1.2 の単純置換基接頭語に当たるもの。網羅ではない)
_ALK = r"(?:meth|eth|prop|but|pent|hex|hept|oct|non|dec|undec|dodec|tridec|tetradec|pentadec|hexadec|heptadec|octadec|nonadec|icos)"
_SIMPLE = re.compile(
    rf"^(?:{_ALK}yl|{_ALK}oxy|{_ALK}ylidene|cyclo{_ALK}yl|phenyl|benzyl|phenoxy|benzoyl|acetyl|formyl|"
    r"benzylidene|tert-butyl|sec-butyl|tert-pentyl|fluoro|chloro|bromo|iodo|hydroxy|oxo|amino|imino|nitro|nitroso|cyano|"
    r"azido|carboxy|carbamoyl|sulfamoyl|sulfanyl|sulfo|sulfanylidene|isocyanato|isothiocyanato|thiocyanato|"
    r"cyanato|hydrazinyl|hydroxyamino|ethenyl|ethynyl|methylidene|oxido|aminium|azaniumyl|ammonio|"
    r"acetamido|anilino|phosphono|tert-butyl)$")
# 慣用的に括弧に入れない例外(PIN でも括弧なし)
_NOPAREN = {"tert-butyl", "sec-butyl"}


def _depth_map(s: str) -> list[int]:
    d, out = 0, []
    for ch in s:
        if ch in "([{":
            out.append(d)
            d += 1
        elif ch in ")]}":
            d -= 1
            out.append(d)
        else:
            out.append(d)
    return out


def _balanced(s: str) -> bool:
    d = 0
    for ch in s:
        d += ch in "([{"
        d -= ch in ")]}"
        if d < 0:
            return False
    return d == 0


# ---------------------------------------------------------------- 分解

def parent_checker(reads):
    """reads(名前のリスト) -> 構造に読めたかのリスト、から parent_ok を作る(結果は覚えておく)。
    置換基(…yl / …ylidene)は methane につないで読む。"""
    memo: dict[str, bool] = {}

    def ok(tail: str) -> bool:
        if tail not in memo:
            probe = tail + "methane" if re.search(r"yl(?:idene)?$", tail) else tail
            memo[tail] = bool(reads([probe])[0])
        return memo[tail]
    return ok


def split(frag: str, parent_ok=None):
    """断片 -> (立体記号, [(位置番号, 置換基)], 主骨格)。分けられなければ None。
    parent_ok(s): 主骨格として読めるかを判定する関数(最後の置換基と主骨格がつながっているときに使う)。"""
    if not _balanced(frag):
        return None
    stereo = ""
    m = _STEREO.match(frag)
    if m:
        stereo, frag = m.group(0), frag[m.end():]
    depth = _depth_map(frag)
    # 区切り: 深さ 0 の "-" のあとに 位置番号 + "-" + 文字/括弧、かつ前が置換基の終わり方
    cuts = []
    for i, ch in enumerate(frag):
        if ch != "-" or depth[i] != 0:
            continue
        m = re.match(rf"({_LOCS})-(?=[a-z(\[{{])", frag[i + 1:])
        if m and _SUB_END.search(frag[:i]):
            cuts.append(i)
    pieces, start = [], 0
    for c in cuts:
        pieces.append(frag[start:c])
        start = c + 1
    pieces.append(frag[start:])
    segs = []
    for p in pieces:
        m = re.match(rf"({_LOCS})-(.*)$", p, re.S)
        segs.append((m.group(1), m.group(2)) if m else ("", p))
    # 最後のかたまりを 置換基 + 主骨格 に分ける
    loc, last = segs[-1]
    parent, body = _split_parent(last, parent_ok)
    if parent is None:
        return None
    if not body and loc:   # 主骨格の頭の位置番号は、環名の一部(1,3-benzodioxole・1-benzofuran)のときだけ主骨格に含める
        if parent_ok is not None and parent_ok(f"{loc}-{parent}"):
            return stereo, segs[:-1], f"{loc}-{parent}"
        return None
    segs = segs[:-1] + ([(loc, body)] if body else [])
    return stereo, segs, parent


def _split_parent(text: str, parent_ok):
    """'methylsulfinylbutane' -> ('butane', 'methylsulfinyl')。主骨格だけなら (text, '')。"""
    depth = _depth_map(text)
    best = None
    for k in range(1, len(text)):
        if depth[k] != 0 or not text[k].isalpha() and text[k] not in "([{" :
            continue
        head, tail = text[:k], text[k:]
        if not _SUB_END.search(head) or not _balanced(head) or not _balanced(tail) or re.search(r"(?:cyclo|benzo|carbo|thio|thia|aza|oxa|phospha|sila|bora|selena|tellura)$", head):
            continue
        if parent_ok is None or parent_ok(tail):
            best = (tail, head)  # 主骨格は一番短いもの(置換基は一番長いもの)
    if best:
        return best
    return text, ""


def join(stereo: str, segs: list[tuple[str, str]], parent: str) -> str:
    parts = [f"{loc}-{body}" if loc else body for loc, body in segs]
    out = ""
    for p in parts:
        out = p if not out else (out + "-" + p if re.match(_LOC, p) else out + p)
    if out and parent and parent[0].isdigit():
        out += "-"
    return stereo + out + parent


# ---------------------------------------------------------------- 個々の規則

def _strip_mult(body: str) -> tuple[int, str]:
    if body[:1] in "([{":   # 括弧で包まれたものの中の di... は倍数ではない
        return 1, body
    m = _MULTC_RE.match(body)
    if m:
        n = {v: k for k, v in _MULT_C.items()}[m.group(1)]
        inner = body[m.end():]
        if inner[0] in "([{" and _balanced(inner[1:-1]) and inner[-1] in ")]}":
            return n, inner[1:-1]
    m = _MULT_RE.match(body)
    if m:
        n = {v: k for k, v in _MULT.items()}[m.group(1)]
        rest = body[m.end():]
        if rest.startswith(("(", "[", "{")) and rest.endswith((")", "]", "}")) and _balanced(rest[1:-1]):
            return n, rest[1:-1]
        if _SIMPLE.match(rest):   # dimethylamino のような複合置換基は 'di' + 'methylamino' ではない
            return n, rest
    return 1, body


def _unwrap(body: str) -> str:
    if body and body[0] in "([{" and body[-1] in ")]}" and _balanced(body[1:-1]):
        return body[1:-1]
    return body


def _is_simple(core: str) -> bool:
    return bool(_SIMPLE.match(core)) or core in _NOPAREN


def enclose(body: str) -> str:
    """複合・位置番号つきの置換基を括弧に入れ、倍数は di / bis を使い分ける(P-16.5.1)。"""
    n, core = _strip_mult(body)
    core = _unwrap(core)
    if not core:
        return body
    if _is_simple(core):
        return (_MULT[n] if n > 1 else "") + core
    if n > 1:
        return _MULT_C.get(n, _MULT[n]) + "(" + core + ")"
    return "(" + core + ")"


def alpha_key(body: str) -> str:
    """英数字順(P-14.5)の並べ替えの鍵: 単純置換基の倍数接頭語・イタリック接頭語・位置番号・立体記号を除く。"""
    n, core = _strip_mult(body)
    core = _unwrap(core)
    core = _STEREO.sub("", core)
    while True:   # 先頭の イタリック接頭語・位置番号 を全部外す(2-tert-butyl... -> butyl...)
        new = re.sub(r"^(?:tert|sec|cis|trans|[NOS]\d*'*|\d+[a-z]?'*(?:,\d+[a-z]?'*)*)-", "", core)
        new = re.sub(rf"^(?:{_LOCS})-", "", new)
        if new == core:
            break
        core = new
    return re.sub(r"[^a-z]", "", core.lower())


def reorder(segs):
    """置換基を英数字順に並べる。同じ鍵なら元の順。"""
    return sorted(segs, key=lambda s: alpha_key(s[1]))


def merge_same(segs):
    """'N-(X)-N'-(X)' のような同じ置換基を 1 つにまとめる(N,N'-bis(X))。"""
    out: list[tuple[str, str]] = []
    for loc, body in segs:
        n, core = _strip_mult(body)
        core = _unwrap(core)
        for i, (l2, b2) in enumerate(out):
            n2, c2 = _strip_mult(b2)
            if _unwrap(c2) == core and loc and l2:
                locs = l2.split(",") + loc.split(",")
                out[i] = (",".join(locs), enclose(_MULT.get(len(locs), "") + core if _is_simple(core)
                                                  else _MULT_C.get(len(locs), "") + "(" + core + ")"))
                break
        else:
            out.append((loc, body))
    return out



# シクロヘキサトリエン(ベンゼン環の別の書き方)を benzene / phenyl に直し、番号を最小に付け直す
_TRIENE = re.compile(r"^cyclohexa-(?:1,3,5|2,4,6)-trien(e?)(.*)$")
_BENZ_SPECIAL = [
    (r"^benzene-1-ol$", "phenol"), (r"^benzene-1-amine$", "aniline"),
    (r"^benzene-1-carboxylic acid$", "benzoic acid"), (r"^benzene-1-carboxylate$", "benzoate"), (r"^benzene-1-carboxamide$", "benzamide"),
    (r"^benzene-1-carbonitrile$", "benzonitrile"), (r"^benzene-1-carbaldehyde$", "benzaldehyde"),
    (r"^benzene-1-yl$", "phenyl"),
]


def _ringloc(loc: str):
    return [int(x) for x in loc.split(",")] if re.fullmatch(r"\d+(?:,\d+)*", loc) else None


def benzenize(segs, parent):
    """segs, parent -> (segs', parent')。ベンゼン環でなければそのまま。"""
    m = _TRIENE.match(parent)
    if not m:
        return segs, parent
    rest = m.group(2)
    mm = re.match(r"^-?((?:\d+,)*\d+)-(.*)$", rest)
    if rest and not mm:
        return segs, parent
    suf_locs = [int(x) for x in mm.group(1).split(",")] if mm else []
    suf_word = mm.group(2) if mm else ""
    if rest and not suf_locs:
        return segs, parent
    ring_segs = []
    for loc, body in segs:
        ring_segs.append((_ringloc(loc), body, loc))
    ring_segs = [x for x in ring_segs]
    if any(rl is None and loc and not re.match(r"^[NOSP]", loc) for rl, _, loc in ring_segs):
        return segs, parent
    best = None
    for st in range(6):
        for d in (1, -1):
            f = lambda l, st=st, d=d: (st + d * (l - 1)) % 6 + 1
            suf = sorted(f(l) for l in suf_locs)
            pre = sorted(f(l) for rl, _, _ in ring_segs if rl for l in rl)
            order = [f(l) for rl, b, _ in sorted(ring_segs, key=lambda x: alpha_key(x[1])) if rl for l in sorted(rl)]
            key = (suf, pre, order)
            if best is None or key < best[0]:
                best = (key, f)
    f = best[1]
    new_segs = [((",".join(str(x) for x in sorted(f(l) for l in rl)) if rl else loc), b) for rl, b, loc in ring_segs]
    sl = ",".join(str(x) for x in sorted(f(l) for l in suf_locs))
    base = "benzene" + (f"-{sl}-{suf_word}" if suf_locs else "")
    for pat, rep in _BENZ_SPECIAL:
        base = re.sub(pat, rep, base)
    return new_segs, base


# 置換基の名前の言い換え(PIN の形へ)
_SUB_RENAME = [
    (r"^methylsulfinyl$", "methanesulfinyl"), (r"^methylsulfonyl$", "methanesulfonyl"),
    (r"^ethylsulfinyl$", "ethanesulfinyl"), (r"^ethylsulfonyl$", "ethanesulfonyl"),
    (r"^ethenyl$", "ethenyl"),
    (r"^prop-2-enyl$", "prop-2-en-1-yl"), (r"^prop-1-enyl$", "prop-1-en-1-yl"), (r"^prop-2-ynyl$", "prop-2-yn-1-yl"),
    (r"^but-(\d)-enyl$", r"but-\1-en-1-yl"), (r"^butan-2-yl$", "butan-2-yl"),
    (r"^(\d+(?:,\d+)*)-dimethylphenyl$", r"\1-dimethylphenyl"),
    (r"^phenylphenyl$", "[1,1'-biphenyl]-4-yl"), (r"^(\d)-phenylphenyl$", r"[1,1'-biphenyl]-\1-yl"),
    (r"^dimethylamino$", "dimethylamino"),
]


def rename_sub(body: str) -> str:
    n, core = _strip_mult(body)
    core = _unwrap(core)
    new = core
    for pat, rep in _SUB_RENAME:
        new = re.sub(pat, rep, new)
    if new == core:
        return body
    if n > 1:
        return (_MULT_C.get(n, "") + "(" + new + ")") if not _is_simple(new) else _MULT[n] + new
    return new


# 主骨格の言い換え(P-31.1.4: 置換された主骨格では末端の位置番号も書く、など)
_PARENT_RENAME = [
    (r"^ethanone$", "ethan-1-one"), (r"^ethanamine$", "ethan-1-amine"), (r"^ethanol$", "ethan-1-ol"),
    (r"^(?!meth)(\w+?)anol$", r"\1an-1-ol"), (r"^butanone$", "butan-2-one"), (r"^(\w+?)anethiol$", r"\1ane-1-thiol"),
    (r"^ethanethiol$", "ethane-1-thiol"), (r"^ethanimine$", "ethan-1-imine"), (r"^propanone$", "propan-2-one"),
    (r"^(?!meth)(\w*[a-z])anamine$", r"\1an-1-amine"),
]


def rename_parent(parent: str, substituted: bool) -> str:
    if not substituted:   # 炭素に置換基が無い(N- だけ)ときは位置番号を書かない
        return re.sub(r"^ethan-1-(amine|ol)$", r"ethan\1", parent)
    for pat, rep in _PARENT_RENAME:
        new = re.sub(pat, rep, parent)
        if new != parent:
            return new
    return parent


# ---------------------------------------------------------------- 全体

_GLOBAL = [
    # 主骨格の語尾の母音の残し方(trien-4,6-diyne -> triene-4,6-diyne)
    (r"(\d)-(di|tri|tetra)?en-(\d+(?:,\d+)*-(?:di|tri|tetra)?yne)", r"\1-\2ene-\3"),
    (r"lambda(\d)", r"λ\1"),
    (r"diazane\b", "hydrazine"), (r"\b(\d+(?:,\d+)*)-(\w+)diazane\b", r"\1-\2hydrazine"),
    (r"\balumane\b", "aluminium hydride"),
    (r"\bhydroxy\(oxo\)methanolate\b", "hydrogen carbonate"),
    (r"\b((?:\d+-)?[a-z]*?)(meth|eth|prop|but|pent|hex|hept|oct|non|dec|undec|dodec|tridec|tetradec|pentadec|hexadec|heptadec|octadec|nonadec|icos)yl\(trimethyl\)azanium\b",
     lambda m: (m.group(1) + "-" if m.group(1) and not m.group(1).endswith("-") else m.group(1)) + "N,N,N-trimethyl" + m.group(2) + "an-1-aminium"),
    (r"^1-amino-(.*?)([a-z]+)an-1-one$", r"\1\2anamide"),
    (r"\bcyclohexa-1,3,5-triene\b", "benzene"), (r"\bcyclohexa-2,4,6-trien-1-ol\b", "phenol"),
    (r"\bcyclohexa-1,3,5-triene-1-carboxylic acid\b", "benzoic acid"),
    (r"\bcyclohexa-1,3,5-triene-1-carboxamide\b", "benzamide"),
    (r"\bcyclohexa-1,3,5-triene-1-carbonitrile\b", "benzonitrile"),
    (r"\bcyclohexa-1,3,5-triene-1-carbaldehyde\b", "benzaldehyde"),
    (r"\bcyclohexa-1,3,5-trien-1-ol\b", "phenol"), (r"\bcyclohexa-1,3,5-trien-1-amine\b", "aniline"),
    (r"\bcyclohexa-(?:1,3,5|2,4,6)-trien-1-yl\b", "phenyl"),
    (r"\bbenzenesulfonic acid\b", "benzene-1-sulfonic acid"),
    (r"(?<!m)ethanoate\b", "acetate"), (r"(?<!m)ethanoyl", "acetyl"), (r"(?<!m)ethanoic acid\b", "acetic acid"),
    (r"\(phenylmethyl\)|(?<![a-z(])phenylmethyl(?![a-z)])", "benzyl"), (r"\bmethanoate\b", "formate"),
    (r"(\w+amine) oxide\b", r"\1 N-oxide"),
    (r"\bbenzo\[c\]chromen", "dibenzo[b,d]pyran"), (r"(?<=[)\]}a-z])isochromen(?:e\b|(?=-))", "-2-benzopyran"), (r"\bisochromen(?:e\b|(?=-))", "2-benzopyran"),
    (r"(?<=[)\]}a-z])chromen(?:e\b|(?=-))", "-1-benzopyran"), (r"\bchromen(?:e\b|(?=-))", "1-benzopyran"),
]


def tidy_nesting(name: str) -> str:
    """括弧の入れ子を ( [ { にする。[1,1'-biphenyl] の角括弧は残す。"""
    from n2s_stereo import tidy_brackets
    name = re.sub(r"\((\d+[a-z]?H(?:,\d+[a-z]?H)*)\)", "\x02\\1\x03", name)   # 付加水素 (2H) は入れ子として数えない
    out = tidy_brackets(name.replace("[1,1'-biphenyl]", "\x01")).replace("\x01", "[1,1'-biphenyl]")
    return out.replace("\x02", "(").replace("\x03", ")")


def _unwrap_first(name: str) -> str:
    """エステルの最初の語が丸ごと括弧に入っているとき((4-nitrophenyl) acetate)、外す。"""
    m = re.match(r"^([(\[{])(.*?)([)\]}]) (?=\S)", name)
    if m and _balanced(m.group(2)):
        return m.group(2) + name[m.end() - 1:]
    return name


def rewrites(name: str, parent_ok=None) -> list[str]:
    """PIN に近づけた候補(強い順)。呼び出し側で照合して採る。"""
    outs = []
    g = _unwrap_first(name)
    for pat, rep in _GLOBAL:
        g = re.sub(pat, rep, g)
    for base in dict.fromkeys([g, name]):
        for words in _word_variants(base, parent_ok):
            outs.append(words)
        outs.append(base)
    outs = [tidy_nesting(o) for o in outs] + outs
    return [o for o in dict.fromkeys(outs) if o != name]


def _word_variants(name: str, parent_ok):
    """空白で分かれた語(エステル・塩など)ごとに、置換基の整形を試す。"""
    words = re.sub(r" (acid|oxide)\b", r"~\1", name).split(" ")   # 'benzoic acid' は 1 語として扱う
    fixed = [_fix_fragment(w.replace("~", " "), parent_ok) for w in words]
    yield " ".join(fixed)
    # 位置番号を全部外した形(tetrachloroethene など、置換できる所が全部埋まっているとき)
    yield " ".join(_drop_locants(f) for f in fixed)


_HALO_ETH = re.compile(r"^(?:\d+,)*\d+-((?:di|tri|tetra|penta|hexa)?(?:fluoro|chloro|bromo|iodo)+eth(?:ene))$")


def _drop_locants(frag: str) -> str:
    """tetrachloroethene のように、置換できる所が全部埋まっていて位置番号が要らない形だけ。"""
    m = _HALO_ETH.match(frag)
    return m.group(1) if m else frag


def c_substituted(segs) -> bool:
    """炭素(骨格)に置換基があるか。N- / O- / S- / P- だけなら False(N,N-diethylethanamine は位置番号を書かない)。"""
    return any(not re.match(r"^[NOSP]", loc) for loc, _ in segs)


def n_locants(segs, parent):
    """尿素: 1,3 -> N,N' / ジアミン: N,N' -> N1,N4(N' だけなら N1)。"""
    if re.search(r"urea$", parent):
        def f(loc):
            return ",".join({"1": "N", "3": "N'"}.get(t, t) for t in loc.split(","))
        return [(f(loc) if re.fullmatch(r"[13](?:,[13])*", loc) else loc, b) for loc, b in segs]
    m = re.search(r"-(\d+),(\d+)-diamine$", parent)
    if m and all(re.fullmatch(r"N'*(?:,N'*)*", loc) or not re.match(r"N", loc) for loc, _ in segs):
        a, b = m.groups()
        toks = [t for loc, _ in segs for t in loc.split(",") if t.startswith("N")]
        if toks and "N" not in toks:   # N' だけのときは N1(対称なら同じ構造。照合で確かめる)
            return [(",".join(f"N{a}" if t.startswith("N") else t for t in loc.split(",")), bd) for loc, bd in segs]
        return [(",".join({"N": f"N{a}", "N'": f"N{b}"}.get(t, t) for t in loc.split(",")), bd) for loc, bd in segs]
    return segs


def _fix_fragment(frag: str, parent_ok) -> str:
    sp = split(frag, parent_ok)
    if sp is None:
        return frag
    stereo, segs, parent = sp
    if not segs:
        return frag
    segs, parent = benzenize(segs, parent)
    segs = n_locants(segs, parent)
    segs = [(loc, _fix_inner(rename_sub(body), parent_ok)) for loc, body in segs]
    segs = [(loc, enclose(body)) for loc, body in segs]
    segs = merge_same(segs)
    segs = reorder(segs)
    parent = rename_parent(parent, c_substituted(segs))
    return join(stereo, segs, parent)


def _groups(text: str):
    """深さ 0 の括弧のかたまりを (開始, 終了) で返す。"""
    d, start = 0, 0
    for i, ch in enumerate(text):
        if ch in "([{":
            if d == 0:
                start = i
            d += 1
        elif ch in ")]}":
            d -= 1
            if d == 0:
                yield start, i + 1


def _fix_inner(body: str, parent_ok) -> str:
    """括弧の中の置換基名(例: [2-chloro-4-methylphenyl])も同じように整える。入れ子は内側から。"""
    n, core = _strip_mult(body)
    wrapped = core != _unwrap(core)
    core = _unwrap(core)
    # 内側のかたまりを先に整える
    out, last = [], 0
    for i, j in _groups(core):
        out.append(core[last:i])
        out.append(_fix_group(core[i:j], parent_ok))
        last = j
    out.append(core[last:])
    core2 = "".join(out)
    sp = split(core2, parent_ok)
    if sp is not None and sp[1]:
        stereo, segs, parent = sp
        segs, parent = benzenize(segs, parent)
        segs = [(loc, enclose(rename_sub(b))) for loc, b in segs]
        core2 = join(stereo, reorder(merge_same(segs)), parent)
    if core2 == core:
        return body
    if n > 1:
        return _MULT_C.get(n, "") + "(" + core2 + ")"
    return f"({core2})" if wrapped else core2


def _fix_group(group: str, parent_ok) -> str:
    inner = _fix_inner(group, parent_ok)
    return inner if inner[:1] in "([{" else group[0] + inner + group[-1]



# ---------------------------------------------------------------- PIN らしさ

_PENALTY = [
    r"oxidanyl", r"azanyl", r"(?:brom|chlor|fluor|iod)anyl", r"cyclohexa-[\d,]+-trien", r"(?:eth|meth)ano(?:ic|ate)\b",
    r"ethanamide", r"benzenecarbo", r"bis\((?:fluoro|chloro|bromo|iodo|methyl|ethyl|hydroxy|nitro|amino|oxo|methoxy|cyano)\)",
    r"\d-(?!(?:meth|eth|prop|but|pent|hex|hept|oct|non|dec|phen)oxy)[a-z]{3,}oxy(?=-|\)|\]|\}|$)", r"tetra\(|tri\(ethyl", r"yl\)?formate", r"1-amino-[^ ]*an-1-one", r"nitrilo", r"chloroformate",
    r"(?<![\d,]-)(?<!benzo)(?<!naphtho)(?:dithiolan|dioxolan|dioxan|dithian|diazin|triazin|diazol|triazol|dioxin)", r"lambda\d", r"azanium\b", r"diazane\b", r"\balumane\b",
    r"(?<!meth)(?<!eth)anol\b", r"(?<!meth)(?<!eth)anamine\b", r"methylsulfinyl", r"methylsulfonyl", r"phenylphenyl",
    r"\d-en-\d+(?:,\d+)*-(?:di|tri)?yn", r"\(oxo\)methyl", r"oxomethyl",
    r"(?<![a-z])phenylamino",                      # 無置換のフェニルアミノは anilino(PIN)
    r"yl[)\]}]?formic acid",                      # 置換した蟻酸は安息香酸などの形(…-carboxylic acid)にする
    r"\((?:[EZ],)+[EZ]\)-",                       # 二重結合が複数あるときの E/Z は (2E,4E) と位置番号をつける
    r"(?<![\w,])\d+-O(?:,\d+-O)+-(?:di|bis)",      # 同じ基の二酸エステルに位置記号は付けない(dibutyl …-1,2-dicarboxylate)
    r"\b(?:lignocerol|cerotol|behenol|lauryl|myristyl|palmityl|cetyl|stearyl|oleyl|arachidyl|ceryl)\b",   # 長鎖アルコールの慣用名より体系名
]


def _flat(s: str) -> str:
    return s.translate(str.maketrans("[]{}", "()()"))


def _pen_frag(frag: str, parent_ok, depth: int = 0) -> int:
    sp = split(frag, parent_ok)
    if not sp:
        return 0
    parent = sp[2]
    segs = [(l, b) for l, b in sp[1] if not re.fullmatch(r"(?:di|tri|tetra|penta|hexa|hepta|octa|nona|deca)?hydro", b)]   # hydro 接頭語は英数字順に並べない
    p = 0
    bodies = [b for _, b in segs]
    if [alpha_key(b) for b in bodies] != sorted(alpha_key(b) for b in bodies):
        p += 1
    p += sum(_flat(enclose(b)) != _flat(b) for l, b in segs if l)   # 位置番号の無い置換基の括弧は、読み間違いを避けるために付けてよい(tribromo(fluoro)methane)
    cores = [_flat(_unwrap(_strip_mult(b)[1])) for b in bodies]
    if len(set(cores)) < len(cores):
        p += 1   # 同じ置換基が別々に書かれている(N,N-dimethyl-6-methyl は N,N,6-trimethyl に束ねる)
    if segs and rename_parent(parent, c_substituted(segs)) != parent:
        p += 1
    if not c_substituted(segs) and re.search(r"ethan-1-(?:amine|ol)$", parent) and segs:
        p += 1
    if depth < 3:
        for b in bodies:
            for i, j in _groups(b):
                inner = b[i + 1:j - 1]
                if inner and not inner.isalpha():
                    p += _pen_frag(inner, parent_ok, depth + 1)
    return p


def pin_penalty(name: str, parent_ok=None) -> int:
    p = sum(len(re.findall(x, name)) for x in _PENALTY)
    if re.match(r"^(?:\d+,?)+-", name) and re.search(r"methane$", name) and " " not in name:
        p += 1   # メタンには位置番号を付けない(1,1,1-tribromo-1-fluoromethane は tribromo(fluoro)methane)
    for w in re.sub(r" (acid|oxide)\b", r"~\1", name).split(" "):
        p += _pen_frag(w.replace("~", " "), parent_ok)
    return p


def locant_vector(name: str) -> tuple:
    """括弧の外にある位置番号を小さい順に並べたもの。小さいほど『最小の位置番号』の規則に合う。"""
    d, out = 0, []
    for m in re.finditer(r"(?<![\w,])(\d+)H-|([\d,]+)-|[\[\](){}]", name):
        if m.group(0) in "[({":
            d += 1
        elif m.group(0) in "])}":
            d -= 1
        elif d == 0 and m.group(1):   # 指示水素の位置(1H-pyrazole)も最小の位置番号の比較に入れる
            out.append(int(m.group(1)))
        elif d == 0 and m.group(2):
            out += [int(x) for x in m.group(2).split(",") if x.isdigit()]
    return tuple(sorted(out))
