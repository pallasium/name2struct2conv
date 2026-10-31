"""name2struct2 の中身: IUPAC 名 <-> 構造(完全オフライン)。

名前 -> 構造 : 辞書(略号・慣用名・日本語名) -> OPSIN -> 綴りの自動修正 -> 機械学習(推定、要確認)
構造 -> 名前 : Knowledgator の MT5 翻訳モデルで候補を出し、全候補を OPSIN で構造に戻して InChIKey を照合。
              立体は CIP 判定から位置番号つきで付け直し、照合を通ったものだけ採る。
外部への送信はしない(HF_HUB_OFFLINE)。
"""
from __future__ import annotations

import copy
import json
import os
import re
import threading
import unicodedata
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HOME", str(MODELS / ".hf_home"))  # C に書かない

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors

import n2s_opsin as opsin
import n2s_assemble as assemble
import n2s_hydro as hydro
import n2s_pin as pin
import n2s_pin2 as pin2
import n2s_stereo as stereo

RDLogger.DisableLog("rdApp.*")

MODEL_CANONICAL = "SMILES2IUPAC-canonical-base"   # 立体なし、公称 87%
MODEL_ISOMERIC = "SMILES2IUPAC-isomeric-small"    # @ の立体を扱える小型
MODEL_NAME2SMI = "IUPAC2SMILES-isomeric-base"     # OPSIN が読めない名前の予備(推定)
DICT_FILES = [HERE / "names_dict.tsv", HERE / "user_dict.tsv"]


# ================================================================ 結果

@dataclass
class Result:
    query: str
    direction: str                 # "名前→構造" / "構造→名前"
    ok: bool = False
    name: str = ""                 # IUPAC 名(構造→名前では一番良い候補)
    level: str = ""                # 一致 / 立体以外一致 / 未検証 / OPSIN / 辞書 / 利用者辞書 / 推定 / 失敗
    mol: Chem.Mol | None = None
    info: dict = field(default_factory=dict)
    candidates: list = field(default_factory=list)   # (名前 or SMILES, 判定)
    aliases: list = field(default_factory=list)      # 辞書にある略号・慣用名
    locants: dict = field(default_factory=dict)      # 原子番号 -> IUPAC の位置番号
    message: str = ""
    confidence: str = ""           # rule(PIN の規則で組んだ) / model(モデルの名前そのまま) / non-pin(PIN でない可能性)
    rules: list = field(default_factory=list)        # 適用した規則の ID(n2s_pin2.RULES / G00…、assemble:*、hydro)
    pool: list = field(default_factory=list)         # 選ぶときの候補と特徴(評価・選び方の調整用。row() には出さない)

    def row(self) -> dict:
        d = {"input": self.query, "direction": self.direction, "iupac_name": self.name,
             "check": self.level, "pin": self.confidence, "rules": " ".join(self.rules),
             "aliases": " / ".join(self.aliases), "message": self.message}
        d.update(self.info)
        return d


def inchikey(mol) -> str:
    return stereo.inchikey(mol)


def flat_key(mol) -> str:
    """立体だけを消した InChIKey(同位体などは残る)。"""
    if mol is None:
        return ""
    m = Chem.Mol(mol)
    Chem.RemoveStereochemistry(m)
    return inchikey(m)


def fixedh_key(mol) -> str:
    """互変異性体(動く水素の位置)まで区別する InChIKey。作れなければ空。"""
    if mol is None:
        return ""
    try:
        inchi = Chem.MolToInchi(mol, options="/FixedH")
        return (Chem.InchiToInchiKey(inchi) or "") if inchi else ""
    except Exception:
        return ""


def mol_info(mol: Chem.Mol) -> dict:
    try:
        inchi = Chem.MolToInchi(mol) or ""
    except Exception:
        inchi = ""
    return {
        "smiles": Chem.MolToSmiles(mol),
        "inchi": inchi,
        "inchikey": inchikey(mol),
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "mw": round(Descriptors.MolWt(mol), 3),
        "exact_mass": round(Descriptors.ExactMolWt(mol), 4),
    }


# ================================================================ 入力の判定

def parse_smiles(text: str) -> Chem.Mol | None:
    text = text.strip()
    if not text or " " in text or len(text) > 2000:
        return None
    return Chem.MolFromSmiles(text)


def parse_structure(text: str) -> tuple[str, Chem.Mol | None]:
    """構造として読めるか: ('InChI'|'MOL'|'SMILES'|'', mol)"""
    t = text.strip()
    if t.startswith("InChI="):
        return "InChI", Chem.MolFromInchi(t)
    if "V2000" in t or "V3000" in t or "M  END" in t:
        # MOL の 1 行目(タイトル)は空のことがあるので、先頭の改行は残す
        return "MOL", Chem.MolFromMolBlock(text.rstrip() + "\n", sanitize=True)
    if dictionary().lookup(t):   # DMF・TEA など辞書にある語は名前として扱う
        return "", None
    m = parse_smiles(t)
    if m is not None and not _looks_like_word(t):
        return "SMILES", m
    return "", None


def _looks_like_word(t: str) -> bool:
    """SMILES としても読める英単語('Br' は SMILES、'Brine' のようなもの)を名前扱いにする。"""
    return bool(re.fullmatch(r"[A-Z][a-z]{2,}", t))


def read_file(path: Path) -> list[tuple[str, Chem.Mol]]:
    """MOL / SDF / CDXML / SMI を読む -> [(タイトル, mol)]"""
    suf = path.suffix.lower()
    if suf in (".mol", ".sdf", ".sd"):
        sup = Chem.SDMolSupplier(str(path))
        out = [(m.GetProp("_Name") if m.HasProp("_Name") else "", m) for m in sup if m is not None]
    elif suf == ".cdxml":
        from rdkit.Chem import rdChemReaders
        out = [("", m) for m in rdChemReaders.MolsFromCDXMLFile(str(path)) if m is not None]
    elif suf in (".smi", ".txt"):
        out = []
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            parts = line.split()
            if parts:
                m = parse_smiles(parts[0])
                if m is not None:
                    out.append((" ".join(parts[1:]), m))
    else:
        raise ValueError(f"読めない形式です: {path.suffix}")
    return out


# ================================================================ 辞書

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).replace("・", "·")
    return re.sub(r"\s+", "", s).casefold()


def _is_smiles_entry(t: str) -> bool:
    """辞書の右側が SMILES か(利用者辞書では名前の代わりに SMILES を書いてよい)。"""
    return " " not in t and not re.search(r"[a-z]{3,}", t) and Chem.MolFromSmiles(t) is not None


class Dictionary:
    def __init__(self):
        self.alias: dict[str, tuple[str, str]] = {}   # 正規化した別名 -> (名前 or SMILES, 出典)
        self.labels: dict[str, list[str]] = {}       # 名前 or SMILES -> 別名(表示用)
        self._by_key: dict[str, dict] | None = None
        self._lock = threading.Lock()
        for f in DICT_FILES:
            if not f.exists():
                continue
            src = "利用者辞書" if f.name.startswith("user") else "辞書"
            for line in f.read_text(encoding="utf-8-sig").splitlines():
                if not line.strip() or line.startswith("#"):
                    continue
                p = line.split("\t")
                if len(p) < 2:
                    continue
                names = [a.strip() for a in p[0].split("|") if a.strip()]
                target = p[1].strip()
                for a in names:
                    self.alias[_norm(a)] = (target, src)
                self.labels.setdefault(target, [])
                self.labels[target] += [a for a in names if a not in self.labels[target]]

    def lookup(self, text: str):
        return self.alias.get(_norm(text))

    def _entry(self, key: str) -> dict:
        with self._lock:
            if self._by_key is None:
                self._by_key = self._build_index()
        return self._by_key.get(key, {})

    def aliases_for(self, key: str) -> list[str]:
        """InChIKey -> 辞書の別名(略号・日本語名)。"""
        return self._entry(key).get("aliases", [])

    def names_for(self, key: str) -> list[str]:
        """InChIKey -> 辞書の英語名(構造→名前の候補にも使う)。"""
        return self._entry(key).get("names", [])

    def _build_index(self) -> dict[str, dict]:
        """初回だけ全項目を構造にする(結果は models/.dict_index.json に保存)。"""
        cache = MODELS / ".dict_index.json"
        stamp = [f.stat().st_mtime for f in DICT_FILES if f.exists()] + [2]
        try:
            d = json.loads(cache.read_text(encoding="utf-8"))
            if d.get("stamp") == stamp:
                return d["index"]
        except Exception:
            pass
        targets = list(self.labels)
        mols = [Chem.MolFromSmiles(t) if _is_smiles_entry(t) else None for t in targets]
        need = [t for t, m in zip(targets, mols) if m is None]
        parsed = dict(zip(need, opsin.parse(need)))
        index: dict[str, dict] = {}
        for t, m in zip(targets, mols):
            m = m or (parsed[t].mol if t in parsed else None)
            k = inchikey(m)
            if k:
                e = index.setdefault(k, {"aliases": [], "names": []})
                e["aliases"] += [a for a in self.labels[t] if a not in e["aliases"]]
                if not _is_smiles_entry(t) and t not in e["names"]:
                    e["names"].append(t)
        try:
            cache.write_text(json.dumps({"stamp": stamp, "index": index}, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass
        return index


_dict: Dictionary | None = None
_dict_lock = threading.Lock()


def dictionary() -> Dictionary:
    global _dict
    with _dict_lock:   # GUI の先読みと変換のスレッドが同時に作らないように
        if _dict is None:
            _dict = Dictionary()
        return _dict


def reload_dictionary():
    global _dict
    with _dict_lock:
        _dict = None


# ================================================================ 翻訳モデル

class _Translator:
    """Knowledgator の MT5(エンコーダとデコーダで語彙が別)を標準の transformers で動かす。"""

    def __init__(self, model_dir: Path, src: str, tgt: str):
        import torch
        from torch import nn
        from transformers import MT5Config, MT5ForConditionalGeneration, PreTrainedTokenizerFast

        cfg = MT5Config.from_pretrained(model_dir)
        enc_vocab, dec_vocab = cfg.encoder_vocab_size, cfg.decoder_vocab_size
        cfg = copy.deepcopy(cfg)
        cfg.vocab_size = dec_vocab
        cfg.tie_word_embeddings = False
        model = MT5ForConditionalGeneration(cfg)
        model.encoder.embed_tokens = nn.Embedding(enc_vocab, cfg.d_model)

        weights = model_dir / "model.safetensors"
        if weights.exists():
            from safetensors.torch import load_file
            sd = dict(load_file(weights))
        else:
            sd = dict(torch.load(model_dir / "pytorch_model.bin", map_location="cpu", weights_only=True))
        sd["encoder.embed_tokens.weight"] = sd.pop("encoder_embedding.weight")
        dec = sd.pop("decoder_emebedding.weight")  # 元モデルの綴りのまま
        sd["decoder.embed_tokens.weight"] = dec
        sd["shared.weight"] = dec
        missing, unexpected = model.load_state_dict(sd, strict=False)
        bad = [k for k in missing if "relative_attention_bias" not in k]
        if bad or unexpected:
            raise RuntimeError(f"重みの読み込みに失敗: missing={bad} unexpected={unexpected}")

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            torch.set_num_threads(os.cpu_count() or 1)
        threads = os.environ.get("N2S_THREADS", "")   # 評価を並列で回すとき、プロセスごとのスレッド数を絞る
        if threads.isdigit() and int(threads) > 0:
            torch.set_num_threads(int(threads))
        self.model = model.eval().to(self.device)
        self.enc_vocab = enc_vocab
        self.src_tok = PreTrainedTokenizerFast.from_pretrained(MODELS / src)
        self.tgt_tok = PreTrainedTokenizerFast.from_pretrained(MODELS / tgt)
        self.torch = torch
        self.lock = threading.Lock()

    def can_encode(self, text: str) -> bool:
        ids = self.src_tok(text)["input_ids"]
        return len(ids) <= 160 and all(i < self.enc_vocab and i != self.src_tok.unk_token_id for i in ids)

    def translate(self, texts: list[str], n: int, max_new: int) -> list[list[tuple[str, float]]]:
        """各入力について (訳, ビームのスコア) を n 個。"""
        if not texts:
            return []
        with self.lock, self.torch.no_grad():
            enc = self.src_tok(texts, return_tensors="pt", padding=True, truncation=True,
                               max_length=160).to(self.device)
            out = self.model.generate(
                input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                max_new_tokens=max_new, num_beams=n, num_return_sequences=n,
                output_scores=True, return_dict_in_generate=True,
            )
        names = [self.tgt_tok.decode(o, skip_special_tokens=True).strip() for o in out.sequences]
        scores = out.sequences_scores.tolist()
        return [list(zip(names[i * n:(i + 1) * n], scores[i * n:(i + 1) * n])) for i in range(len(texts))]


_translators: dict[str, _Translator] = {}
_tlock = threading.Lock()
_TOKENIZERS = {MODEL_CANONICAL: ("SMILES-FAST-TOKENIZER", "IUPAC-FAST-TOKENIZER"),
               MODEL_ISOMERIC: ("SMILES-FAST-TOKENIZER", "IUPAC-FAST-TOKENIZER"),
               MODEL_NAME2SMI: ("IUPAC-FAST-TOKENIZER", "SMILES-FAST-TOKENIZER")}


def translator(name: str) -> _Translator:
    with _tlock:  # GUI の先読みと変換が同時に読み込まないように
        if name not in _translators:
            d = MODELS / name
            if not d.exists():
                raise RuntimeError(f"モデル {d} がありません。python setup_assets.py を実行してください。")
            _translators[name] = _Translator(d, *_TOKENIZERS[name])
        return _translators[name]


def warmup():
    opsin.server()
    dictionary().aliases_for("")
    for m in (MODEL_CANONICAL, MODEL_ISOMERIC):
        translator(m)


# ================================================================ 位置番号(構造図の注記用)

def locants_for(name: str, mol: Chem.Mol) -> dict[int, str]:
    """名前を OPSIN で読んだ CML の位置番号を、mol の原子番号に対応させる。"""
    try:
        r = opsin.parse([name], lenient=True, cml=True)[0]
        cmol = opsin.cml_to_mol(r.cml) if r.ok else None
        if cmol is None:
            return {}
        flat = Chem.Mol(mol)
        Chem.RemoveStereochemistry(flat)
        match = cmol.GetSubstructMatch(flat)
        if not match:
            return {}
        out = {}
        for i, j in enumerate(match):
            loc = opsin.numeric_locant(cmol.GetAtomWithIdx(j))
            if loc:
                out[i] = loc
        return out
    except Exception:
        return {}


# ================================================================ 名前 -> 構造

_FIXES = [
    (r"[‐‑‒–—−]", "-"), (r"[′’‘`´]", "'"), (r"[″”“]", "''"), (r"\s*-\s*", "-"), (r"\s*,\s*", ","),
    (r"\(\s+", "("), (r"\s+\)", ")"),
]


def _variants(name: str) -> list[str]:
    """綴りの揺れを直した候補(全角・ダッシュ・空白・大文字・acid の書き忘れ)。"""
    s = unicodedata.normalize("NFKC", name).strip()
    out = [s]
    for pat, rep in _FIXES:
        s = re.sub(pat, rep, s)
    out += [s, s.lower(), re.sub(r"(?i)\b(\w+ic)$", r"\1 acid", s)]
    return [v for v in dict.fromkeys(out) if v and v != name]


def _target_mol(target: str):
    return Chem.MolFromSmiles(target) if _is_smiles_entry(target) else opsin.parse([target])[0].mol


def name_to_struct(text: str, ml_fallback: bool = True) -> Result:
    text = text.strip()
    res = Result(text, "名前→構造")
    d = dictionary()

    for v in [text] + _variants(text):
        hit = d.lookup(v)
        if hit:
            mol = _target_mol(hit[0])
            if mol is not None:
                name = "" if _is_smiles_entry(hit[0]) else hit[0]
                if not name:  # 利用者辞書に SMILES で書いたもの: 名前は構造から付ける
                    return _named_from_structure(res, mol, hit[1], f"{hit[1]}: {text} = {hit[0]}")
                return _struct_result(res, mol, name, hit[1], f"{hit[1]}: {text} = {name}")

    r = opsin.parse([text])[0]
    if r.ok:
        return _struct_result(res, r.mol, text, "OPSIN", _warn(r.warning))
    r2 = opsin.parse([text], lenient=True)[0]
    if r2.ok:
        return _struct_result(res, r2.mol, text, "OPSIN",
                              "名前の立体記号の一部を読めなかったので無視しました。立体を確認してください。 "
                              + _warn(r2.warning))
    for v in _variants(text):
        rv = opsin.parse([v])[0]
        if rv.ok:
            return _struct_result(res, rv.mol, v, "OPSIN", f"「{v}」に直して読みました。")
    ra = opsin.parse([text], acids=True)[0]
    if ra.ok:
        return _struct_result(res, ra.mol, text, "OPSIN", "「acid」を補って読みました。")

    res.level = "失敗"
    res.message = "OPSIN で読めませんでした: " + _explain(r.message)
    fixes = _spell_fix(text, r.message)
    guess = _ml_guess(text) if ml_fallback else []
    guess_keys = {inchikey(m) for m in guess}
    if fixes:
        # 綴りの直し方が複数あれば、機械学習の推定に一番似た構造になるものを優先
        fixes.sort(key=lambda f: (inchikey(f[1]) not in guess_keys, _rarity(f[0]), -_similarity(f[1], guess)))
        v, mol = fixes[0]
        out = _struct_result(res, mol, v, "推定", f"綴りを直して「{v}」として読みました(要確認)。")
        out.candidates = [(fv, "綴り修正") for fv, _ in fixes[:10]]
        return out
    if guess:
        best = guess[0]
        res.mol, res.info, res.level = best, mol_info(best), "推定"
        res.candidates = [(Chem.MolToSmiles(m), "推定") for m in guess]
        back = smiles_to_name(best, beams=6)
        tip = f" この構造の検証済みの名前は「{back.name}」。入力と見比べてください。" if back.level == "一致" else ""
        res.message += " 機械学習による推定構造を表示しています(誤りが多いので要確認)。" + tip
    return res


_iupac_tok = None   # None = まだ読んでいない / False = 読めなかった(モデル未導入など)


def _rarity(name: str) -> float:
    """名前のありふれなさ。IUPAC 語彙の番号(よく出る語ほど小さい)の対数の平均。
    語彙(モデルの tokenizer)が無いときは 0(順位付けに使わない)。綴りの修正自体は機械学習なしで動く。"""
    global _iupac_tok
    import math
    if _iupac_tok is None:
        try:
            from transformers import PreTrainedTokenizerFast
            _iupac_tok = PreTrainedTokenizerFast.from_pretrained(MODELS / "IUPAC-FAST-TOKENIZER")
        except Exception:
            _iupac_tok = False
    if _iupac_tok is False:
        return _rarity_simple(name)
    ids = _iupac_tok(name, add_special_tokens=False)["input_ids"]
    return sum(math.log(i + 2) for i in ids) / max(len(ids), 1)


# 語彙が無いときの代わり: IUPAC 名によく出る語(長いものから当てる)
_COMMON = re.compile(
    r"methyl|ethyl|propyl|butyl|pentyl|hexyl|phenyl|benzyl|benzo|benz|hydroxy|amino|imino|nitro|cyano|chloro|fluoro|bromo|iodo"
    r"|methoxy|ethoxy|carboxy|carbonyl|carbox|amide|amine|acid|oxo|oxy|acet|form|meth|eth|prop|but|pent|hex|hept|oct|non|dec"
    r"|cyclo|pyridin|pyrrol|furan|thiophen|indol|quinolin|azol|azin|imidazol|oxazol|thiazol|naphthalen|sulfon|sulfan|phosph"
    r"|di|tri|tetra|yl|ol|one|al|ane|ene|yne|ate|ic|an|en|yn")


def _rarity_simple(name: str) -> float:
    """名前のうち、よく出る語で覆えない文字の割合(0 = 全部ありふれた語)。"""
    letters = re.sub(r"[^a-z]", "", name.lower())
    if not letters:
        return 1.0
    covered = sum(len(m) for m in _COMMON.findall(letters))
    return 1.0 - covered / len(letters)


def _similarity(mol, others) -> float:
    from rdkit import DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    if not others:
        return 0.0
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp = gen.GetFingerprint(mol)
    return max(DataStructs.TanimotoSimilarity(fp, gen.GetFingerprint(o)) for o in others)


def _spell_fix(text: str, msg: str, limit: int = 3000) -> list[tuple[str, Chem.Mol]]:
    """読めなかった部分の周りで 1 文字の誤り(抜け・余分・入れ替え・取り違え)を総当たりし、OPSIN が読めるものを返す。"""
    bad = _bad_part(msg)
    i = text.find(bad) if bad else -1
    lo, hi = (max(0, i - 6), min(len(text), i + max(len(bad), 1) + 2)) if i >= 0 else (0, len(text))
    letters = "abcdefghijklmnopqrstuvwxyz-,()[] '0123456789"
    seen, cands = {text}, []

    def add(v):
        if v not in seen:
            seen.add(v)
            cands.append(v)
    for k in range(lo, hi + 1):
        if k < len(text):
            add(text[:k] + text[k + 1:])                                   # 余分
            if k + 1 < len(text):
                add(text[:k] + text[k + 1] + text[k] + text[k + 2:])       # 入れ替え
            for ch in letters:
                add(text[:k] + ch + text[k + 1:])                          # 取り違え
        for ch in letters:
            add(text[:k] + ch + text[k:])                                  # 抜け
        if len(cands) > limit:
            break
    out, keys = [], set()
    for v, r in zip(cands, opsin.parse(cands)):
        m = r.mol
        k = inchikey(m)
        if m is not None and k not in keys:
            keys.add(k)
            out.append((v, m))
    return out


def _warn(w: str) -> str:
    return f"OPSIN の注意: {w}" if w else ""


def _bad_part(msg: str) -> str:
    """OPSIN の失敗メッセージから、読めなかった部分を取り出す。"""
    parts = [p.strip() for p in re.split(r":\s", msg) if p.strip()]
    if len(parts) < 2:
        return ""
    last = parts[-1]
    return re.split(r"\s{2,}|\s(?=The following)", last)[0].strip()


def _explain(msg: str) -> str:
    bad = _bad_part(msg)
    return f"「{bad}」の部分が読めません。" if bad else msg


def _struct_result(res: Result, mol, name: str, level: str, msg: str) -> Result:
    res.ok, res.mol, res.name, res.level, res.message = True, mol, name, level, msg.strip()
    res.info = mol_info(mol)
    res.aliases = dictionary().aliases_for(res.info["inchikey"])
    res.locants = locants_for(name, mol)
    return res


def _named_from_structure(res: Result, mol, level: str, msg: str) -> Result:
    r = smiles_to_name(mol)
    r.query, r.direction, r.level = res.query, "名前→構造", level
    r.ok = True
    r.message = msg + (f"  IUPAC 名は構造から付けました({'検証済み' if r.name else '付けられず'})。")
    return r


def _ml_guess(text: str) -> list[Chem.Mol]:
    """OPSIN が読めない名前を機械学習で推定(誤りが多いので候補として出すだけ)。"""
    try:
        tr = translator(MODEL_NAME2SMI)
    except Exception:
        return []
    out, seen = [], set()
    for smi, _ in tr.translate([text], 5, 200)[0]:
        smi = re.sub(r"^<[A-Z]+>", "", smi)
        frags = list(dict.fromkeys(f for f in smi.split(".") if f))  # 同じ成分の繰り返しを 1 つに
        m = parse_smiles(".".join(frags))
        if m is not None and Chem.MolToSmiles(m) not in seen:
            seen.add(Chem.MolToSmiles(m))
            out.append(m)
    return out


# ================================================================ 構造 -> 名前

# 正しいが好まれない書き方: 置換基の親水素化物名、ベンゼン環を cyclohexatriene と呼ぶもの、
# acetic/formic acid などの保存名を使わないもの
_UNUSUAL = re.compile(
    r"oxidanyl|azanyl|(?:brom|chlor|fluor|iod)anyl|cyclohexa-?[\d,]*-?trien"
    r"|\b(?:eth|meth)ano(?:ic|ate)|ethanamide|benzenecarbo|lambda\d|oxido\(oxo\)|hydroxy\(oxo\)")
_RANK = {"一致": 0, "互変異性体違い": 1, "立体以外一致": 2, "別の構造": 3, "OPSIN で読めない": 4}

# 単原子イオンなど、モデルに頼らない名前
_IONS = {"[Na+]": "sodium", "[K+]": "potassium", "[Li+]": "lithium", "[Cs+]": "caesium", "[Rb+]": "rubidium",
         "[NH4+]": "ammonium", "[Mg+2]": "magnesium", "[Ca+2]": "calcium", "[Zn+2]": "zinc", "[Ba+2]": "barium",
         "[Cu+2]": "copper(2+)", "[Fe+3]": "iron(3+)", "[Fe+2]": "iron(2+)", "[Ag+]": "silver",
         "[F-]": "fluoride", "[Cl-]": "chloride", "[Br-]": "bromide", "[I-]": "iodide", "[OH-]": "hydroxide",
         "F[P-](F)(F)(F)(F)F": "hexafluorophosphate", "F[B-](F)(F)F": "tetrafluoroborate",
         "O=Cl(=O)(=O)[O-]": "perchlorate", "O=[N+]([O-])[O-]": "nitrate", "O": "water"}


def _kekule(mol: Chem.Mol, stereo_: bool, seed: int | None = None) -> str:
    """モデルの語彙に芳香族の小文字が無いので、ケクレ形で渡す。seed があれば原子の順番を(再現できる形で)並べ替える。"""
    import random as _random
    m = Chem.Mol(mol)
    if seed is not None:
        order = list(range(m.GetNumAtoms()))
        _random.Random(seed).shuffle(order)
        m = Chem.RenumberAtoms(m, order)
    Chem.Kekulize(m, clearAromaticFlags=True)
    return Chem.MolToSmiles(m, isomericSmiles=stereo_, kekuleSmiles=True, canonical=seed is None)


def _judge(names: list[str], key: str, fkey: str, mol: Chem.Mol | None = None) -> list[tuple[str, str]]:
    out = []
    hkey = fixedh_key(mol) if mol is not None else ""
    for c, r in zip(names, opsin.parse(names)):
        bm = r.mol
        if bm is None:
            out.append((c, "OPSIN で読めない"))
        elif inchikey(bm) == key:
            # 標準 InChI は動く水素を同一視するので、FixedH で互変異性体の違いを見る
            bk = fixedh_key(bm) if hkey else ""
            out.append((c, "互変異性体違い" if hkey and bk and bk != hkey else "一致"))
        elif flat_key(bm) == fkey:
            out.append((c, "立体以外一致"))
        else:
            out.append((c, "別の構造"))
    return out


def _name_single(mol: Chem.Mol, beams: int) -> tuple[list[tuple[str, str]], dict[str, float]]:
    """1 成分の名前候補と判定。"""
    key, fkey = inchikey(mol), flat_key(mol)
    can = Chem.MolToSmiles(mol)
    if can in _IONS:
        return [(_IONS[can], "一致")], {_IONS[can]: 0.0}
    has_stereo = can != Chem.MolToSmiles(mol, isomericSmiles=False)
    score: dict[str, float] = {}

    def run(seed: int | None, n: int = beams) -> list[str]:
        new = []
        jobs = {MODEL_CANONICAL: [s + _kekule(mol, False, seed) for s in ("<SYST>", "<BASE>")]}
        if has_stereo:
            jobs[MODEL_ISOMERIC] = [s + _kekule(mol, True, seed) for s in ("<SYST>", "<BASE>")]
        if seed is not None:  # 再挑戦では小さいモデルにも立体なしで聞いて候補を増やす
            jobs.setdefault(MODEL_ISOMERIC, [])
            jobs[MODEL_ISOMERIC] += [s + _kekule(mol, False, seed) for s in ("<SYST>", "<BASE>")]
        for model_name, texts in jobs.items():
            tr = translator(model_name)
            texts = [t for t in texts if tr.can_encode(t)]  # '/' '\' はどちらのモデルにも無い
            for outs in tr.translate(texts, n, 200):
                for c, sc in outs:
                    if not c:
                        continue
                    if c not in score:
                        new.append(c)
                    score[c] = max(sc, score.get(c, -1e9))
        return new

    known = dictionary().names_for(key)   # 辞書の英語名(安息香酸 -> benzoic acid など)
    for k in known:
        score[k] = 0.0
    judged = _judge(known + run(None), key, fkey, mol)
    if not any(lv == "一致" for _, lv in judged):
        for i in range(6):  # SMILES の書き方(原子の順番)を変えて再挑戦。後半はビームを広げる
            judged += _judge(run(i + 1, beams if i < 3 else beams * 2), key, fkey, mol)
            if any(lv in ("一致", "互変異性体違い", "立体以外一致") for _, lv in judged):
                break

    # 骨格が合う候補に立体を付け直す(位置番号つき -> だめなら位置番号なしの総当たり)
    if not any(lv == "一致" for _, lv in judged):
        near = sorted({c for c, lv in judged if lv == "立体以外一致"}, key=lambda c: (_style(c), -score[c]))
        for c in near[:5]:
            fixed = stereo.assign(c, mol) or stereo.brute(c, mol)
            if fixed:
                score[fixed] = score[c]
                judged += _judge([fixed], key, fkey, mol)
                if judged[-1][1] == "一致":
                    break
    return judged, score


_KEEP_HYPHEN = re.compile(r"(?:^|[\s(\[{,-])(?:tert|sec|cis|trans|endo|exo|syn|anti|rac|rel|meso|alpha|beta|gamma"
                          r"|delta|omega|ortho|meta|para|o|m|p|[A-Z][a-z]?|lambda\d*|[a-z]-[a-z])$")


# 同じ構造を表す、より普通の書き方への言い換え(どれも OPSIN で同じ構造に戻るときだけ使う)
_REWRITE = [
    (r"\b1-oxoethoxy\b", "acetyloxy"), (r"\(1-oxoethyl\)", "acetyl"), (r"\b1-oxoethyl\b", "acetyl"),
    (r"bis\(oxidanyl\)", "dihydroxy"), (r"tris\(oxidanyl\)", "trihydroxy"), (r"oxidanylidene", "oxo"),
    (r"oxidanyl", "hydroxy"), (r"bis\(azanyl\)", "diamino"), (r"azanyl", "amino"),
    (r"(brom|chlor|fluor|iod)anyl", r"\1o"),
    (r"\[\((\w+?)amino\)-?\(?oxo\)?methyl\]", r"(\1carbamoyl)"),
    (r"\[hydroxy\(oxo\)methyl\]", "carboxy"), (r"\(hydroxycarbonyl\)", "carboxy"),
    (r"\b(\d+)-N,(\d+)-N-", r"N\1,N\2-"), (r"\b(\d+)-N-", r"N\1-"),
    (r"\bethanoic acid\b", "acetic acid"), (r"\bethanoate\b", "acetate"), (r"\bethanamide\b", "acetamide"),
    (r"\bmethanoic acid\b", "formic acid"), (r"\bmethanoate\b", "formate"),
    (r"\bcyclohexa-(?:2,4,6|1,3,5)-trien-1-yl\b", "phenyl"), (r"\bcyclohexa-2,4,6-trien-1-ol\b", "phenol"),
]


def rewrite_name(name: str) -> str:
    for pat, rep in _REWRITE:
        name = re.sub(pat, rep, name)
    return name


def tidy_name(name: str) -> str:
    """見た目だけ整える: 言い換え、'phenyl-propanoyl' の不要なハイフン、囲みの入れ子を ( [ { にする。"""
    name = rewrite_name(name)
    out, i = [], 0
    for m in re.finditer(r"(?<=[a-z])-(?=[a-z])", name):
        left = name[:m.start()]
        if _KEEP_HYPHEN.search(left) or re.search(r"\d'*-[a-z]+$", left) is None and re.search(r"[a-z]{3,}$", left) is None:
            continue
        out.append(name[i:m.start()])
        i = m.end()
    out.append(name[i:])
    return stereo.tidy_brackets("".join(out))


def _tidy_verified(cands: list[tuple[str, str]], key: str) -> list[tuple[str, str]]:
    """一致した上位候補を整え、整えた形も OPSIN で同じ構造に戻るときだけ置き換える。"""
    tops = [c for c, lv in cands[:8] if lv == "一致"]
    tries = []   # (元の名前, 整えた形) を、整え方の強い順に
    for c in tops:
        for t in (tidy_name(c), stereo.tidy_brackets(rewrite_name(c)), rewrite_name(c), stereo.tidy_brackets(c)):
            if t != c and (c, t) not in tries:
                tries.append((c, t))
    ok: dict[str, str] = {}
    for (c, t), r in zip(tries, opsin.parse([t for _, t in tries])):
        if c not in ok and r.mol is not None and inchikey(r.mol) == key:
            ok[c] = t
    return [(ok.get(c, c), lv) for c, lv in cands]


_part_cache: dict[str, tuple] = {}


def _name_part(m: Chem.Mol) -> tuple[list[tuple[str, str]], dict[str, float]]:
    """部品(エステルの酸・アルコール側など)の名前。メチル・フェニルのような同じ部品は、一括変換で何度も出るので覚えておく(ビーム 4)。"""
    k = Chem.MolToSmiles(m)
    if k not in _part_cache:
        if len(_part_cache) >= 2000:
            _part_cache.clear()
        _part_cache[k] = _name_single(m, 4)
    judged, score = _part_cache[k]
    return list(judged), dict(score)


def _pinify(cands: list[tuple[str, str]], mol: Chem.Mol, key: str) -> tuple[list[tuple[str, str]], dict]:
    """一致した上位候補を PIN の書き方に書き換え(n2s_pin)、OPSIN で互変異性体まで同じ構造に戻るものだけを採り、
    PIN らしさ(pin_penalty)が最小のものを先頭にする。同点なら元の順。"""
    tops = [c for c, lv in cands[:8] if lv == "一致"]
    meta = {"confidence": "", "rules": []}
    if not tops:
        return cands, meta
    try:
        hkey = fixedh_key(mol)
        po = pin.parent_checker(lambda xs: [r.mol is not None for r in opsin.parse(xs, lenient=True)])
        parts, extra = [], []
        if mol.GetNumAtoms() < 80 and len(Chem.GetMolFrags(mol)) == 1:   # 部分に分けて名付け、組み立てる(エステル・リンの酸のエステル)
            def part_names(m):
                judged, score = _name_part(m)
                return [n for n, lv in _order(judged, score) if lv == "一致"]
            parts = assemble.ester_names(mol, part_names) + assemble.phosphorus_names(mol, part_names)
            extra = assemble.amide_names(mol, part_names) + assemble.urea_names(mol, part_names)   # 実績が浅いので最優先にはしない
        tries = list(dict.fromkeys(v for c in parts + tops + extra for v in [c] + pin2.rewrites(c, po)))
        ok = [v for v, r in zip(tries, opsin.parse(tries))
              if r.mol is not None and inchikey(r.mol) == key and (not hkey or fixedh_key(r.mol) in ("", hkey))]
        seeds = list(dict.fromkeys(ok[:2] + tops[:2]))   # 書き換え後・元の上位に、構造から決めた hydro / 指示水素を付ける
        hv = list(dict.fromkeys(v for sd in seeds for v in hydro.hydro_names(sd, mol, po)))
        hok = [v for v, r in zip(hv, opsin.parse(hv))
               if r.mol is not None and inchikey(r.mol) == key and (not hkey or fixedh_key(r.mol) in ("", hkey))] if hv else []
        built = [v for v in ok if v in parts or any(v in pin2.rewrites(p, po) for p in parts)]
        pool = list(dict.fromkeys(built + hok + tops + ok))   # extra 由来の名前は ok に入る(照合を通ったときだけ)
        flat_tops = {pin._flat(t) for t in tops}   # 括弧の種類だけが違う書き換えは優先しない
        meta["pool"] = [{"name": x, "pen": pin.pin_penalty(x, po), "src": ("built" if x in built else "") + ("hydro" if x in hok else "")
                         + ("top" if x in tops else "") + ("rw" if (x in ok and x not in tops) else ""),
                         "rank": tops.index(x) if x in tops else -1, "lv": list(pin.locant_vector(x)),
                         "flat": pin._flat(x) in flat_tops, "lam": "λ" in x} for x in pool]
        cands_ = built or pool
        if built and min(pin.pin_penalty(x, po) for x in built) > min(pin.pin_penalty(x, po) for x in pool):
            cands_ = pool     # 組み立てた名前より PIN らしい(penalty の小さい)名前があれば、そちらも選べるようにする
        best = min(cands_, key=lambda x: (pin.pin_penalty(x, po), 0 if ((x in ok or x in hok) and x not in tops and 'λ' not in x
                                                                     and pin._flat(x) not in flat_tops) else 1,
                                          pin.locant_vector(x), pool.index(x)))
        penalty = pin.pin_penalty(best, po)
        if best in built:
            meta["rules"].append("assemble")
        if best in hok:
            meta["rules"].append("hydro")
        if best not in tops:
            meta["rules"] += pin2.applied_rules(tops[0], best, po)
        meta["confidence"] = "non-pin" if penalty else ("rule" if meta["rules"] else "model")
    except Exception as e:
        meta["confidence"], meta["error"] = "", f"{type(e).__name__}: {e}"   # 理由を残す(握りつぶさない)
        return cands, meta
    if best == cands[0][0]:
        return cands, meta
    return [(best, "一致")] + [(c, lv) for c, lv in cands if c != best], meta


def _style(name: str) -> int:
    return len(_UNUSUAL.findall(name))


def _order(judged, score):
    uniq: dict[str, str] = {}
    for c, lv in judged:
        if c not in uniq or _RANK[lv] < _RANK[uniq[c]]:
            uniq[c] = lv
    return sorted(uniq.items(), key=lambda x: (_RANK[x[1]], _style(x[0]), -score.get(x[0], -1e9)))


_MULT = {2: "di", 3: "tri", 4: "tetra", 5: "penta", 6: "hexa"}
_MULT2 = {2: "bis", 3: "tris", 4: "tetrakis", 5: "pentakis", 6: "hexakis"}


def smiles_to_name(smiles, beams: int = 8) -> Result:
    """smiles: SMILES 文字列か RDKit Mol。"""
    mol = smiles if isinstance(smiles, Chem.Mol) else parse_smiles(smiles)
    query = smiles if isinstance(smiles, str) else Chem.MolToSmiles(smiles)
    res = Result(query, "構造→名前")
    if mol is None:
        res.level, res.message = "失敗", "構造として読めません。"
        return res
    res.mol, res.info = mol, mol_info(mol)
    res.aliases = dictionary().aliases_for(res.info["inchikey"])
    key = res.info["inchikey"]

    frags = Chem.GetMolFrags(mol, asMols=True)
    ordered = _order(*_name_single(mol, beams)) if len(frags) == 1 else _name_mixture(frags, beams, key)
    if not ordered:
        res.level, res.message = "未検証", "翻訳モデルが扱えない構造です(大きすぎる・珍しい元素など)。"
        return res

    tidied = list(dict.fromkeys(_tidy_verified(ordered, key)))
    ordered = sorted(tidied, key=lambda x: (_RANK[x[1]], _style(x[0])))  # 整えた後に並べ直す(元の順は保つ)
    if ordered[0][1] == "一致":
        ordered, meta = _pinify(ordered, mol, key)
        res.confidence, res.rules = meta["confidence"], list(dict.fromkeys(meta["rules"]))
        res.pool = meta.get("pool", [])
    best, level = ordered[0]
    res.candidates = ordered[:30]
    res.ok = level in ("一致", "互変異性体違い", "立体以外一致")
    res.name, res.level = best, (level if res.ok else "未検証")
    if level == "一致":
        res.message = "OPSIN で構造に戻して InChIKey が完全一致。"
        if _has_unassigned_stereo(mol):
            res.message += " 入力に立体の指定が無い立体中心があります(名前にも立体記号なし)。"
    elif level == "互変異性体違い":
        res.message = "OPSIN で戻る構造は標準 InChI では同じですが、動く水素の位置(互変異性体)が入力と違います。指示水素・dihydro を確認してください。"
    elif level == "立体以外一致":
        res.message = "骨格・置換位置は一致、立体記号(R/S・E/Z)を合わせられませんでした。名前の立体は自分で付けてください。"
    else:
        res.message = "検証できた名前がありません。表示中の名前は誤りの可能性が高い参考です。"
        if len(_kekule(mol, False)) > 140:
            res.message += " 分子が大きく、翻訳モデルの得意な長さを超えています。"
    if res.ok:
        res.locants = locants_for(best, mol)
    if res.ok and res.confidence == "non-pin":
        res.message += " PIN の書き方になっていない可能性があります(括弧・並び順・位置番号などを確認)。"
    return res


def _has_unassigned_stereo(mol) -> bool:
    centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False)
    return any(lab == "?" for _, lab in centers)


# 酸付加塩・水和物: 本体の名前の後ろに付ける語
_ADDUCTS = {"Cl": "hydrochloride", "Br": "hydrobromide", "I": "hydroiodide", "F": "hydrofluoride",
            "O": "hydrate", "O=S(=O)(O)O": "sulfate", "O=C(O)C(F)(F)F": "trifluoroacetate"}
_ADDUCT_N = {1: "mono", 2: "di", 3: "tri", 4: "tetra"}


def _anion_names(frag: Chem.Mol, beams: int) -> list[str]:
    """陰イオンは、H を付けた中性の酸に名前を付けて -ic acid -> -ate に直した候補も足す。"""
    m = Chem.RWMol(frag)
    for a in m.GetAtoms():
        if a.GetFormalCharge() < 0 and a.GetSymbol() in ("O", "S", "N"):
            a.SetNumExplicitHs(a.GetTotalNumHs() - a.GetFormalCharge())
            a.SetFormalCharge(0)
    try:
        acid = m.GetMol()
        Chem.SanitizeMol(acid)
    except Exception:
        return []
    out = []
    for c, lv in _order(*_name_single(acid, beams)):
        if lv != "一致":
            continue
        a = re.sub(r"ic acid$", "ate", c)
        a = re.sub(r"ous acid$", "ite", a)
        if a != c:
            out.append(a)
    return out[:3]


def _name_mixture(frags, beams, key):
    """塩・混合物: 成分ごとに名前を付け、陽イオン -> 陰イオン -> 中性(付加物は後ろ)の順につないで照合する。"""
    groups: dict[str, list] = {}
    for f in frags:
        groups.setdefault(Chem.MolToSmiles(f), []).append(f)
    parts = []   # (順番, 数, 名前候補, 付加物か)
    for can, fs in groups.items():
        n = len(fs)
        if can in _ADDUCTS:
            word = _ADDUCTS[can]
            pre = _ADDUCT_N.get(n, str(n)) if (n > 1 or can == "O") else ""
            parts.append((4 if can == "O" else 3, 1, [pre + word], True))
            continue
        ordered = _order(*_name_single(fs[0], beams))
        charge = Chem.GetFormalCharge(fs[0])
        names = [c for c, lv in ordered if lv == "一致"][:3]
        if charge < 0:
            names = _anion_names(fs[0], beams) + names
        if not names:
            names = [ordered[0][0]] if ordered else []
        if not names:
            return []
        parts.append((0 if charge > 0 else 1 if charge < 0 else 2, n, names, False))
    parts.sort(key=lambda p: p[0])

    def with_count(name, n):
        if n == 1:
            return name
        if re.fullmatch(r"[a-z]+", name):
            return _MULT.get(n, str(n)) + name
        return _MULT2.get(n, str(n)) + f"({name})"

    tries = [" ".join(with_count(c, p[1]) for c, p in zip(combo, parts))
             for combo in product(*[p[2][:3] for p in parts])]
    judged = [(t, "一致") for t, r in zip(tries, opsin.parse(tries)) if r.mol is not None and inchikey(r.mol) == key]
    if judged:
        return judged
    # つないだ名前が通らなければ、成分ごとの名前を並べる
    joined = " . ".join(f"{p[1]} × {p[2][0]}" if p[1] > 1 else p[2][0] for p in parts)
    return [(joined, "別の構造")]


# ================================================================ 自動判定の入口

def convert(text: str, to: str = "auto", name_it: bool = True) -> Result:
    """to: auto / name(構造→名前) / struct(名前→構造)。name_it=False なら、構造を入れたときの IUPAC 名の計算(数秒〜十数秒)を省く。"""
    raw, text = text, text.strip()
    if to in ("auto", "name"):
        kind, mol = parse_structure(raw)
        if mol is not None and not name_it:
            r = Result(text, "構造→名前", ok=True, mol=mol, level="名前なし",
                       message="IUPAC 名は計算していません(「構造 → 名前を計算する」を入れるか、「この構造に名前を付け直す」で付けられます)。")
            r.info = mol_info(mol)
            r.aliases = dictionary().aliases_for(r.info["inchikey"])
            return r
        if mol is not None:
            r = smiles_to_name(mol)
            r.query = text
            return r
        if to == "name":
            return Result(text, "構造→名前", level="失敗", message="SMILES / InChI / MOL として読めません。")
    return name_to_struct(text)


# ================================================================ 描画・保存

def mol_to_png(mol: Chem.Mol, size=(460, 340), locants: dict | None = None, svg: bool = False):
    from rdkit.Chem.Draw import rdMolDraw2D
    m = Chem.Mol(mol)
    if locants:
        for i, loc in locants.items():
            if i < m.GetNumAtoms():
                m.GetAtomWithIdx(i).SetProp("atomNote", loc)
    d = rdMolDraw2D.MolDraw2DSVG(*size) if svg else rdMolDraw2D.MolDraw2DCairo(*size)
    o = d.drawOptions()
    o.addStereoAnnotation = True
    o.annotationFontScale = 0.6
    o.fixedBondLength = 40 * max(size[0], 460) / 460   # 小さい分子を大きく描きすぎない
    rdMolDraw2D.PrepareAndDrawMolecule(d, m)
    d.FinishDrawing()
    return d.GetDrawingText()


def save_mol(mol: Chem.Mol, path: Path, title: str = ""):
    from rdkit.Chem import AllChem
    m = Chem.Mol(mol)
    AllChem.Compute2DCoords(m)
    if title:
        m.SetProp("_Name", title)
    Chem.MolToMolFile(m, str(path))
