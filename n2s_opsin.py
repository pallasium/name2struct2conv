"""常駐 OPSIN(lib/OpsinServer.class)との通信と、CML からの構造・位置番号の読み取り。"""
from __future__ import annotations

import re
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

from rdkit import Chem

LIB = Path(__file__).resolve().parent / "lib"
OPSIN_JAR = next(iter(sorted(LIB.glob("opsin-cli-*.jar"))), None)


@dataclass
class OpsinResult:
    name: str
    ok: bool
    smiles: str = ""
    warning: str = ""
    message: str = ""
    cml: str = ""

    @property
    def mol(self) -> Chem.Mol | None:
        return Chem.MolFromSmiles(self.smiles) if self.ok and self.smiles else None


class _Server:
    def __init__(self):
        if OPSIN_JAR is None or not (LIB / "OpsinServer.class").exists():
            raise RuntimeError("lib/ に OPSIN の jar か OpsinServer.class がありません。")
        java = shutil.which("java")
        if java is None:
            raise RuntimeError("java が見つかりません(OPSIN に必要)。")
        sep = ";" if Path(java).suffix.lower() == ".exe" else ":"
        self.proc = subprocess.Popen(
            [java, "-Xss8m", "-cp", f"{OPSIN_JAR}{sep}{LIB}", "OpsinServer"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", bufsize=1,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        head = self.proc.stdout.readline()
        if not head.startswith("READY"):
            raise RuntimeError(f"OPSIN が起動しません: {head!r}")
        self.version = head.split("\t")[1].strip()
        self.lock = threading.Lock()

    def ask(self, names: list[str], flags: str) -> list[OpsinResult]:
        clean = [" ".join(n.split()) for n in names]
        with self.lock:
            # 書き込みは別スレッドで。同じスレッドでまとめて書くと、OPSIN の出力側のパイプが
            # 一杯になったところで互いに待ち合って止まる(Windows のパイプは小さい)。
            def write():
                try:
                    for n in clean:
                        self.proc.stdin.write(f"{flags}\t{n}\n")
                    self.proc.stdin.flush()
                except OSError:
                    pass
            w = threading.Thread(target=write, daemon=True)
            w.start()
            out = []
            try:
                for n in clean:
                    line = self.proc.stdout.readline()
                    if not line:
                        raise RuntimeError("OPSIN が止まりました。")
                    p = line.rstrip("\r\n").split("\t")
                    p += [""] * (5 - len(p))
                    out.append(OpsinResult(n, p[0] in ("SUCCESS", "WARNING") and bool(p[1]), p[1], p[2], p[3], p[4]))
            except BaseException:
                # 途中で抜けると、読み残した返事が次の問い合わせの答えとして読まれてしまう(行がずれる)。
                # プロセスを止めて、次の呼び出しで起動し直させる。
                self.close()
                raise
            w.join()
            return out

    def close(self):
        try:
            self.proc.kill()
            self.proc.wait(timeout=5)
        except Exception:
            pass


_server: _Server | None = None
_server_lock = threading.Lock()


def server() -> _Server:
    global _server
    with _server_lock:
        if _server is None or _server.proc.poll() is not None:
            _server = _Server()
        return _server


def parse(names: list[str], *, lenient: bool = False, acids: bool = False, cml: bool = False) -> list[OpsinResult]:
    """名前 -> OpsinResult。lenient: 読めない立体を無視して通す(照合には使わない)。"""
    if not names:
        return []
    flags = ("l" if lenient else "") + ("a" if acids else "") + ("c" if cml else "")
    return server().ask(names, flags)


# ---------------------------------------------------------------- CML -> RDKit(位置番号つき)

_ATOM = re.compile(r'<atom id="(a\d+)"([^>]*?)(?:/>|>(.*?)</atom>)')
_BOND = re.compile(r'<bond[^>]*atomRefs2="(a\d+) (a\d+)"[^>]*order="([^"]+)"')
_ATTR = re.compile(r'(\w+)="([^"]*)"')
_LABEL = re.compile(r'<label value="([^"]+)" dictRef="cmlDict:locant"')
_ORDER = {"1": Chem.BondType.SINGLE, "S": Chem.BondType.SINGLE, "2": Chem.BondType.DOUBLE,
          "D": Chem.BondType.DOUBLE, "3": Chem.BondType.TRIPLE, "T": Chem.BondType.TRIPLE,
          "A": Chem.BondType.AROMATIC}


def cml_to_mol(cml: str) -> Chem.Mol | None:
    """OPSIN の CML -> RDKit Mol。各原子に "locants"(位置番号のリスト、'|' 区切り)を付ける。立体は読まない。"""
    if not cml:
        return None
    rw = Chem.RWMol()
    idx = {}
    for m in _ATOM.finditer(cml):
        attrs = dict(_ATTR.findall(m.group(2)))
        el = attrs.get("elementType", "C")
        if el == "R":
            el = "*"
        a = Chem.Atom(0 if el == "*" else Chem.GetPeriodicTable().GetAtomicNumber(el))
        a.SetFormalCharge(int(attrs.get("formalCharge", 0)))
        if "isotopeNumber" in attrs:
            a.SetIsotope(int(attrs["isotopeNumber"]))
        if "spinMultiplicity" in attrs:
            a.SetNumRadicalElectrons(int(attrs["spinMultiplicity"]) - 1)
        a.SetNoImplicit(True)
        labels = _LABEL.findall(m.group(3) or "")
        a.SetProp("locants", "|".join(labels))
        idx[m.group(1)] = rw.AddAtom(a)
    for a1, a2, order in _BOND.findall(cml):
        if a1 in idx and a2 in idx:
            rw.AddBond(idx[a1], idx[a2], _ORDER.get(order, Chem.BondType.SINGLE))
    try:
        mol = rw.GetMol()
        Chem.SanitizeMol(mol)
        return Chem.RemoveHs(mol, sanitize=True)
    except Exception:
        return None


def numeric_locant(atom: Chem.Atom) -> str:
    """原子の位置番号のうち、立体記号に使える形('2', "1'", '4a', '10b')の最初のもの。"""
    for lab in atom.GetProp("locants").split("|") if atom.HasProp("locants") else []:
        if re.fullmatch(r"\d+[a-z]?'*", lab):
            return lab
    return ""
