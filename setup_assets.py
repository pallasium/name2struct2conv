"""初回だけ実行: OPSIN(jar)と翻訳モデルを各配布元から取ってくる。以後はオフラインで動く。

  python setup_assets.py

  lib/opsin-cli-2.9.0-jar-with-dependencies.jar   … GitHub の OPSIN リリース(MIT)。SHA-256 を確認する
  models/<名前>/                                  … Hugging Face の knowledgator/*(版を固定)
  lib/OpsinServer.class が無ければ javac で作る(JDK が必要。同梱の .class があれば不要)
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIB = HERE / "lib"
MODELS = HERE / "models"

OPSIN_JAR = "opsin-cli-2.9.0-jar-with-dependencies.jar"
OPSIN_URL = f"https://github.com/dan2097/opsin/releases/download/2.9.0/{OPSIN_JAR}"
OPSIN_SHA256 = "c2e29326c281f87b59a05d934d8589adac6e9d17b95b984931b3e739111b360f"

# (Hugging Face のリポジトリ, 固定する版)
MODEL_REPOS = [
    ("knowledgator/SMILES2IUPAC-canonical-base", "035c811bea50996dae04fc657e5e610aa17cb8f6"),
    ("knowledgator/SMILES2IUPAC-isomeric-small", "954cae78b7e0cc9c417d461c7ffc4524502b0085"),
    ("knowledgator/IUPAC2SMILES-isomeric-base", "b034166ad15bdc51327d316bd6e4c4456b1c4797"),
    ("knowledgator/SMILES-FAST-TOKENIZER", "9b829a79eae149c4045b4b9dc5713d0d93a0b1ca"),
    ("knowledgator/IUPAC-FAST-TOKENIZER", "03272d7907902084d9dd77964f375f1327b5175e"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def get_opsin():
    jar = LIB / OPSIN_JAR
    if jar.exists() and sha256(jar) == OPSIN_SHA256:
        print("ok (既にある)", jar)
        return
    tmp = jar.with_suffix(".part")
    print("取得中:", OPSIN_URL)
    urllib.request.urlretrieve(OPSIN_URL, tmp)
    if sha256(tmp) != OPSIN_SHA256:
        tmp.unlink()
        sys.exit("OPSIN の jar のハッシュが一致しません。取得を中止しました。")
    tmp.replace(jar)
    print("ok", jar)


def build_server():
    if (LIB / "OpsinServer.class").exists():
        return
    javac = shutil.which("javac")
    if javac is None:
        sys.exit("lib/OpsinServer.class が無く、javac(JDK)も見つかりません。")
    subprocess.run([javac, "--release", "11", "-encoding", "UTF-8", "-cp", str(LIB / OPSIN_JAR),
                    "-d", str(LIB), str(LIB / "OpsinServer.java")], check=True)
    print("ok", LIB / "OpsinServer.class")


def get_models():
    from huggingface_hub import snapshot_download
    for repo, rev in MODEL_REPOS:
        dest = MODELS / repo.split("/")[1]
        snapshot_download(repo, revision=rev, local_dir=dest)
        print("ok", dest)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    LIB.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    get_opsin()
    build_server()
    get_models()
    if shutil.which("java") is None:
        print("注意: java が PATH にありません。Java 11 以降を入れてください(OPSIN に必要)。")
    print("準備完了")
