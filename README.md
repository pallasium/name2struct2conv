# name2struct2conv

IUPAC 名 ⇄ 化学構造(SMILES / InChI / MOL)を相互に変換する、Windows 向けのデスクトップツール(GUI とコマンド)。
**初回のセットアップ後は完全にオフラインで動き、入力した名前や構造を外部に送らない。**

- **名前 → 構造**: [OPSIN](https://github.com/dan2097/opsin)(規則ベース)に、略号・慣用名・日本語名の辞書、綴りの自動修正、機械学習による推定を組み合わせる。
- **構造 → 名前**: 機械学習の翻訳モデルが出した候補を、**すべて OPSIN で構造に戻して InChIKey を照合**し、同じ構造に戻る名前だけを「一致」として出す。立体(R/S・E/Z)は位置番号つきで付け直す。

## できること

| | |
|---|---|
| 入力 | IUPAC 名・慣用名・略号(DMF, DIPEA, EDC·HCl など)・日本語名(酢酸エチル など)/ SMILES / InChI / MOL(貼り付け)/ MOL・SDF・CDXML・SMI ファイル |
| 出力 | IUPAC 名、SMILES、InChI、InChIKey、分子式、分子量、精密質量、構造図(立体と IUPAC の位置番号つき)、MOL / SDF / PNG / SVG |
| 一括変換 | CSV / TXT / XLSX(1 列目)/ SDF など → Excel(判定で色分け)/ CSV / SDF |
| 塩・水和物 | `sodium acetate`、`disodium butanedioate`、`… dihydrochloride dihydrate` など |
| 立体 | 立体中心・二重結合の数に上限なし。`4-[(1R)-1-hydroxyethyl]…`、`methyl (2S)-…` のように括弧の中の正しい位置に置く |
| 利用者辞書 | 自分の略号・化合物番号を `user_dict.tsv` に登録できる |

## 動作環境

- Windows 10 / 11(作者の確認環境)。コードは Linux でも動く想定だが未確認。
- Python 3.10 以降
- Java 11 以降(OPSIN に必要。`java` が PATH にあること)
- GPU(CUDA)があれば自動で使う。無くても CPU で動く(遅くなる)。
- ディスク: モデルと OPSIN で約 390 MB

## セットアップ

```
git clone https://github.com/pallasium/name2struct2conv.git
cd name2struct2conv
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt        # GPU を使うなら、先に PyTorch の CUDA 版を入れる
python setup_assets.py                 # OPSIN の jar とモデルを各配布元から取得(初回だけ、約 390 MB)
```

`setup_assets.py` は OPSIN の jar を GitHub のリリースから取り、SHA-256 を確かめる。モデルは Hugging Face の `knowledgator/*` から、版(コミット)を固定して取る。

## 使い方

### GUI
`name2struct2conv.bat` をダブルクリック(または `python name2struct.py`)。
上の欄に名前・SMILES・InChI・MOL を入れて Enter。方向は自動で判定する(ラジオボタンで固定もできる)。

- 判定の色: 緑 = 検証済み / 黄 = 立体のみ要確認・推定 / 赤 = 未検証・失敗
- 「この構造に名前を付け直す」: 慣用名や崩れた書き方で入れた名前から、検証済みの名前を作り直す
- 起動直後の 20〜30 秒はモデルの読み込み中(名前 → 構造はすぐ使える)

### コマンド
```
python name2struct.py "2-acetyloxybenzoic acid"               # 名前 → 構造
python name2struct.py "CC(=O)Oc1ccccc1C(=O)O"                 # 構造 → 名前
python name2struct.py "DMF" --mol a.mol --png a.png --locants  # 構造を保存(IUPAC 位置番号つきの図)
python name2struct.py "2-acetoxy benzoic acid" --rename       # 名前 → 構造 → 検証済みの名前も出す
python name2struct.py --file list.csv -o out.xlsx              # 一括変換
python name2struct.py "CCO" --json                             # JSON で出力
python name2struct.py -h
```

## しくみ

### 名前 → 構造(上から順に試す)
1. 辞書(`names_dict.tsv` 226 件 + `user_dict.tsv`)。全角半角・大文字小文字は区別しない。辞書の全項目は、OPSIN で読んだ構造の分子式が書いた分子式と一致することをテストで確かめている。
2. OPSIN。読めない立体記号は無視して注意を出す。
3. 綴りの揺れの修正(全角・ダッシュ・空白・"acid" の書き忘れ)。
4. 1 文字の誤字の総当たり(例: `metyl` → `methyl`)。OPSIN で読める直し方のうち、IUPAC 名によく出る語でできたものを優先する。「推定」と表示する。
5. それでも読めなければ、機械学習(IUPAC2SMILES)の推定構造を「推定・要確認」で出す。

### 構造 → 名前
1. 構造をケクレ形の SMILES にして、翻訳モデル(SMILES2IUPAC)に名前の候補を数十個出させる。
2. 各候補を OPSIN で構造に戻し、InChIKey を比べる(完全一致 / 立体以外一致 / 別の構造)。
3. 骨格だけ合う候補には、RDKit の CIP 判定から立体記号を位置番号つきで付け直す。OPSIN は、括弧の中や 2 語目以降の前に置いた立体記号をその範囲の中だけで探すので、照合を通った位置のうち一番狭い範囲を採る。
4. 当たらなければ、SMILES の書き方(原子の順番)を変えて最大 4 回やり直す。
5. 一致した候補が複数あれば、`oxidanyl`・`ethanoic acid` のような書き方を `hydroxy`・`acetic acid` などに言い換え(言い換えた形も OPSIN で照合する)、モデルの自信が高いものを選ぶ。

### 判定の意味
| 判定 | 意味 |
|---|---|
| 一致 | 名前を OPSIN で構造に戻すと、立体も含めて入力と同じ |
| 立体以外一致 | 骨格・置換位置は同じ。R/S・E/Z は合わせられなかった(自分で直す) |
| 未検証 | 同じ構造に戻る名前が見つからなかった。表示される名前は誤りの可能性が高い |
| OPSIN / 辞書 / 利用者辞書 | 名前 → 構造で、その方法で読んだ |
| 推定 | 綴りを直して読んだ、または機械学習で推定した(要確認) |

「一致」は**その名前から同じ構造が描ける**ことの確認であって、IUPAC 推奨名(PIN)であることの保証ではない。

## 精度

`bench.py` の 100 分子(医薬品・天然物、保護アミノ酸・ペプチド、糖、ステロイド、複素環、塩、E/Z を含む。作者が選んだ一般的な化合物)で、構造 → 名前を評価した。

- **100 分子中 98 分子が「一致」**(上の検証方法で、名前から同じ構造に戻る)。残る 2 分子(クルクミン、ジクロフェナク)は「未検証」で、モデルがベンゼン環を 7 員環と読み違えた。
- 平均 **約 2 秒 / 分子**(GPU 使用時、ほかに負荷のないとき。初回のモデル読み込みを除く)

検証は「出した名前を OPSIN で構造に戻せるか」であり、名前の書き方の良さ(推奨名かどうか)は測っていない。100 分子は学習データとの重なりを確かめていない一般的な化合物で、新規の大きな分子では成績が下がると考えられる。ほかの命名ソフトとは比べていない。

## 苦手なもの

- 大きな分子(ケクレ形 SMILES で約 150 字超)・珍しい元素・金属錯体: 「未検証」になりやすい
- 入力に立体の指定が無い立体中心: 名前にも立体記号は付かない(メッセージで知らせる)
- 名前 → 構造の機械学習推定は誤りが多い(必ず構造図で確認する)

## 利用者辞書

`user_dict.tsv`(タブ区切り、UTF-8。GUI の「利用者辞書…」で作られる):
```
化合物A|cmpd-A	4-methoxybenzoic acid
化合物B	COc1ccc(cc1)C(=O)OC
```
右側は名前でも SMILES でもよい。名前 → 構造で使えるほか、構造 → 名前の結果に「別名」として出る。

## テスト

```
python -m pytest tests -q    # 28 件(OPSIN とモデルが要る)
python bench.py              # 100 分子の評価
```

## ファイル

| | |
|---|---|
| `name2struct.py` | 入口(コマンド / 引数なしで GUI) |
| `n2s_core.py` | 変換の中身 |
| `n2s_gui.py` | GUI(tkinter) |
| `n2s_opsin.py` | 常駐させた OPSIN との通信、CML(位置番号)の読み取り |
| `n2s_stereo.py` | 立体記号を外す・付け直す |
| `names_dict.tsv` | 辞書(略号・慣用名・日本語名 → OPSIN が読める名前、検査用の分子式) |
| `lib/OpsinServer.java` (`.class`) | OPSIN を常駐させる小さな Java プログラム(Java 11 向けにコンパイル済み) |
| `setup_assets.py` | OPSIN の jar とモデルの取得 |

## 第三者のソフトウェアとモデル

このリポジトリには、以下のものを**含めていない**。`setup_assets.py` が各配布元から取ってくる。

- **OPSIN** 2.9.0(Daniel Lowe 他)— MIT License。jar には依存ライブラリが同梱されており、それぞれのライセンスに従う。
  D. M. Lowe, P. T. Corbett, P. Murray-Rust, R. C. Glen, "Chemical Name to Structure: OPSIN, an Open Source Solution", *J. Chem. Inf. Model.* **2011**, 51, 739–753.
- **Knowledgator のモデル** `SMILES2IUPAC-canonical-base`、`SMILES2IUPAC-isomeric-small`、`IUPAC2SMILES-isomeric-base` — Apache License 2.0。トークナイザ `SMILES-FAST-TOKENIZER`、`IUPAC-FAST-TOKENIZER` は同じ作者の Hugging Face リポジトリから取る(リポジトリにライセンスの明記は無い。同作者の Apache-2.0 のライブラリ chemical-converters が使うもの)。モデルは同ライブラリを使わず、`transformers` の MT5 の上に読み込んでいる。
- 依存ライブラリ(pip で入る): RDKit(BSD-3-Clause)、PyTorch(BSD-3-Clause)、transformers・safetensors・huggingface_hub(Apache-2.0)、openpyxl(MIT)。

このリポジトリのコードは MIT License(`LICENSE`)。

## AI の利用について

このリポジトリのコードと README の作成には、AI(Claude / Claude Code)を支援ツールとして利用した。

## 変更履歴

### v1.0.0
- 最初の公開版。
