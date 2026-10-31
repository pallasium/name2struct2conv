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
| 計算値 | 元素分析(質量 %)、マススペクトル(同位体パターン。M+・[M+H]+・[M+Na]+・[M-H]-)、化学物性の推算(沸点・融点・臨界定数・logP・tPSA など。GUI の「物性を計算する」) |
| フラグメント | 構造から MS/MS のフラグメントを予測して、ピークとその部分構造を表示(下の「フラグメント予測」)。MassBank の実測との照合もできる |
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

### フラグメント予測を学習モデルで使う(任意)

学習済みのモデルと MassBank のデータは**このリポジトリに含まれない**(再配布しない)。何もしなくても、フラグメント予測は規則ベースで動く。MassBank の実測 MS/MS から学習したモデルを使いたい人は、自分で作る。

```
python n2s_msdb.py --download          # MassBank の公式リリースを取得(約 140MB)
python n2s_msdb.py --build             # 検索用の索引を data/massbank/ に作る
python n2s_fragml_train.py --group ESI+ --ntrain 2500 --save    # 勾配ブースティング(ESI+ / ESI- / EI ごと)
python n2s_fragnn.py --group ESI+ --ntrain 4000 --epochs 40 --save   # グラフニューラルネット(GPU 向け。任意)
```

モデルが `data/fragml/`・`data/fragnn/` にあれば、自動で使う(GNN があれば GNN、無ければ勾配ブースティング、それも無ければ規則)。
MassBank は CC BY などのオープンデータで、レコードごとにライセンスが違う。引用: MassBank Consortium and its contributors(https://github.com/MassBank/MassBank-data)。

## 使い方

### GUI
`name2struct2conv.bat` をダブルクリック(または `python name2struct.py`)。
上の欄に名前・SMILES・InChI・MOL を入れて Enter。方向は自動で判定する(ラジオボタンで固定もできる)。

- 判定の色: 緑 = 検証済み / 黄 = 立体のみ要確認・推定 / 赤 = 未検証・失敗
- 「この構造に名前を付け直す」: 慣用名や崩れた書き方で入れた名前から、検証済みの名前を作り直す
- メイン画面のチェックボックス: 「構造 → 名前を計算する」(既定オン。切ると構造を入れたとき名前を計算せず、構造・元素分析・MS だけ即座に出す)/「物性を計算する」(既定オフ。別ウィンドウに推算値を出す)
- 「その他…」: コマンドの引数(`--locants` 位置番号、`--rename` 名前の付け直し、`--json` JSON 表示)をチェックボックスで切り替える
- 元素分析(計算値 %)とマススペクトル(計算値)を表示。構造図・スペクトルはホイールで拡大縮小、ドラッグで移動、ダブルクリックで元に戻る
- 「フラグメント…」: 別ウィンドウに、予測したフラグメントのスペクトル・ピークリスト・構造(選んだピークの原子を赤で強調)。予測方式を選べる(学習 / 規則)
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
5. 一致した候補を、IUPAC 推奨名(PIN)の書き方に近づける(v1.1.0)。名前を木にして、括弧の種類・英数字順・di / bis・位置番号を直し(`n2s_pin2.py`)、ベンゼン環・尿素・ジアミンなどの言い換えをし、環の水素化・指示水素は構造から決める(`n2s_hydro.py`)。エステル・リンの酸のエステルは酸とアルコールに分けて名付けて組み立てる(`n2s_assemble.py`)。書き換えた名前も、**OPSIN で互変異性体まで同じ構造に戻るものだけ**を採る。
6. 候補の中から、PIN の書き方に合わない箇所(`oxidanyl`・`bis(chloro)`・括弧の抜け・並び順の違いなど)が少ないものを選び、同じなら最小の位置番号、それでも同じならモデルの自信が高いものを選ぶ。
7. 結果に確度を付ける: `rule`(PIN の規則で組み直した)/ `model`(モデルの名前そのまま)/ `non-pin`(PIN の書き方になっていない可能性)。適用した規則は `rules` 列に出る。

### 判定の意味
| 判定 | 意味 |
|---|---|
| 一致 | 名前を OPSIN で構造に戻すと、立体も含めて入力と同じ |
| 立体以外一致 | 骨格・置換位置は同じ。R/S・E/Z は合わせられなかった(自分で直す) |
| 互変異性体違い | 標準 InChI では同じだが、動く水素の位置(互変異性体)が違う。指示水素・dihydro を確認する |
| 未検証 | 同じ構造に戻る名前が見つからなかった。表示される名前は誤りの可能性が高い |
| OPSIN / 辞書 / 利用者辞書 | 名前 → 構造で、その方法で読んだ |
| 推定 | 綴りを直して読んだ、または機械学習で推定した(要確認) |

「一致」は**その名前から同じ構造が描ける**ことの確認であって、IUPAC 推奨名(PIN)であることの保証ではない。

## 精度

`bench.py` の 100 分子(医薬品・天然物、保護アミノ酸・ペプチド、糖、ステロイド、複素環、塩、E/Z を含む。作者が選んだ一般的な化合物)で、構造 → 名前を評価した。

- **100 分子中 98 分子が「一致」**(上の検証方法で、名前から同じ構造に戻る)。残る 2 分子(クルクミン、ジクロフェナク)は「未検証」で、モデルがベンゼン環を 7 員環と読み違えた。
- 平均 **約 2 秒 / 分子**(GPU 使用時、ほかに負荷のないとき。初回のモデル読み込みを除く)

検証は「出した名前を OPSIN で構造に戻せるか」であり、名前の書き方の良さ(推奨名かどうか)は上の測定では見ていない。次の節で別に測った。100 分子は学習データとの重なりを確かめていない一般的な化合物で、新規の大きな分子では成績が下がると考えられる。ほかの命名ソフトとは比べていない。

### IUPAC 推奨名(PIN)との一致(v1.1.0)

Wikipedia の化学物質ページに載っている PIN と SMILES の組 1197 件(OPSIN で読んだ構造が SMILES と InChIKey で一致する分)を使い、乱数 20261030 で並べ替えて、先頭 300 件を調整用(dev)、301〜600 件を最終評価用(test)に分けた。test は、コードを固定したあとに**一度だけ**回した。比較は大文字小文字のみ無視した完全一致(括弧の種類は区別する)。

| test 300 件 | v1.0.0 | v1.1.0 |
|---|---|---|
| 名前が OPSIN で構造に戻った(一致) | 285 | 285 |
| **PIN と完全一致** | 124(41.3%) | **150(50.0%)** |
| PIN が候補の中にある | 130(43.3%) | 168(56.0%) |

- 調整に使った dev では 42.3% → 60.7% だったが、test では 41.3% → 50.0% にとどまった。dev に合わせた調整のぶんは test に効いていない。目標にしていた「test で 70%」には届いていない。
- 出した名前が元の構造に戻ることは 100% 保証している(戻らない名前を「一致」にしない)。上の数字は「PIN と同じ書き方か」であり、PIN でない名前でも正しい構造を表している。
- 外れの主な種類: エステル・アミド・ウレアの書き方、水素化・指示水素(`3,7-dihydro-1H-` など)、縮環の名前(`pyrido[3,2-b][1,4]benzothiazine` など。構造から縮環名を作る処理は無い)、リンの酸のエステル、アミニウム。Wikipedia 側の PIN にも揺れがあり(`di(octyl)`、`acetylene`、互変異性体の違いなど)、完全一致が取れないものが一定数ある。
- 確度の表示は目安として働く(test の PIN 一致率: `model` 62%、`rule` 51%、`non-pin` 26%)。`non-pin` の名前は PIN の書き方になっていない可能性が高い。
- 測っていないこと: ほかの命名ソフトとの比較、有機金属・高分子、Wikipedia 以外の PIN。評価に使った Wikipedia のデータは再配布しない(このリポジトリに含まない)。

#### final 297 件

601〜900 件(hold)と 901〜1197 件(final、297 件)も取り分けていた。final は、コードを固定してから最初に一度だけ回した(156 件、52.5%)。その結果の外れを見て、既知の誤りの型を 5 つ直した(置換した蟻酸の形、立体記号の位置番号、括弧の要らない置換基、長鎖アルコールの慣用名、二酸エステルの位置記号)あとで、もう一度回した:

| final 297 件 | v1.0.0 | v1.1.0 |
|---|---|---|
| **PIN と完全一致** | 139(46.8%) | **166(55.9%)** |

- **final は 5 つの修正の根拠にしたので、この 55.9% は未知のデータでの成績より高めに出ている。** 最初の 1 回の 52.5% のほうが、未知のデータの目安に近い。
- test と final のどちらでも v1.0.0 より 6〜9 ポイント以上上がった。目標の「test で 70%」には届いていない(基準は下げていない)。
- 外れの主な種類は上と同じ。新しい評価データ(別の PIN の出典、または手で作った正解)が無いと、これ以上の改善は正直には測れない。

## 苦手なもの

- 大きな分子(ケクレ形 SMILES で約 150 字超)・珍しい元素・金属錯体: 「未検証」になりやすい
- 入力に立体の指定が無い立体中心: 名前にも立体記号は付かない(メッセージで知らせる)
- 名前 → 構造の機械学習推定は誤りが多い(必ず構造図で確認する)
- 縮環・橋かけ・スピロの名前: モデルが出した名前をそのまま使うので、PIN の縮環名(`pyrido[...]benzothiazine` など)になるとは限らない

## 利用者辞書

`user_dict.tsv`(タブ区切り、UTF-8。GUI の「利用者辞書…」で作られる):
```
化合物A|cmpd-A	4-methoxybenzoic acid
化合物B	COc1ccc(cc1)C(=O)OC
```
右側は名前でも SMILES でもよい。名前 → 構造で使えるほか、構造 → 名前の結果に「別名」として出る。

## テスト

```
python -m pytest tests -q    # 86 件(学習モデルや MassBank が無いと 4 件は skip になる)
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
| `n2s_pin2.py` | 名前を PIN の書き方に直す(名前を木にして、括弧・英数字順・倍数・位置番号を決める。規則に ID が付いている) |
| `n2s_pin.py` | 上の字句の規則(置換基の言い換えなど)と、PIN らしさの数え上げ |
| `n2s_hydro.py` | 環の水素化・指示水素・付加水素を構造から決める |
| `n2s_assemble.py` | エステル・リンの酸のエステル・アミド・尿素を、部分に分けて名付けて組み立てる |
| `n2s_ms.py` | 元素分析・マススペクトル(同位体パターン)の計算 |
| `n2s_props.py` | 化学物性の推算(Joback 法・Crippen・Ertl の tPSA など。推算値で、実測ではない) |
| `n2s_view.py` | 構造図・スペクトルの表示(ホイールで拡大縮小・ドラッグ) |
| `n2s_frag.py` | フラグメント予測の入口(規則ベース + 学習モデルの呼び分け) |
| `n2s_fragml.py` / `n2s_fragml_train.py` | 候補フラグメントの列挙と特徴量 / 勾配ブースティングの学習 |
| `n2s_fragnn.py` | グラフニューラルネット(PyTorch のみ)の学習・推論 |
| `n2s_fragwin.py` | フラグメントのウィンドウ(GUI) |
| `n2s_msdb.py` / `msdb_bench.py` / `fragml_coverage.py` | MassBank の索引・照合 / 評価 / 候補の網羅率 |
| `names_dict.tsv` | 辞書(略号・慣用名・日本語名 → OPSIN が読める名前、検査用の分子式) |
| `lib/OpsinServer.java` (`.class`) | OPSIN を常駐させる小さな Java プログラム(Java 11 向けにコンパイル済み) |
| `setup_assets.py` | OPSIN の jar とモデルの取得 |

## 第三者のソフトウェアとモデル

このリポジトリには、以下のものを**含めていない**。`setup_assets.py` が各配布元から取ってくる。

- **OPSIN** 2.9.0(Daniel Lowe 他)— MIT License。jar には依存ライブラリが同梱されており、それぞれのライセンスに従う。
  D. M. Lowe, P. T. Corbett, P. Murray-Rust, R. C. Glen, "Chemical Name to Structure: OPSIN, an Open Source Solution", *J. Chem. Inf. Model.* **2011**, 51, 739–753.
- **Knowledgator のモデル** `SMILES2IUPAC-canonical-base`、`SMILES2IUPAC-isomeric-small`、`IUPAC2SMILES-isomeric-base` — Apache License 2.0。トークナイザ `SMILES-FAST-TOKENIZER`、`IUPAC-FAST-TOKENIZER` は同じ作者の Hugging Face リポジトリから取る(リポジトリにライセンスの明記は無い。同作者の Apache-2.0 のライブラリ chemical-converters が使うもの)。モデルは同ライブラリを使わず、`transformers` の MT5 の上に読み込んでいる。
- **推算式の出典**(コードは自作で、公表された手法の式と係数を使っている。第三者のコードは含めない): Joback 法 — K. G. Joback, R. C. Reid, *Chem. Eng. Commun.* **1987**, 57, 233–243 / logP・MR — S. A. Wildman, G. M. Crippen, *J. Chem. Inf. Comput. Sci.* **1999**, 39, 868–873(RDKit の実装)/ tPSA — P. Ertl, B. Rohde, P. Selzer, *J. Med. Chem.* **2000**, 43, 3714–3717(RDKit の実装)/ ESOL — J. S. Delaney, *J. Chem. Inf. Comput. Sci.* **2004**, 44, 1000–1005。フラグメント予測の「CFM-ID 型」は、学習の枠組みを参考にしただけで、CFM-ID のコードもモデルも使っていない(C. Allen et al., *Anal. Chem.* **2014**, 86, 7274)。
- **MassBank**(任意。フラグメントの学習・照合に使う)— CC BY などのオープンデータで、レコードごとにライセンスが違う。リポジトリには含めず、`n2s_msdb.py --download` で公式リリースを取る。MassBank Consortium and its contributors, https://github.com/MassBank/MassBank-data
- 依存ライブラリ(pip で入る): RDKit(BSD-3-Clause)、PyTorch(BSD-3-Clause)、scikit-learn・joblib(BSD-3-Clause)、transformers・safetensors・huggingface_hub(Apache-2.0)、openpyxl(MIT)。

このリポジトリのコードは MIT License(`LICENSE`)。

## AI の利用について

このリポジトリのコードと README の作成には、AI(Claude / Claude Code)を支援ツールとして利用した。

## 変更履歴

### v1.1.0(未リリース)
- 元素分析(計算値)・マススペクトル(同位体パターン)の表示、化学物性の推算(`n2s_props.py`)、フラグメント予測(規則ベース + MassBank から学習するモデル。モデルとデータは含めない)を追加。GUI に「その他…」と、構造 → 名前・物性のチェックボックスを追加。
- final 297 件で PIN との完全一致 46.8% → 52.5%(最初に一度だけ回した値)。その外れを見て 5 つの型を直したあとは 55.9% だが、こちらは final を根拠にしているので高めに出ている。
- 出す名前を IUPAC 推奨名(PIN)の書き方に近づける処理を追加した(`n2s_pin2.py`・`n2s_pin.py`・`n2s_hydro.py`・`n2s_assemble.py`)。Wikipedia の PIN との完全一致は、test 300 件で 41.3% → 50.0%。
- 互変異性体まで区別する照合(InChI の FixedH)と、判定「互変異性体違い」を追加。
- 結果に確度(`pin` 列: rule / model / non-pin)と適用した規則(`rules` 列)を追加。

### v1.0.0
- 最初の公開版。
