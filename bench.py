"""動作確認: 構造 -> 名前 を約 100 分子で回し、OPSIN 照合の結果(一致 / 立体以外一致 / 未検証)を数える。

  python bench.py            全部
  python bench.py --quick    最初の 30 分子だけ
"""
import sys
import time
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")
import n2s_core as c

SET = """
CC(=O)Oc1ccccc1C(=O)O	アスピリン
CN1CCC[C@H]1c1cccnc1	ニコチン
CC(C)NCC(O)COc1cccc2ccccc12	プロプラノロール
COc1ccc2[nH]cc(CCNC(C)=O)c2c1	メラトニン
O=C1NC(=O)C(c2ccccc2)(c2ccccc2)N1	フェニトイン
CCN(CC)CC(=O)Nc1c(C)cccc1C	リドカイン
C[C@]12CC[C@H]3[C@@H](CCC4=CC(=O)CC[C@]34C)[C@@H]1CC[C@@H]2O	テストステロン
OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O	β-D-グルコース
CC(C)(C)OC(=O)N[C@@H](Cc1ccccc1)C(=O)O	Boc-Phe-OH
CCCCCCCCCCCCCCCCCC(=O)O	ステアリン酸
O=C(O)/C=C/c1ccc(O)c(O)c1	カフェ酸
CC(=O)N[C@@H](CS)C(=O)O	N-アセチルシステイン
FC(F)(F)c1ccc(Oc2ccc(Cl)cc2)cc1	ジアリールエーテル
NS(=O)(=O)c1ccc(N)cc1	スルファニルアミド
O=C1c2ccccc2C(=O)c2ccccc12	アントラキノン
CC(C)(C)c1cc(C)cc(C(C)(C)C)c1O	BHT
O=C(N[C@@H](Cc1ccccc1)C(=O)OC)[C@@H](N)CC(=O)O	アスパルテーム
OC(=O)CCCCCCC/C=C\\CCCCCCCC	オレイン酸
c1ccc(cc1)P(c1ccccc1)c1ccccc1	トリフェニルホスフィン
CCCCOC(=O)c1ccccc1C(=O)OCCCC	フタル酸ジブチル
CC(C)CC1=CC=C(C=C1)C(C)C(=O)O	イブプロフェン(ラセミ)
O=C(OCC1c2ccccc2-c2ccccc12)N[C@@H](CC(C)C)C(=O)O	Fmoc-Leu-OH
O=C(OCc1ccccc1)N[C@@H](C)C(=O)OC	Cbz-Ala-OMe
O=C(Nc1ccccc1)Nc1ccccc1	ジフェニルウレア
CN1C=NC2=C1C(=O)N(C(=O)N2C)C	カフェイン
CC(=O)Nc1ccc(O)cc1	アセトアミノフェン
C1CCC(CC1)N=C=NC1CCCCC1	DCC
OC(=O)[C@@H](N)Cc1c[nH]c2ccccc12	L-トリプトファン
N[C@@H](CCCCN)C(=O)O	L-リシン
NC(=N)NCCC[C@H](N)C(=O)O	L-アルギニン
OC(=O)[C@@H]1CCCN1	L-プロリン
O=C(O)C[C@H](O)C(=O)O	L-リンゴ酸
OC(=O)[C@H](O)[C@@H](O)C(=O)O	L-酒石酸
C[C@@H](O)[C@H](N)C(=O)O	L-トレオニン
OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O	D-グルコース(アノマー未指定)
CC(C)[C@@H]1CC[C@@H](C)C[C@H]1O	L-メントール
CC1(C)C2CCC1(C)C(=O)C2	カンファー
C/C=C/C=O	クロトンアルデヒド
C/C(=C\\C(=O)O)c1ccccc1	3-フェニル-2-ブテン酸(E)
Cl/C=C\\Cl	cis-1,2-ジクロロエテン
c1ccc2c(c1)ccc1ccccc12	フェナントレン
c1ccc2cc3ccccc3cc2c1	アントラセン
c1cc2ccc3cccc4ccc(c1)c2c34	ピレン
O=c1cc(-c2ccccc2)oc2ccccc12	フラボン
O=C1CCCCN1	δ-バレロラクタム
C1COCCN1	モルホリン
c1ccc2[nH]ccc2c1	インドール
c1ccc2ncccc2c1	キノリン
c1cnc2ccccc2n1	キノキサリン
O=C1NC(=O)c2ccccc21	フタルイミド
CC1=CC(=O)c2ccccc2C1=O	メナジオン
Oc1ccc(cc1)/C=C/c1cc(O)cc(O)c1	レスベラトロール
COc1cc(/C=C/C(=O)CC(=O)/C=C/c2ccc(O)c(OC)c2)ccc1O	クルクミン
CC(C)C[C@H](NC(=O)[C@H](Cc1ccccc1)NC(=O)OC(C)(C)C)C(=O)OC	Boc-Phe-Leu-OMe
CC(=O)OC[C@H]1O[C@@H](OC(C)=O)[C@H](OC(C)=O)[C@@H](OC(C)=O)[C@@H]1OC(C)=O	ペンタアセチルグルコース
O=C(O)c1cc(C(=O)O)cc(C(=O)O)c1	トリメシン酸
OCCN(CCO)CCO	トリエタノールアミン
CCCCCCCCCCCCOS(=O)(=O)[O-].[Na+]	SDS
CCCC[N+](CCCC)(CCCC)CCCC.[F-]	TBAF
Cl.CCN=C=NCCCN(C)C	EDC·HCl
NCCN.Cl.Cl	エチレンジアミン二塩酸塩
CC(=O)[O-].[NH4+]	酢酸アンモニウム
C[C@@H](C(=O)O)c1ccc2cc(OC)ccc2c1	ナプロキセン
CN[C@@H](C)[C@H](O)c1ccccc1	エフェドリン
CC(=O)O[C@H]1C[C@@H]2CC[C@H]1C2	ノルボルニルアセテート
O=C1O[C@H]([C@@H](O)CO)C(O)=C1O	アスコルビン酸
OC1=CC=C(C=C1)C1=CC=C(O)C=C1	ビフェノール
O=[N+]([O-])c1ccc(cc1)C(=O)O	4-ニトロ安息香酸
Brc1ccc(Br)cc1	1,4-ジブロモベンゼン
FC(F)(F)C(=O)O	TFA
CS(C)=O	DMSO
CC(C)(C)OC(=O)OC(=O)OC(C)(C)C	Boc2O
O=C(OC1CCCCC1)C(F)(F)F	トリフルオロ酢酸シクロヘキシル
C=CC(=O)OCCCC	アクリル酸ブチル
C=C(C)C(=O)OC	MMA
O=C1OCCO1	エチレンカーボナート
C1CCC2(CC1)OCCO2	シクロヘキサノンエチレンアセタール
O=S1(=O)CCCC1	スルホラン
CC1=NN(c2ccccc2)C(=O)C1	エダラボン
OC(=O)c1cccnc1	ニコチン酸
Nc1ncnc2[nH]cnc12	アデニン
O=c1[nH]cc(F)c(=O)[nH]1	5-FU
CC12CCC(CC1)C(C)(C)O2	1,8-シネオール
C[Si](C)(C)OC(C)(C)C	TMS エーテル
CCOC(=O)/C=C/C(=O)OCC	フマル酸ジエチル
CCOC(=O)/C=C\\C(=O)OCC	マレイン酸ジエチル
OC(=O)/C=C/C=C/C	ソルビン酸
CCCCCC/C=C\\CCCCCCCC(=O)NCCO	パルミトレオイルエタノールアミド
C[C@H](N)C(=O)N[C@@H](C)C(=O)O	Ala-Ala
O=C(N[C@@H](Cc1ccccc1)C(=O)O)[C@@H]1CCCN1	Pro-Phe
OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl	ジクロフェナク
CN(C)C(=N)NC(N)=N	メトホルミン
COC(=O)c1ccccc1O	サリチル酸メチル
CCOC(=O)c1ccc(N)cc1	ベンゾカイン
Oc1ccc(O)cc1	ヒドロキノン
CCCCNC(=O)c1ccccc1	N-ブチルベンズアミド
[Na+].[Na+].[O-]C(=O)CCC([O-])=O	コハク酸二ナトリウム
N#Cc1cccc(c1)C#N	イソフタロニトリル
O=C(NC1CCCCC1)NC1CCCCC1	ジシクロヘキシル尿素
O=C(O)c1cccc(c1)C(=O)O	イソフタル酸
""".strip()


def main():
    rows = [l.split("\t") for l in SET.splitlines() if l.strip()]
    if "--quick" in sys.argv:
        rows = rows[:30]
    c.warmup()
    t0 = time.time()
    count = Counter()
    for smi, label in rows:
        t = time.time()
        r = c.smiles_to_name(smi)
        count[r.level] += 1
        print(f"{r.level:6s} {time.time() - t:4.1f}s  {label}: {r.name}", flush=True)
    n = len(rows)
    print(f"\n{n} 分子: " + ", ".join(f"{k} {v} ({v / n:.0%})" for k, v in count.most_common()),
          f" / 平均 {(time.time() - t0) / n:.1f} s/分子")


if __name__ == "__main__":
    main()
