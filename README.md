<h1 align="center">Sigmoid_Attenion_ViT</h1>
<!--p align="center">hogehoge</p-->
MIRU2024でのポスター発表であった論文：**Sigmoid AttentionによるAttentionの修正機構を導入したDINOの提案及びHuman in the loopによる精度向上の試み**の再現実装を行う．
論文では，DINOによる自己教師あり学習手法による評価実験を行なっており，独自の工業製品画像のデータセットを使用している(論文3.1).<br>
そこで，公開されているデータセットを用いてSigmoid Attentionを利用したAttentionの修正機構の有効性の確認を行う．本実験で用いるCUB_200_2011データセットは，200クラスの詳細画像分類タスク用のデータセットでlabelが付与されているため，学習法はDINOではなく，通常の教師あり学習として学習し，詳細画像分類タスクにおける教師あり学習への応用を検証する．<br>

---

## 準備
使用するモデルは，論文同様`ViT-Small/16`を使用する．<br>
ソースコードや事前学習済みの重みは，timmライブラリで公開されているものを活用する．
### 環境構築/モデルの準備
以下のコマンドを実行してtimmで公開されている事前学習済みのモデルの重みをローカルに保存する．<br>
デフォルトの保存先は，`models/vit_small_patch16_224.pt`である．<br>
optionで`--model_name`，`--save_dir`でtimmのモデル名と重みの保存先を指定可能．
```
cd Sigmoid_Attention_ViT
pip install -r requirements.txt
python3 src/load_pretrained.py 
```
<!--timmライブラリのソースコード/重みパラメータを使用したローカルでの動作確認は，`local_model_test.py`を実行．-->

### Datasetの準備
[CUB_200_2011](https://www.vision.caltech.edu/datasets/cub_200_2011/)データセットを使用して実験を行う．<br>
※コード上では，CUB_200_2010データセットでの実験も収録されているが，CUB_200_2010データセットのBubbleデータはprivateであるためここでは詳しく説明しない．<br>
この先の実験進めるにはCUB_200_2011のデータセットを事前にダウンロードしておく必要がある．<br>
[ここ](https://data.caltech.edu/records/65de6-vp158)からデータセットをダウンロードして`./datasets`の下に置く．<br>
また，CUB_200_2011データセットに対応するBubbleデータは，[ここ](https://github.com/yaorong0921/CUB-GHA)からダウンロードして`./datasets`の下に置く．<br>

#### データセットの前処理
以下のコマンドを実行して，データを学習データと検証用データに分割する．分割の際に，各クラス毎のデータセットの枚数が異なるため，各クラス毎にN:(1-N)の割合で分割する．
引数として`--shuffle`を与えることで，クラス毎に分割する際にクラス内でデータをシャッフルしながら分割する．
```
python3 src/make_cub200_2011.py --N 0.8 --shuffle
```
これによりTrainとTextに分けられ`["img_path", label, "Bubble_data_name"]`でリスト化された`./datasets/cub200_2011_dataset.json`が保存される．

## Sigmoid AttentionによるAttentionの修正機構したVision Transformer
論文で提案されている手法はSigmoid関数を通すことで一度0-1の範囲に正規化し，Human in the loopによるAttentionの修正を行う．
これによりユーザーが0-1の範囲でAttentionの修正を行うことを可能にしている．
さらに，逆Sigmoid関数に通して元のlogitの値へと再変換して後続のSoftmax関数による処理を行うため，Attention Weightの行方向の和が1であることが担保される．
また，論文では通常のDINOでの学習後に，人間がAttentionを修正したい数枚のみをアノテーションしHuman in the loopを行う．<br>
しかし，本実験では，CUB_200_2010データとそれに対応するBubbleデータを用いて人間の注視領域と同じ部分に注目したAttentionを獲得するように学習を行う．
そのため，通常学習で人間によって理解できないAttentionにを修正するのではなく，全てのデータにおいて人間の注視領域と同じ部分に注目するように学習を行う．
- Step1:学習(NOT Bubble Data) / 通常のFTによりモデルをデータセットにFitさせる．
- Step2:HITL学習(Bubble Data) / Bubbleデータを用いて人の注目領域と同じ部分に注目するように追加学習

## Step1：学習(NOT Bubble Data)
Bubble DataによるAttentionの学習を行う前に，通常のViTのモデルをFTして精度を確認する．<br>
以下のコマンドにより学習を開始する．wandbで学習進捗を確認するには`run_vit_trainig.sh`の`WANDB_KEY`などを適切に変更しておく必要がある．
```
bash scripts/run_vit_training.sh
```

### 学習結果の確認
<!--学習結果の確認は，`result.ipynb`を実行することで，テストデータにおける正解率による評価と，サンプルデータにおけるAttentionを用いた判断根拠の可視化による評価を行っている．<br>
このファイルを実行すると，`./result./result%Y%m%d_%H%M`の形式で結果画像が保存されます．-->

## Step2:HITL学習(Bubble Data)
