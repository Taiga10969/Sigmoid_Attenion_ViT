# Sigmoid_Attenion_ViT
MIRU2024でのポスター発表であった論文：**Sigmoid AttentionによるAttentionの修正機構を導入したDINOの提案及びHuman in the loopによる精度向上の試み**の再現実装を行う．<br>
論文では，DINOによる自己教師あり学習手法による評価実験を行なっており，独自の工業製品画像のデータセットを使用している(論文3.1).<br>
そこで，CIFAR-10のような一般的な画像での有効性の確認を行う．CIFAR-10はラベル付きのデータセットであるため，学習法はDINOではなく，通常の教師あり学習として学習する．<br>

## 準備
使用するモデルは，論文同様`ViT-Small/16`を使用する．<br>
ソースコードや事前学習済みの重みは，timmライブラリで公開されているものを活用する．
### 環境構築/モデルの準備
以下のコマンドを実行してtimmで公開されている事前学習済みのモデルの重みをローカルに保存する．<br>
デフォルトの保存先は，`models/vit_small_patch16_224.pt`である．optionで`--model_name`，`--save_dir`でtimmのモデル名と重みの保存先を指定可能．
```
cd Sigmoid_Attention_ViT
pip install -r requirements.txt
python3 src/load_pretrained.py 
```
timmライブラリのソースコード/重みパラメータを使用したローカルでの動作確認は，`local_model_test.py`を実行．

### Datasetの準備
データセットは，CIFAR-10と[CUB_200_2010](https://www.vision.caltech.edu/datasets/cub_200_2010/)を使用する．
<!-- 
CIFAR-10データセットでのViT学習時には，自動的にダウンロードされるようになっているが，CUB_200_2010の学習には，事前にデータセットをダウンロードしておく必要がある．<br>
[ここ](https://data.caltech.edu/records/65de6-vp158)からデータセットをダウンロードして任意の場所に保存してください．
-->

## Sigmoid Attention
論文で提案されている手法はSigmoid関数を通すことで一度0-1の範囲に正規化し，その上でHuman in the loopによるAttentionの修正を行う．
そして，逆Sigmoid関数に通すことで，元のlogitの値へと再変換する．これにより，ユーザーが内積値のlogitではなく0-1の範囲で修正が可能となり，Softmaxの前での処理のため，Attention Weightの行方向の和が1であることも担保される．


## Step1：学習(NOT Human in the loop)⚠️この先現在コード整理中（非公開）
Human in the loopによる精度向上を行う前に，通常のViTのモデルをFTして精度を確認する．
### CIFAR-10
CIFAR-10データセットを用いた10クラス分類タスクでのFTは以下のコマンドで行う．
```
bash run_vit_training.sh
```
⚠️この時，`run_vit_training.sh`の設定値でwandbをtrueにすることでwandbで学習経過を確認できる．(WANDB_KEYにAPIキーを指定してください．)<br>
学習結果は，`./result`に保存されます．
- CIFAR-10の学習設定例
```
# ViT-Trainingの設定
PROJECTS_NAME="Sigmoid_ViT_Training"
RUNS_NAME="cifar10_FT"
DATASET="cifar10"
BATCH_SIZE=512
MODEL_NAME="vit_small_patch16_224"
IMG_SIZE=224
IS_DATAPARALLEL=false
LR=1e-4
OPT="adam"
EPOCHS=10
WANDB=true
WANDB_KEY="your_wandb_api_key"
```
NVIDIA RTX 6000Ada GPU 1枚で動作可能であることを確認済みです．（Memory-Usage:33401MiB）

### CUB_200
CUB_200データセットを用いた200クラス分類タスクでのFTもCIFAR-10と同様のコマンドで行う．<br>
このとき，`run_vit_training.sh`の設定値を以下のように変更してから学習を行う．<br>
学習結果は，`./result`に保存されます．
- CIFAR-10の学習設定例
```
# ViT-Trainingの設定
PROJECTS_NAME="Sigmoid_ViT_Training"
RUNS_NAME="cub200_FT"
DATASET="cub200"
DATASET_PATH="your_path/dataset/cub200"
BATCH_SIZE=128
MODEL_NAME="vit_small_patch16_224"
IMG_SIZE=224
IS_DATAPARALLEL=false
LR=1e-5
OPT="adam"
EPOCHS=500
WANDB=true
WANDB_KEY="your_wandb_api_key"
```
### 学習結果の確認
学習結果の確認は，`result.ipynb`を実行することで，テストデータにおける正解率による評価と，サンプルデータにおけるAttentionを用いた判断根拠の可視化による評価を行っている．<br>
このファイルを実行すると，`./result./result%Y%m%d_%H%M`の形式で結果画像が保存されます．

## Step2:HITL学習(Human in the loop)
