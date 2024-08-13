# Sigmoid_Attenion_ViT
MIRU2024でのポスター発表であった**Sigmoid AttentionによるAttentionの修正機構を導入したDINOの提案及びHuman in the loopによる精度向上の試み**の論文の再現実装を行う．<br>
論文では，DINOによる自己教師あり学習手法による評価実験を行なっており，独自の工業製品画像のデータセットを使用している(論文3.1).<br>
そこで，CIFAR-10のような一般的な画像での有効性の確認を行う．CIFAR-10はラベル付きのデータセットであるため，学習法はDINOではなく，通常の教師あり学習として学習する．<br>

## モデル
使用するモデルは，論文同様`ViT-Small/16`を使用する．<br>
ソースコードや事前学習済みの重みは，timmライブラリで公開されているものを活用する．
### モデルの準備
以下のコマンドを実行してtimmで公開されている事前学習済みのモデルの重みをローカルに保存します．<br>
デフォルトの保存先は，`models/vit_small_patch16_224.pt`です．
```
cd Sigmoid_Attention_ViT
python3 src/load_pretrained.py 
```
