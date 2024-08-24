# Sigmoid_Attention_ViT/models

## `./models/vit.py`について
このファイルは，timmで提供されているVision Transformerクラスを基に以下の機能を追加したクラスである．<br>
モデルの読み込みには，model_configを用いて直接インスタンス化する形で行う．この時，モデルのパラメータは乱数によって初期化されたものになるため，学習済みのモデルパラメータなどを別途読み込む必要がある．このクラスの重み等は，基のtimmで提供されているものをそのまま使用可能である．パラメータの読み込みは，**環境構築/モデルの準備**で保存した`.pt`ファイルを読み込むことでtimmから読み込んだものと同様になる．<br>

### 変更点
- **Attention Weightの出力**
  - `forward()`時に`output_attentions=True`を渡すことで，各層のAttention Weightが出力される．
  - さらに，`only_Sigmoid=True`を渡すことで，最終層のAttention Weightのみ出力します．これは，学習時などにメモリ使用量を減らすために使用する．
- **Attentionの修正機構**
  - Attentionを修正して推論を行うために，Sigmoid Attention(最終層)のみのAttentionを操作できるように変更した．
  - 操作するAttentionの情報は，`forward()`時に`attn_info=`に変更情報を渡す．
