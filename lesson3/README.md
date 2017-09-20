# 講義３回目

## ゴール

+ RNNについて理解する
+ LSTMを理解する
+ LSTMを使った文章生成を実践する

## 内容


## デモ

準備

```
$ pip install janome
$ pip install tensorflow
```

### 1. データの取得

[create_natsume_data.py](create_natsume_data.py)

夏目漱石の小説(吾輩は猫である、坊っちゃん、こころ)をダウンロードし、形態素解析し、テキストファイルに保存します

```
$ python create_natsume_data.py
Found and verified 789_ruby_5639.zip
Found and verified 752_ruby_2438.zip
Found and verified 773_ruby_5968.zip
```

### 2. 文章生成モデルの学習

[natsume_word_lm.py](natsume_word_lm.py)

夏目漱石の小説(吾輩は猫である、坊っちゃん)を学習データ、小説(こころ)を検証データとして利用して、文章生成モデルを学習します。

+ GPUを使わない場合

  ```
  $ python natsume_word_lm.py --num_gpus=0
  ```

+ GPUを使う場合

  ```
  $ python natsume_word_lm.py
  Epoch: 1 Learning rate: 1.000
  0.058 perplexity: 6034.395 speed: 11167 wps
  0.158 perplexity: 2213.151 speed: 17699 wps
  0.257 perplexity: 1514.663 speed: 20862 wps
  0.357 perplexity: 1209.611 speed: 22662 wps
  0.456 perplexity: 1035.825 speed: 24194 wps
  0.556 perplexity: 916.692 speed: 25057 wps
  0.655 perplexity: 837.670 speed: 25740 wps
  0.754 perplexity: 771.061 speed: 26218 wps
  0.854 perplexity: 720.813 speed: 26471 wps
  0.953 perplexity: 688.813 speed: 26755 wps
  Epoch: 1 Train Perplexity: 674.196
  Epoch: 1 Valid Perplexity: 237.986
  ...
  Epoch: 13 Train Perplexity: 49.212
  Epoch: 13 Valid Perplexity: 125.326
  Test Perplexity: 125.131
  Saving model to result.
  ```

### 3. 文章生成

[predict_next_word.py](predict_next_word.py)

学習した文章生成モデルを使って、'吾輩は'から始まる文章を生成します

```
$ python predict_next_word.py
吾輩は
吾輩は人並
吾輩は人並。
```

## 課題

+ LSTMを使った文章生成を利用したSlack botを作る
