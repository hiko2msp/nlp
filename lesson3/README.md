# 講義３回目

## ゴール

+ RNNについて理解する
+ LSTMを理解する
+ LSTMを使った文章生成を実践する

## 内容


## デモ

1. データの取得

  [create_natsume_data.py](create_natsume_data.py)

  夏目漱石の小説(吾輩は猫である、坊っちゃん、こころ)をダウンロードし、形態素解析し、テキストファイルに保存します

2. 文章生成モデルの学習

  [natsume_word_lm.py](natsume_word_lm.py)

  夏目漱石の小説(吾輩は猫である、坊っちゃん)を学習データ、小説(こころ)を検証データとして利用して、文章生成モデルを学習します。

3. 文章生成

  [predict_next_word.py](predict_next_word.py)

  学習した文章生成モデルを使って、'吾輩は'から始まる文章を生成します

## 課題

+ LSTMを使った文章生成を利用したSlack botを作る
