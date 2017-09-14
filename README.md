# 講義２回目(2017/9/14)

## お願いしたいこと

+ 準備
  + https://github.com/hiko2msp/nlp をダウンロードもしくはgit cloneしておいてください
  + 追加のPythonライブラリ

    ```
    pip install gensim
    ```
+ 今日はこちらの資料を使って進めたいと思います。

## 今回のゴール

+ テキストの分類モデルを使ったbotを作れるようになっている
+ Word2Vecを理解し、Word2Vecを使ったbotを作れるようになっている

## 目次

+ 前回の復習(10min)
  + 形態素解析について
+ 基礎の復習(30min)
  + ベクトルの類似度・距離について
    + 類似度・距離の説明
    + 類似度をつかったデモ
      + 実習(デモを動かしてみる)
  + bag of wordsを使ってロジスティック回帰でテキスト分類を行うデモ
    + 実習(デモを動かしてみる)
+ Word2Vec(40min)
  + Word2Vecについて
  + Word2Vecを使ったデモ
    + 実習(デモを動かしてみる)
+ LSTMについて(10min)
  + LSTMの説明

## 基礎の復習

### 類似度・距離の説明

+ ベクトルとベクトルの類似度(または距離)を計測する方法
  + L1距離(距離は類似度と逆で、値が小さいほど似ていると考える)

    $ \sum_i |x_i - y_i| $

    ```python
    x_vector = [1, 2, 3]
    y_vector = [4, 5, 6]
    def l1_distance(x_vector, y_vector):
      return sum([ abs(x - y) for x, y in zip(x_vector, y_vector)])
    ```

  + L2距離

    $ \sqrt{\sum_i (x_i - y_i)^2} $

    ```python
    from math import sqrt
    x_vector = [1, 2, 3]
    y_vector = [4, 5, 6]
    def l2_distance(x_vector, y_vector):
      return sqrt(sum([ (x - y) * (x - y) for x, y in zip(x_vector, y_vector)]))
    ```

  + コサイン類似度(類似度は、値が大きいほど似ていると考える)

    $ \frac{\sum_i x_i \cdot y_i}{\sqrt{\sum_i{x_i^2} \cdot \sum_i{y_i^2}}} $

    $ \frac{x \cdot y}{||x||_2 \cdot ||y||_2 } $

    ```python
    from math import sqrt
    x_vector = [1, 2, 3]
    y_vector = [4, 5, 6]
    def l2_norm(vector):
      return sqrt(sum([elem**2 for elem in vector]))
    def cos_similarity(x_vector, y_vector):
      return sum([ x * y for x, y in zip(x_vector, y_vector)]) / (l2_norm(x_vector) * l2_norm(y_vector)) 
    ```

### 類似度をつかったデモ

[プログラム](bow_similarity.py)

+ 準備
  
  ```
  pip install scipy
  pip install gensim
  pip install sklearn
  pip install six
  pip install janome
  ```


+ 夏目漱石の小説(吾輩は猫である, 坊っちゃん)をそれぞれBag of Wordsに変換します
+ 入力した文字列もBag of Wordsに変換します
+ 入力した文字列が、どちらの小説と似ているかを出力します

+ 出力

  ```
  Found and verified 789_ruby_5639.zip
  Found and verified 752_ruby_2438.zip
  入力文字: 吾輩は犬である
  候補の小説:
  ['吾輩は猫である', '坊っちゃん']
  最も似ている小説: 吾輩は猫である
  ```

### bag of wordsを使ってロジスティック回帰でテキスト分類を行うデモ

[プログラム](bow_logistic_regression.py)

+ 準備
  
  ```
  pip install scipy
  pip install gensim
  pip install sklearn
  pip install six
  pip install janome
  ```

+ 4つのニュース記事をそれぞれBag of Wordsにします
+ それぞれのニュースに対して、経済の記事かどうかのラベルをつけます
+ ラベルをつけた教師データを使って、経済の記事かどうかを分類するロジスティック回帰モデルを学習します
+ 別の記事が入ってきたときに、その記事が経済の記事かどうかを分類します

+ 出力

  ```
  $ python bow_logistic_regression.py
  学習したパラメータ(特徴的なもの)
      token      coef
  40      ３ -0.236599
  147     目 -0.214302
  32      士 -0.142868
  103     ２ -0.118300
  50     連合  0.107633
  5       日  0.120704
  28      米  0.125380
  97     13  0.131370
  42    テレビ  0.133517
  98     転倒  0.133517
  111     件  0.133517
  110    東芝  0.143511
  16      の  0.165101
  入力テキスト:
  東芝は１３日、半導体子会社「東芝メモリ」の売却について、政府系の産業革新機構や 米ファンドのベインキャピタル、韓国半導体のＳＫハイニックスなどの「日米韓連合」 と、９月下旬の契約締結を目指して覚書を結んだ、と発表した。ただ、日米韓連合を「 排他的な交渉先としない」ともしており、他の売却先も引き続き検討する模様だ。
  ['経済でない', '経済']
  [ 0.19917571  0.80082429]
  ```

## Word2Vec

### Word2Vecの説明

+ Word2Vecは効率的に、機械学習のために使えるベクトルを学習することができる
+ Word2Vecには大きく2つのアプローチがある
  + Continuous Bag-of-Words model(CBOW)
    + CBOWは対象の単語をsource context wordsから予測する
    + 全体の文脈を一つの観測として扱うため、データ集合が少ない場合に有用である
  + Skip-gram model
    + Skip-gramはsource context wordsを対象の単語から予測する
    + context-targetペアを新しい観測としてあつかうので、データ集合が大きい時に有用である
+ 応用例
  + 基本的にはTfIDFで学習したベクトルと同じ使い方ができる
  + テキスト分類
  + 類義語・
+ 参考文献
  + [Word2Vec のニューラルネットワーク学習過程を理解する](http://tkengo.github.io/blog/2016/05/09/understand-how-to-learn-word2vec/)
  + [models.word2vec – Deep learning with word2vec](https://radimrehurek.com/gensim/models/word2vec.html)
  + [Word2Vec Tutorial Part II: The Continuous Bag-of-Words Model](http://mccormickml.com/assets/word2vec/Alex_Minnaar_Word2Vec_Tutorial_Part_II_The_Continuous_Bag-of-Words_Model.pdf)

+ ツール
  + [fasttext](https://github.com/facebookresearch/fastText)
  + [gensim](https://radimrehurek.com/gensim/tut1.html)
  + [spyCy](https://spacy.io/)

### Word2Vecを使ったデモ

[プログラム](word2vec_similarity.py)

+ 準備

  ```
  pip install gensim
  pip install pyemd
  ```

+ 夏目漱石の小説(吾輩は猫である, 坊っちゃん)のテキストを使って、WordVectorを学習します
+ 二つの小説について、それぞれテキスト内の単語のベクトルを平均して、ドキュメントベクトルを作ります
+ 入力した文字列についても、単語のベクトルの平均ベクトルを計算します。
+ 類似度を求めて、入力した文字列が、どちらの小説と似ているかを出力します

### LSTM

+ 言語の生成モデル。前の単語列から、次の単語を予測する
+ 応用例
  + 翻訳などに利用される
  + 音声認識
  + 画像からのキャプション生成
+ 参考文献
  + [わかるLSTM](http://qiita.com/t_Signull/items/21b82be280b46f467d1b)


### 今回の課題

1. Bag of Wordsを使って、ロジスティック回帰を使った分類モデルを作ってみましょう
1. Slackに投稿した内容を入力として、分類結果を返すbotを作りましょう
  + 例:
    + いいね！システム: 入力テキストがいい内容か悪い内容かを分類し、いい内容であればいいね！と返す
    + 作詞者予測システム: 入力テキストが、どの人の歌詞に近いかを判定する
1. Word2Vecでモデルを学習してみましょう
1. Word2Vecで類似した単語・テキストを返すbotを作りましょう

### 参考文献

+ [Python用のトピックモデルのライブラリgensim の使い方(主に日本語のテキストの読み込み)](http://sucrose.hatenablog.com/entry/2013/10/29/001041)
