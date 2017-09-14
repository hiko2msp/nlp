# 講義２回目(2017/9/14)

## 目次

+ 前回の復習
  + 形態素解析について
+ 基礎の復習
  + ベクトルの類似度・距離について
    + 類似度・距離の説明
    + 類似度をつかったデモ
  + bag of wordsを使ってロジスティック回帰でテキスト分類を行うデモ
+ Word2Vecの説明
+ Word2Vecを使って分類モデルを作る
+ corpusの説明
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

## Word2Vec

+ Word2Vecは効率的に、機械学習のために使えるベクトルを学習することができる
+ Word2Vecには大きく2つのアプローチがある
  + Continuous Bag-of-Words model(CBOW)
    + CBOWは対象の単語をsource context wordsから予測する
    + 全体の文脈を一つの観測として扱うため、データ集合が少ない場合に有用である
  + Skip-gram model
    + Skip-gramはsource context wordsを対象の単語から予測する
    + context-targetペアを新しい観測としてあつかうので、データ集合が大きい時に有用である
+ Bag of WordsやTfIDFとの違い
  + 
+ 応用例
  + 基本的にはTfIDFで学習したベクトルと同じ使い方ができる
  + テキスト分類
  + 類義語・

### Word2Vecの学習

+ [Word2Vec のニューラルネットワーク学習過程を理解する](http://tkengo.github.io/blog/2016/05/09/understand-how-to-learn-word2vec/)
+ [Word2Vec Tutorial Part II: The Continuous Bag-of-Words Model](http://mccormickml.com/assets/word2vec/Alex_Minnaar_Word2Vec_Tutorial_Part_II_The_Continuous_Bag-of-Words_Model.pdf)

+ fasttext

### LSTM

+ 言語の生成モデル。前の単語列から、次の単語を予測する
+ [わかるLSTM](http://qiita.com/t_Signull/items/21b82be280b46f467d1b)


+ 翻訳などに利用される
+ 言語の生成モデルを学習することができる
  + 言語の生成モデルとは？過去の文字列から、次の文字列を生成するモデル


### 参考文献

+ [Python用のトピックモデルのライブラリgensim の使い方(主に日本語のテキストの読み込み)](http://sucrose.hatenablog.com/entry/2013/10/29/001041)
