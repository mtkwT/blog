---
title: "編集距離（Levenshtein Distance）の正規化手法"
date: 2019-03-09T16:20:49+09:00
draft: false
tags: ["自然言語処理", "機械学習"]
---
### 編集距離（Levenshtein Distance）
自然言語処理では2つの文字列間の類似度を測りたい場面がよくあります。  
文章分類、スパムメール検知などが有名なタスクですね。類似度というと、コサイン類似度が有名でよく用いられます。確かにコサイン類似度はかなり便利ですが、2つの文字列を固定長のベクトルにしないと測れません。  
そこで、もっと簡単に文字列間の距離を測る尺度として"編集距離（Levenshtein Distance）"というものが使われます。

### 編集距離の定義と例
Wikipediaでは以下のような定義がされています。   
「1文字の挿入・削除・置換によって、一方の文字列をもう一方の文字列へ変形するのに必要な手順の最小回数として定義される」  
たとえば、「機械学習」と「器械体操」という2つの文字列間の編集距離は3です。これは「器械体操」という文字列において「器 -> 機」、「体 -> 学」、「操 -> 習」という3回の置換で「機械学習」が得られるので、編集距離は3です。3回以下の操作では「器械体操」を「機械学習」にすることはできません。  

### pythonによる実装
実装は動的計画法によるものです。計算量は2つの文字列の長さをn,mとするとO(nm)です。

```python
def levenshtein_disrtance(str1, str2):
    length1 = len(str1)
    length2 = len(str2)

    dp_matrix = [[0] * (length2 + 1) for _ in range(length1 + 1)]

    for row in range(length1 + 1):
        dp_matrix[row][0] = row

    for col in range(length2 + 1):
        dp_matrix[0][col] = col

    for row in range(1, length1 + 1):
        for col in range(1, length2 + 1):
            if str1[row - 1] == str2[col - 1]:
                cost = 0
            else:
                cost = 1
            
            dp_matrix[row][col] = min(dp_matrix[row - 1][col] + 1,          # 文字の挿入
                                      dp_matrix[row][col - 1] + 1,          # 文字の削除
                                      dp_matrix[row - 1][col - 1] + cost)   # 文字の置換

    return dp_matrix[length1][length2]
```

### 正規化
今のままでは、たとえば2つの文字列が3文字同士でまったく異なる文字列と、1000文字同士で異なる部分が3箇所の場合、編集距離はどちらも3です。普通の感覚からすると、後者の場合の方が類似度という意味では近いと感じるかと思います。そこで正規化の出番です。距離なので、正規化をする場合よく使われるのは大きい方の文字列の長さで割るというものです。これにより先ほどの場合だと、前者は3/3 = 1、後者は3/1000 = 0.003となり後者の方が、距離が近いという感覚に合う結果となります。

また、より後者の方が近いんだぞという意味合いを強く評価したい場合もあります。そのような場合、負の編集距離を指数関数に放り込むという手法も考えられます。下図のようにexp(-x)のグラフを考えると、距離xが0に近いほどexp(-x)の値は1に近づき、xの値が大きくなればなるほど加速度的に0に近くなっていきます。この際、あらかじめ大きい方の文字列の長さで割った編集距離をxとしても良いし、何らかのパラメータ（たとえば正規分布にしたがっていそうなら、文字列集合の長さの平均値）で割ったものをxとしても良いと思います。また、指数関数による変換を行うと距離ではなく、類似度（0が遠い、1が近い）として扱うことになります。

<!-- ![hugo-img1](/image/exp.png) -->

実装としてはnumpyなりmathなりをimportして最後のreturn部分のみ変えればOKです。

```python
return dp_matrix[length1][length2] / np.max(length1, length2) # 長さによる正規化
return np.exp(-dp_matrix[length1][length2] / param)           # 指数関数による正規化
```

### 参考
以下のサイトやブログ記事を参考にさせていただきました。

- https://takuti.me/note/levenshtein-distance/
- https://xaro.hatenablog.jp/entry/2017/02/24/005126
- http://d.hatena.ne.jp/naoya/20090329/1238307757
- https://ja.wikipedia.org/wiki/レーベンシュタイン距離