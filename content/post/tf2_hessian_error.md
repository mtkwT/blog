---
title: エラーメモ:tf2のSparseCategoricalCrossentropyではヘッセ行列を計算できない
date: 2020-05-06T20:00:38+09:00
draft: false
tags: ["深層学習", "ヘッセ行列", "tensorflow"]
---
### 環境（Dockerhub）
- tensorflow/tensorflow:2.1.0-gpu-py3-jupyter

### エラーログ
```
LookupError: Gradient explicitly disabled. Reason: b"Currently there is no way to take the second derivative of sparse_softmax_cross_entropy_with_logits due to the fused implementation's interaction with tf.gradients()"
```
### 原因
ヘッセ行列を計算するloss関数にtf.keras.losses.SparseCategoricalCrossentropyを使用していたこと。
2020-05-06現在、最新のtensorflow2.1.0ではtf.keras.losses.SparseCategoricalCrossentropyのヘッセ行列を標準では計算できない。

### 解決方法
tf.keras.losses.SparseCategoricalCrossentropyの代わりに、tf.keras.losses.CategoricalCrossentropyを使用する。tf.keras.losses.SparseCategoricalCrossentropyはラベルがワンホットエンコードされていない状態（スカラー値）の時に使用し、tf.keras.losses.CategoricalCrossentropyはラベルがワンホットエンコードされている状態（ベクトル）で使用するので、データセットにおけるラベルのreshapeが必要になる。