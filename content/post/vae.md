---
title: "VAEの理論とTensorflowによる実装~~Fashion-MNISTの生成~~"
date: 2019-06-21T23:12:26+09:00
draft: false
tags: ["深層学習", "生成モデル"]
---
## VAEとは
VAE（Variable Auto Encoder）
は深層生成モデルの一種です。深層生成モデルの神童、GANが設定する確率分布は暗黙的ですが、こちらのVAEは明示的に確率分布を設定します。VAEはAdamを開発したことで有名なDiederik P. Kingma大先生が考案したモデルです。今日はKingma先生の作ったVAEを同じくKingma先生が作ったAdamで学習させましょう。

## VAEの理論概説
ここでは軽くVAEの理論的なお話をします。

VAEでは通常の生成モデル$p(x|\theta)$に潜在変数を加えます。
つまり、$p(x,z|\theta)$を考えます。

ただし、潜在変数$z$は訓練データ中には存在しないので確率の乗法定理と周辺化によって以下のように対数尤度$\log{p(x|\theta)}$を変形します。
$$
\log{p(x|\theta)} = 
\log{\int{p(x,z|\theta)dz}} =
\log{\int{p(x|z,\theta)p(z)dz}}
$$

これでもまだ計算が困難なのでEMアルゴリズムにおける変分下界を考えます。変分下界は以下のように表せます。
$$
\log{p(x|\theta)} = 
\log{\int{p(z|x,\hat{\theta}) \frac{p(x,z|\theta)}{p(z|x,\hat{\theta})} dz}} +
D[p(z|x,\hat{\theta})|p(z|x,\theta)]
\equiv
L(x;\hat{\theta},\theta) +
D[p(z|x,\hat{\theta})|p(z|x,\theta)]
$$

上式における最右辺の第1項$L(x;\hat{\theta},\theta)$が変分下界で、第2項$D[p(z|x,\hat{\theta})|p(z|x,\theta)]$はKL-Divergenceです。
KL-Divergenceは非負なので、変分下界を最大化することが対数尤度を最大化することになるというのがEMアルゴリズムの根拠です。

VAEではこの$p(z|x,\hat{\theta})$に任意の分布$q(z|x,\phi)$を用いても良いという一般化を施します。
すると、VAEにおける変分下界は分布$q(z|x,\phi)$の期待値として以下のように展開されます。

\\(
L(x;\phi,\theta) = 
E[ \log{\frac{p(x,z|\theta)}{q(z|x,\phi)}} ]\\\\\\
= E[ \log{\frac{p(x,z|\theta)}{q(z|x,\phi)}} ]\\\\\\
= E[ \log{p(x,z|\theta)} - \log{q(z|x,\phi)} ]\\\\\\
= E[ \log{p(x|z,\theta)} + \log{p(z|\theta)} - \log{q(z|x,\phi)}]\\\\\\
= E[ \log{p(x|z,\theta)} ] - E[ \log{q(z|x,\phi)} - \log{p(z|\theta)} ]\\\\\\
= E[ \log{p(x|z,\theta)} ] - D[ q(z|x,\phi)|p(z|\theta) ]
\\)

さて、これで目的関数を設定することができたので、この変分下界を最大化するようにDNNのパラメータを勾配を用いて更新すれば良いということになります。

しかし、まだ問題点があります。上式における変分下界の第1項
$E[ \log{p(x|z,\theta)} ]$
では推論モデルについての期待値になっています。
$z$をサンプリングするという行為は確率的な操作なので、そこに微分は定義されません。つまりニューラルネットワークでモデル化する場合、誤差逆伝播の際に勾配が期待値の中に入らないということになります。
そこで、VAEでは以下のようなリパラメトリゼーショントリックというテクニックを用いて勾配を期待値の中に押し込みます。推論モデルにはガウス分布を仮定します。
$$
z = \mu + \sigma * epsilon
$$

## TensorflowによるVAEの実装
以下のjupyter notebook上で結果も確認できます。

<a href="https://gist.github.com/mtkwT/c6991b38ce4584ba222fd74bd9f4ab82">Implementation of VAE by tensorflow</a>

潜在変数の次元や、ネットワーク構造を変えてみると結果も変わるので色々と試してみると興味深いと思います。

## VAEの欠点
上のnotebookの結果を見るとわかるように、ぼやけてしまって細かな部分を再現できていない画像もあります。
これは、VAEが確率分布を明示的にモデル化して最尤推定していることに原因があります。つまり生成分布をガウス分布としてモデル化するので、再構成誤差が二乗誤差になっています。すると、画素をはっきりとさせるよりもピクセル全体での誤差が小さくなるように学習が進みます。
