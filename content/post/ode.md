---
title: "Neural Ordinary Differential Equationsのための常微分方程式入門"
date: 2019-03-10T11:39:28+09:00
draft: false
tags: ["数値計算", "数値積分"]
---
## Neural Ordinary Differential Equations
NeurIPS2018ベストペーパーとして選ばれた論文に、ResNetの中間層を微分方程式として見るという興味深い研究がありました[^Ricky2018]。  
この研究を理解する上で常微分方程式の数値計算手法を理解する必要があり勉強したので、自分なりにまとめてみたいと思います。  
<!-- NeurIPS論文は以下にリンクを貼っておきます。   -->
[^Ricky2018]: https://arxiv.org/abs/1806.07366

## 1　常微分方程式
簡単な常微分方程式を解いていくことでイメージを掴んでいきたいと思います。  
ここでは以下のような常微分方程式を考えます。
$$
y'=(1-t)y\ \ \ \ \ (1.1)
$$
従属変数$y$の導関数$y'$を含む方程式を「微分方程式」と呼びます。また、(1.1)のように独立変数が1つ（$t$のみ）の微分方程式を「常微分方程式」と呼びます。

### 常微分方程式（1.1）の解析的な解法
まず式（1.1）の両辺を$y$で割ります。
$$
\frac{y'}{y}=1-t
$$
次に両辺を$t$で積分します。
$$
\log{|y|}=-\frac{1}{2}(1-t)^{2}+C
$$
よって解は、
$$
y=A\exp{(-\frac{1}{2} (1-t)^{2})}\ \ \ \ \ (1.2)
$$
となります。

### 常微分方程式（1.1）の解の意味
{{< figure src="../../image/ode2.png" class="center" width="550" height="300" >}}
$y(0)$（初期値）で解$y(t)$が一意に定まります。

### 常微分方程式（1.1）の解を一意に定める
まず$t=0$での$y$の値を決めます（初期条件）。
$$
y(0)=1
$$
すると、式（1.2）から$A$の値が定まります。
$$
A=\sqrt{e}
$$
よって、以下のように解が一意に定まります。
$$
y=\exp{(t-\frac{t^2}{2})}
$$

## 2　常微分方程式を数値計算で解く
### 差分方程式
常微分方程式が与えられた時、毎回解析的に解けるとは限りません。そこで数値計算によって近似解を求めるというモチベーションがあります。以下では常微分方程式の数値計算手法を紹介します。
$$
y' = f(t, y)\ \ \ \ \ (2.1),\ \ \ \ y(0) = \alpha\ \ \ \ \ (2.2)
$$
1階常微分方程式（2.1）を初期条件（2.2）の下で解く初期値問題です。

微分をコンピュータ上で扱うので、小さな差分で表すことにします。つまり式（2.1）の左辺を差分商で表します。微分の定義を思い出してもらって極限はごく小さな差分$\Delta{t}$を用いると、
$$
\frac{1}{\Delta{t}} (y(t + \Delta{t}) - y(t)) = f(t, y(t))
$$
と表せます。

このようにして、近似的に表した微分方程式を「差分方程式」と呼びます。微分方程式と差分方程式を区別するために差分方程式では、$y(t)$でなく$Y(t)$と書きます。
$$
\frac{1}{\Delta{t}} (Y(t + \Delta{t}) - Y(t)) = f(t, Y(t))\ \ \ \ \ (2.3)
$$

### 格子点
差分方程式（2.3）に初期条件を与えます。
$$
Y(0)=\alpha
$$
式（2.3）を変形すると、
$$
Y(t+\Delta{t})=Y(t)+\Delta{t} \cdot f(t, Y(t))\ \ \ \ \ (2.4)
$$
となります。  
これに$t=0$を代入すると、
$$
Y(\Delta{t}) = Y(0) + \Delta{t} \cdot f(0, Y(0)) = \alpha + \Delta{t} \cdot f(0, \alpha)\ \ \ \ \ (2.5)
$$
より、$Y(0)$から直ちに$Y(\Delta{t})$が求まります。

同様にして、式（2.4）を繰り返し用いると、
$$
Y(2\Delta{t}) = Y(\Delta{t}) + \Delta{t} \cdot f(\Delta{t}, Y(\Delta{t}))\\\\\\
Y(3\Delta{t}) = Y(2\Delta{t}) + \Delta{t} \cdot f(\Delta{t}, Y(2\Delta{t}))\\\\\\
Y(4\Delta{t}) = Y(3\Delta{t}) + \Delta{t} \cdot f(\Delta{t}, Y(3\Delta{t}))\\\\\\
\cdots\\\\\\
Y((j+1)\Delta{t}) = Y(j\Delta{t}) + \Delta{t} \cdot f(\Delta{t}, Y(j\Delta{t}))
$$
と飛び飛びの時刻でのY(t)の値が求まります（格子点）。

この格子点を求める問題というのは、式（2.1）, (2.2)の微分方程式の問題を以下のように$Y\_{j}$に関する漸化式の問題に置き換えられたものとして扱えます。
$$
\frac{1}{\Delta{t}} (Y\_{j+1}-Y\_{j}) = f(t\_{j}, Y\_{j})\ \ \ \ \ (2.6)\\\\\\
Y\_{0} = \alpha
$$

### 微分解と差分解の関係
式（2.1）と（2.6）は別物ですが、$\Delta{t}$が小さければ$Y\_{j}$が$y(t\_{j})$と近い値になることが期待されます。
そこで初期値問題（1.1）から差分方程式の初期値問題を導き、元の問題と比較してみましょう。  
式（1.1）の初期値問題は、
$$
y'=(1-t)y\\\\\\
y(0) = 1
$$
これに対応する差分方程式の初期値問題は、
$$
\frac{1}{\Delta{t}} (Y\_{j+1}-Y\_{j}) = (1-t)Y\_{j}\\\\\\
Y\_{0}=1
$$
となります。

この時、以下の図で微分解はオレンジ線、差分解は青線（$\Delta{t}=0.2$）となります。
{{< figure src="../../image/ode3.png" class="center" width="500" height="300" >}}
$\Delta{t}$をどんどん小さくすれば、青線がどんどんオレンジ線に近づいていきます。その分計算時間はかかりますが。

このように微分を差分の問題に置き換え、格子点上の値を数値計算によって求め、得られた差分解から微分解を推定する手法を「差分法」と呼びます。

## 3　代表的な差分法のアルゴリズム

### オイラー法
件のNeurIPS論文でも用いらている常微分方程式の数値計算手法は、オイラー法と呼ばれるもっとも基本的な差分法アルゴリズムです。これは愚直に式（2.6）の漸化式を解くだけです。
$$
\frac{1}{\Delta{t}} (Y\_{j+1}-Y\_{j}) = f(t\_{j},\ Y\_{j})\ \ \ \ \ (3.1)\\\\\\
$$

### ホイン法
オイラー法よりも少し複雑に右辺の値を求めています。
$$
k\_{1} = f(t\_{j}, Y\_{j})\\\\\\
k\_{2} = f(t\_{j} + \Delta{t},\ Y\_{j} + \Delta{t} \cdot k\_{1})\\\\\\
\frac{1}{\Delta{t}} (Y\_{j+1}-Y\_{j}) = \frac{1}{2} (k\_{1} + k\_{2})\ \ \ \ \ (3.2)
$$

### ルンゲクッタ法
さらに複雑な形で右辺を求めています。
$$
k\_{1} = f(t\_{j},\ Y\_{j})\\\\\\
k\_{2} = f(t\_{j} + \frac{\Delta{t}}{2},\ Y\_{j} + \frac{\Delta{t}}{2} \cdot k\_{1})\\\\\\
k\_{3} = f(t\_{j} + \frac{\Delta{t}}{2},\ Y\_{j} + \frac{\Delta{t}}{2} \cdot k\_{2})\\\\\\
k\_{4} = f(t\_{j} + \Delta{t},\ Y\_{j} + \Delta{t} \cdot k\_{3})\\\\\\
\frac{1}{\Delta{t}} (Y\_{j+1}-Y\_{j}) = \frac{1}{6} (k\_{1} + k\_{2} + k\_{3} + k\_{4})\ \ \ \ \ (3.3)
$$

ホイン法もルンゲクッタ法も複雑に見えますが、計算としては$Y\_{j+1}$の値を求めるのに$Y\_{j}$の値しか使っていません。このような方法を1段階法と呼び、$Y\_{j+1}$の値を求めるのに$Y\_{j},\ Y\_{j-1}$のように複数前の値まで使う方法を多段階法と呼びます。本記事では1段階法しか扱いません。

### Pythonによる実装
以下のような常微分方程式を考えます。
$$
y' = 2 t y - 2\\\\\\
y(0) = 2
$$
これを上で紹介した3つの差分法アルゴリズムによって計算し、比較するプログラムです。  
$\Delta{t}$を0.2と0.02の場合とで分けています。

```python
"""
変数係数常微分方程式
y' = 2*t*y - 2
y(0) = 2
"""

def main():
    print("オイラー法、dt=0.2の場合")
    euler(f, Y=2, T=2, N=10, dt=2/10)
    print()
    
    print("オイラー法、dt=0.02の場合")
    euler(f, Y=2, T=2, N=100, dt=2/100)
    print()

    print("ホイン法、dt=0.2の場合")
    haun(f, Y=2, T=2, N=10, dt=2/10)
    print()
    
    print("ホイン法、dt=0.02の場合")
    haun(f, Y=2, T=2, N=100, dt=2/100)
    print()

    print("ルンゲクッタ法、dt=0.2の場合")
    runge_kutta(f, Y=2, T=2, N=10, dt=2/10)
    print()

    print("ルンゲクッタ法、dt=0.02の場合")
    runge_kutta(f, Y=2, T=2, N=100, dt=2/100)
    print()

# 入力関数
def f(t, y):
    return 2*t*y - 2

# 入力：y' = f(y, t), y(0) = a, T=2, N：分割数, dt:T/N
# オイラー法
def euler(f, Y, T, N, dt):
    for j in range(0, N):
        t = j * dt
        Y = Y + dt * f(t, Y)
        if round(10*(t+dt), 4) % 2 == 0:
            print("f(%.02f) = %f" % (t+dt, Y))

# ホイン法
def haun(f, Y, T, N, dt):
    for j in range(0, N):
        t = j * dt
        
        k1 = f(t, Y)
        k2 = f(t + dt, Y + (dt * k1))
        
        Y = Y + (k1 + k2) * dt / 2
        if round(10*(t+dt), 4) % 2 == 0:
            print("f(%.02f) = %f" % (t+dt, Y))

# ルンゲクッタ法
def runge_kutta(f, Y, T, N, dt):
    for j in range(0, N):
        t = j * dt

        k1 = f(t, Y)
        k2 = f(t + dt / 2, Y + dt * k1 / 2)
        k3 = f(t + dt / 2, Y + dt * k2 / 2)
        k4 = f(t + dt, Y + dt * k3)

        Y = Y + (k1 + 2*k2 + 2*k3 + k4) * dt / 6
        
        if round(10*(t+dt), 4) % 2 == 0:
            print("f(%.02f) = %f" % (t+dt, Y))

if __name__ == "__main__":
    main()
```

### 結果
実験の結果は以下のようになります。
```
オイラー法、dt=0.2の場合
f(0.20) = 1.600000
f(0.40) = 1.328000
f(0.60) = 1.140480
f(0.80) = 1.014195
f(1.00) = 0.938738
f(1.20) = 0.914233
f(1.40) = 0.953064
f(1.60) = 1.086781
f(1.80) = 1.382320
f(2.00) = 1.977591

オイラー法、dt=0.02の場合
f(0.20) = 1.663896
f(0.40) = 1.442943
f(0.60) = 1.311958
f(0.80) = 1.265990
f(1.00) = 1.322528
f(1.20) = 1.534956
f(1.40) = 2.028438
f(1.60) = 3.089250
f(1.80) = 5.392041
f(2.00) = 10.604015

ホイン法、dt=0.2の場合
f(0.20) = 1.664000
f(0.40) = 1.442330
f(0.60) = 1.310488
f(0.80) = 1.263748
f(1.00) = 1.319577
f(1.20) = 1.530870
f(1.40) = 2.020671
f(1.60) = 3.067178
f(1.80) = 5.315537
f(2.00) = 10.326220

ホイン法、dt=0.02の場合
f(0.20) = 1.670727
f(0.40) = 1.455854
f(0.60) = 1.332365
f(0.80) = 1.298135
f(1.00) = 1.375878
f(1.20) = 1.630357
f(1.40) = 2.213779
f(1.60) = 3.480998
f(1.80) = 6.291466
f(2.00) = 12.841209

ルンゲクッタ法、dt=0.2の場合
f(0.20) = 1.670793
f(0.40) = 1.455986
f(0.60) = 1.332578
f(0.80) = 1.298462
f(1.00) = 1.376392
f(1.20) = 1.631203
f(1.40) = 2.215223
f(1.60) = 3.483448
f(1.80) = 6.295094
f(2.00) = 12.843333

ルンゲクッタ法、dt=0.02の場合
f(0.20) = 1.670782
f(0.40) = 1.455968
f(0.60) = 1.332556
f(0.80) = 1.298445
f(1.00) = 1.376407
f(1.20) = 1.631343
f(1.40) = 2.215832
f(1.60) = 3.485783
f(1.80) = 6.303834
f(2.00) = 12.876270
```
WolframAlphaでこの計算を解くと12.8763…となるので、ルンゲクッタ法（$\Delta{t}=0.02$）の場合がもっとも良く近似できていそうだと分かります。

## 終わりに
今回は常微分方程式の数値計算手法をざっと眺めてみました。NeurIPS論文の方も噛み砕き次第、まとめてみたいと思います。  
実はこちらのスライド[^Ricky2018_slide]が非常によくまとまっているのでこれを読めば良いという話もあります。

[^Ricky2018_slide]: https://www.slideshare.net/DeepLearningJP2016/dlneural-ordinary-differential-equations

## 参考
本文の説明やコードを書く際の擬似コードは以下の書籍を参考にさせて頂きました。

- 数値計算｜高橋大輔｜岩波書店｜1996