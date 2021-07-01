---
print_background: true
---

# EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks

[TOC]

## Introduction
> 核心思想：有沒有一個規則或方法能夠擴增 ConvNet 並且維持一定的運算效率且又能提升準確度？

這篇論文是 2019 年由 Google 團隊所提出，一般在訓練 CNN 的網路時通常都是在以下三個因素上做調整來讓模型學習的更好：模型的深度(depth)、模型的寬度(width)、以及輸入圖片的解析度(resolution)。然而對於這三個因素的調整往往都是憑經驗或任意的調整，沒有一個系統性的調整方式。因此 Google 團隊提出了系統性擴增 (scale up) 整個 model 的方法 - compound scaling method，在實驗中基於 MobileNets 跟 ResNet 作為 baseline 來做擴增並且也透過 NAS (Neural architecture search) 的方式開發出 EfficientNet B0 作為新的 baseline 然後進一步擴增成一系列 B1~B7 的 EfficientNet。

從下圖的結果來看，的確 EfficientNet 系列不但參數比其他 ConvNet 還要少，並且準確度也有顯著的提升。論文也聲稱除了在 ImageNet 上有好的表現外，在其他幾個 datasets 一樣有不錯的成績。

![](https://i.imgur.com/CMnDneH.jpg)


## Compound Model Scaling
### Problem Formulation
回到論文的核心，根據一些經驗，作者認為只要以一定的比例縮放三個維度 (width / height / resolution)，就可以達到有效率的模型。因此以下開始定義問題：每一層的 ConvNet 可以被定義成 $Y_i = F_i(X_i)$，其中 $X_i$ 是維度為 $<H_i, W_i, C_i>$ 的 input。而 $H_i, W_i$ 為影像高寬、$C_i$ 為圖片的 channels。因此整個 model 的架構可以被寫成以下式子：

<span style="font-size:20px;">$$N = F_k\odot ...\odot F_2 \odot F_1(X_1) = \bigodot_{j=1...k}F_j(X_1)$$</span>

而通常 ConvNet 在設計的時候都是以一個 stage 為單位，在 stage 裡面的 layers 都會被設計成相同的架構 (例如 filter 的大小可能都是 3x3)，因此上式可以被改寫成：
<span style="font-size:20px;">$$N = \bigodot_{i = 1...s}F_i^{Li}(X_{<H_i,\ Wi,\ C_i>})$$</span>其中 $F_i^{Li}$ 代表在 stage $i$ 裡面 layer $F_i$ 被重複了 $L_i$ 次。而 $<H_i,\ Wi,\ C_i>$ 代表 X 在第 $i$ 層的 input shape。也就是說如果基於一個 baseline model 做 scaling，我們只需要考慮四個參數：$L_i,\ C_i,\ H_i,\ W_i$，而 $F_i$ (例如 filter 的 size) 將會是定值。更進一步精簡我們的設計，我們限制所有的 layers 都要依據某個固定的比例來做擴增，因此目標就是：**在給定的運算資源下，最大化我們的準確度**，以數學式來表式的話可以寫成：
<span style="font-size:20px;">$$\max_{d,w,r}Accuracy(N(d,w,r))$$$$s.t.\ N(d,w,r) = \bigodot_{i=1...s}\hat{F_i}^{d\cdot \hat{L_i}}(X_{<r\cdot \hat{H_i},\ r\cdot \hat{W_i},\ w\cdot \hat{C_i}>})$$</span>其中 $w, d, r$ 分別為 width, depth, resolution 所對應的係數。而 $\hat{F_i}, \hat{L_i}, \hat{H_i}, \hat{W_i}, \hat{C_i}$ 是 baseline model 最原始定義好的參數。

### Scaling Dimensions
接著問題來了，要最佳化 $w, d, r$ 是非常困難的，因為這三個參數彼此互相影響。因此傳統上通常只針對其中一個做調整，以下舉了一些例子。

**Depth (d)**：最常見的就是盡可能的增加深度，例如 VGG 跟 ResNet。而一味的增加深度會遇到的問題就是 vanishing gradient，雖然說可以用一些 skip connections 或 batch normalization 的技巧來降低這個問題，但你可能會發現只增加深度會讓 accuracy 的增加趨於平緩。例如 ResNet-1000 跟 ResNet-101 比起來巨雞巴深，但 accuracy 也才差沒多少。

**Width (w)**：也有一些 paper 主要在增加寬度，而一味的增加寬度會使得網路過淺 (wide but shallow) 而無法良好的捕捉高維度的特徵 (high level features)

**Resolution (r)**：增加輸入圖片的解析度也是常見的做法，從 ImageNet 的比賽可以發現，最早期的網路都是用 224 x 224 當作輸入，後來幾年的模型設計逐漸改成 299 x 299 以及 331 x 331，以及近期 (2018) 在 ImageNet 上有不錯表現的 GPipe 也使用了 480 x 480 作為輸入。

從上述的觀察以及實驗，得出**小結一：如果只針對三者中的單一參數做增加，都會遇到 accuracy 增加有限的問題**。如下圖由左到右分別對 $w, d, r$ 加大的結果。(按: FLOPS 這裡應該是指 FLOPs，理解為計算量或硬體資源的概念?)

![擷取](https://i.imgur.com/pUqE91P.jpg)

### Compound Scaling
根據直覺，假設 input 的 resolution 很高，那網路的深度以及寬度理應也要夠大才有辦法有效的學習到特徵。為了證明這個直覺，作者做了以下實驗：在不同的 depth 以及 resolution 組合下，width 的改變對於 accuracy 會有多少的影響。實驗結果如下圖所示，其中每一個點代表一種 width。觀察藍色曲線 (d = 1.0 / r = 1.0)，可以發現增加 width (越往右)，準確度上升的幅度有限。而如果比較紅色線與藍色線，在相同運算資源的情況下 (相同 FLOPS)，紅色點的準確度明顯高於藍色點。這些觀察得到了**小結二：單一的增加某個因素是無法達到最準確以及有效的模型，必須三個因素同時平衡的往上調整。**

![擷取](https://i.imgur.com/C4eJNJ0.jpg)

因此此篇論文提出了 **compound scaling method** 來最佳化上面幾個小節所提到的參數 $d,w,r$，也就是利用係數 $\phi$ 來均勻的放大網路的 depth, width, resolution，公式如下：
<span style="font-size:18px;">$$
depth:\ d = \alpha^\phi\\width:\ w = \beta^\phi\\resolution:\ r = \gamma^\phi\\s.t.\ \alpha\cdot\beta^2\cdot\gamma^2\approx 2\quad \quad \alpha\geq 1,\beta\geq 1,\gamma\geq 1
$$</span>其中 $\alpha,\beta,\gamma$ 是由 small grid search 所得到的常數，而 $\phi$ 是由使用者自己根據可使用的計算資源來手動調控的參數。另外，ConvNet 的運算量 FLOPS 會正比於 $d, w^2, r^2$，也就是說如果把 depth 增加一倍則 FLOPS 也會增加一倍，但如果把 width 或 resolution 增加一倍，則 FLOPS 會變為原本的四倍。換句話說，如果用上述式子透過 $\phi$ 來擴增網路，則對於總 FLOPS 的影響會是 $(\alpha\cdot\beta^2\cdot\gamma^2)^\phi$，在這篇 paper 裡面作者多加了一個限制條件使得 $\alpha\cdot\beta^2\cdot\gamma^2$ 會約等於 2，因此總 FLOPS 的增加量會約等於 $2^\phi$。

## EfficientNet Architecture
此篇論文的 model 架構參考了 MnasNet 的做法，同時考慮了準確度與硬體資源的限制，制定了以下公式作為 optimization goal：
$$ACC(m)\times [FLOPS(m)/T]^w$$其中 $m$ 指的是某個 model，而 $T$ 是目標 FLOPS，$w = -0.07$ 是用來控制正確率與 FLOPS 之間 trade-off 的超參數。最後制定出 baseline model EfficientNet-B0 的架構如下：

![擷取](https://i.imgur.com/25rDaOG.jpg)

接著就開始用上面幾個小節提到的方法基於 EfficientNet-B0 做 scaling。步驟如下：
- **步驟一**：固定 $\phi = 1$，根據上面幾個小節所定義的公式做 small grid search，找出 $\alpha, \beta, \gamma$。在實驗中，EfficientNet-B0 的 $\alpha, \beta, \gamma$ 分別為 1.2、1.1、1.15。
- **步驟二**：固定 $\alpha, \beta, \gamma$ 並根據不同的 $\phi$ 來擴增網路架構 (一樣基於上面幾個小節的公式限制)，最後擴增出了 EfficientNet-B1 到 B7 的網路。

當然，上述的方法也可以更改成在更大的網路架構下重新搜尋新的 $\alpha, \beta, \gamma$ 這樣的效果應該會更好，但缺點是運算成本會變得賊大，因此此論文只在 baseline model 做 search。

## Resutls
在模型的訓練上，採取和 MnasNet 相似的設置：使用 RMSProp optimizer (decay = 0.9, momentum = 0.9)、batch normalization (momentum = 0.99)、weight decay (1e-5)、初始 learning rate 0.256 (每 2.4 epochs 下降 0.97 倍)、SiLU activation、AutoAugment、stochastic depth (後面幾個技巧我也沒聽過，請自行 google)。


下圖是 EfficientNet 在 ImageNet 上的表現，並且比較了其他 ConvNet (如 ResNet、DenseNet、Inception) 的模型，可以發現在模型準確率近乎相同的時候，EfficientNet 的參數量都遠小於其他模型。

![擷取](https://i.imgur.com/B36jyFY.jpg)