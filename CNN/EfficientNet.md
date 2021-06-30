<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    tex2jax: {
      inlineMath: [ ['$','$'], ["\\(","\\)"] ],
      processEscapes: true
    }
  });
</script>
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>


# EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks

## Introduction
> 核心思想：有沒有一個規則或方法能夠擴增 ConvNet 並且維持一定的運算效率且又能提升準確度？

這篇論文是 2019 年由 Google 團隊所提出，一般在訓練 CNN 的網路時通常都是在以下三個因素上做調整來讓模型學習的更好：模型的深度(depth)、模型的寬度(width)、以及輸入圖片的解析度(resolution)。然而對於這三個因素的調整往往都是憑經驗或任意的調整，沒有一個系統性的調整方式。因此 Google 團隊提出了系統性擴增 (scale up) 整個 model 的方法 - compound scaling method，在實驗中基於 MobileNets 跟 ResNet 作為 baseline 來做擴增並且也透過 NAS (Neural architecture search) 的方式開發出 EfficientNet B0 作為新的 baseline 然後進一步擴增成一系列 B1~B7 的 EfficientNet。

從下圖的結果來看，的確 EfficientNet 系列不但參數比其他 ConvNet 還要少，並且準確度也有顯著的提升。論文也聲稱除了在 ImageNet 上有好的表現外，在其他幾個 datasets 一樣有不錯的成績。

![](https://i.imgur.com/CMnDneH.jpg)


## Compound Model Scaling
每一層的 ConvNet 可以被定義成 $Y_i = F_i(X_i)$，其中 $X_i$ 是維度為 $<H_i, W_i, C_i>$ 的 input。

<div style="text-align:center">
<img src="https://i.imgur.com/H7yd8fL.jpg" width="400"/>
</div>


$$
N = F_k\odot ...\odot F_2 \odot F_1(X_1) = \bigodot_{j=1...k}F_j(X_1)
$$