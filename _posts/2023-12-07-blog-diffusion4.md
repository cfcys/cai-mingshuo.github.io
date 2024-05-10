---
title: 'Diffusion入门知识4---彻底搞懂扩散模型'
date: 2024-04-16
permalink: /posts/2024/01/blog----彻底搞懂扩散模型/
star: superior
tags:
  - 扩散模型
  - VAE
---

在简单了解了diffusion基本原理之后，我们要对稍微深入一点了

# 前向扩散&反向生成
先简单地回顾下上一节的内容，在diffusion中，我们更为熟悉的Encode和Decode过程被成为了前向扩散过程(向观测数据中逐步加入噪声，直到观测的数据变成高斯分布)和反向生成过程(从一个高斯分布中采样，逐步消除噪声，变成清晰的数据)

# 


## 参考资料

[扩散模型 - Diffusion Model【李宏毅2023】](https://www.bilibili.com/video/BV14c411J7f2/?spm_id_from=333.337.search-card.all.click&vd_source=32f9de072b771f1cd307ca15ecf84087)
[个人很喜欢的较真系列](https://www.bilibili.com/video/BV19H4y1G73r/?spm_id_from=333.337.search-card.all.click&vd_source=32f9de072b771f1cd307ca15ecf84087)
