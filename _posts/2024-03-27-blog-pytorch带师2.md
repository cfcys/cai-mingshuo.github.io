---
title: 'Pytorch进阶之杂七杂八的知识点总结'
date: 2024-04-07
permalink: /posts/2024/03/blog-Pytorch/
star: superior
tags:
  - Pytorch
---

如果不知道干什么，就写写博客总结总结

# `torch.nn.Parameter & torch.Tensor`

> 学这个主要是因为我在深度学习的上机中对于“冻结权重”这个相对常见的操作感到一脸懵逼，这还是在提醒我自己的torch并**不熟练**。

当时的一个操作是这样的，在前50个batch中将原始网络中backbone中的权重去冻结，在后面100个batch中将权重去解冻，这样的话即加快了收敛，也省了很多事情。

```python
# 请将classes_path指向数据集对应的类别
classes_path='model data/Helmet classes.txt'
#请在此冻结主干特征提取网络部分的参数
for param in model.backbone.parameters():
    param.requires grad =False
#当前epoch已经大于冻结学习的代数，请在此对主干特征提取网络解冻
for param in model.backbone.parameters():
    param.requires grad = True
```


##  `torch.nn.parameter`

`torch.nn.parameter`是torch中一种特殊类型的tensor，用于表示神经网络中的可学习参数，for example，全连接层中的`torch.nn.Linear()`中的参数weight和bias。
 
## 与`torch.tensor`的区别

### 是否会自动添加到模型的参数列表中

* 使用`torch.nn.parameter`定义的张量可以被自动的添加到模型的参数列表中，并且可以通过`.parameters()`的方法或者`.name_parameters()`的方法列出

```python
import torch
import torch.nn as nn

class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.,2.]))
    
    def forward(self,input):
        return input * self.weight
layer = Layer()
print(layer.weight.requires_grad)    
for name,param in layer.named_parameters():
    print(name)   # weight
    print(param)   
```
输出结果为：
```bash
True
weight
Parameter containing:
tensor([1., 2.], requires_grad=True)
```

* 而普通的`torch.tensor`对象是不会被自动的添加到模型的参数列表中的，因此也不会被`.parameters()`的方法或者`.name_parameters()`的方法给查询到。

```python
import torch
import torch.nn as nn

class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.tensor([1.,2.])
    
    def forward(self,input):
        return input * self.weight
layer = Layer()
print(layer.weight.requires_grad)    
for name,param in layer.named_parameters():
    print(name)   # weight
    print(param)   # 无输出，打印不出来任何的结果，说明该参数甚至未被放到参数的列表中去

```
output:
```bash
False
```

### required_grad属性

这一点相对容易理解，也即使用`torch.nn.Parameter`定义的对象其`required_grad`属性默认为True,而一个简单的`torch.Tensor`定义的张量其`required_grad`属性默认为False。

* `torch.nn.Parameter`定义的对象的属性被设置为`True`的时候，他们会参与自动的求导，并且可以被优化器自动地更新，这也解释了我们平时设置优化器时候为什么都是`optimizer = optim.SGD(mode.)`。

* 对于普通的`torch.Tensor`对象，**即使将其`required_grad`属性设置为True，也不会被自动的添加到模型的参数中去，也不会被优化器自动更新**。


## `nn.Linear`

`nn.Linear`是我们炼丹时候十分常见且简易一个操作，但是有了上面这些知识的先验，我们印象中的`nn.Linear`应该有所改变,例如，至少我们要明白，其中的`weight`和`bias`属性都是通过`torch.nn.Parameter`封装起来的。

简易版本的pytorch实现

```python


```

## 参考资料
[Enzo_Ai](https://www.bilibili.com/video/BV1tC411L7Hh/?spm_id_from=333.788.0.0&vd_source=32f9de072b771f1cd307ca15ecf84087)

























