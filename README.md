## Introduction

本仓库专注于 Bert  在文本分类领域的应用， 探索在 Bert 之上如何提高文本分类上的表现。

## Requirements

- Pytorch
- scikit-learn
- numpy
- pytorch_pretrained_bert
- tensorboardX

## 数据集

仓库采用了三个数据集，分别是 SST-2 情感分类， Yelp多标签分类， THUCNews 多标签分类。 

其中，  [THUCNews](http://thuctc.thunlp.org/)  只选取了一个子集， 该子集中包括了10个分类，每个分类6500条数据。

数据集我会在后续上传， 不建议自己去下载并做处理，这是因为我自己对数据进行了分析与简单预处理， 我觉得花时间在数据预处理上是不划算的，建议你专注于模型部分。

## 关于 Bert 

这里，使用了 [pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT) 来加载 Bert 模型， 考虑到国内网速问题，推荐先将相关的 Bert 文件下载，主要有两种文件：
> - vocab.txt: 记录了Bert中所用词表
> - 模型参数： 主要包括预训练模型的相关参数

相关文件下载连接在 [Bert](./Bert.md)

## 实验设置

- 本实验均在单1080ti上运行，但并没有删除在单机多卡上的逻辑，只是删除了分布式运算的逻辑，主要是考虑到大多数实验大家都没有必要去用到分布式。
- 删除了采用 fp16 的逻辑， 考虑到文本分类所需的资源并没有那么大， 采用 默认的32位浮点类型在大多数情况下是可以的， 没必要损失精度。
- **注意**： Bert 的参数量随着文本长度的增加呈现接近线性变化的趋势， 而 THUCNews 数据集的文本长度大多在1000-4000之间，这对于大多数机器是不可承受的， 测试在单1080ti上， 文本长度设置为150左右已经是极限。
- **注意：** 我有用 tensorboard 将相关的日志信息保存，推荐采用 tensorboard 进行分析。


## Results

### THUCNews

**注意：**  THUCNews 数据集中的样本长度十分的长，上面说到 Bert 本身对于序列长度十分敏感，因此我在我单1080ti下所能支持的最大长度。这也导致运行时间的线性增加，1个epoch 大概需要1个半小时到2个小时之间

```
python run_CNews.py --max_seq_length=500 --num_train_epochs=5.0 --do_train --gradient_accumulation_steps=8
```

model_name | ACC | F1 | Loss 
--- |--- | --- | --- 
BertOrigin | 97.740% | 97.73% | 0.114

### SST-2

```
python run_SST2.py --max_seq_length=65 --num_train_epochs=5.0 --do_train  # train and test
python run_SST2.py --max_seq_length=65   # test
```

| model_name | Acc     | F1      | Loss  |
| ---------- | ------- | ------- | ----- |
| BertOrigin | 94.656% | 95.217% | 0.306 |
|            |         |         |       |


### Yelp



## 如何适配自己的数据集

如果你想要适配自己的数据集，你只需要添加一个Processor即可， 然后在`run_**.py` 中更改`__main__` 下的几个相关文件夹以及`Processor`即可，简直不要太方便。

注意： 推荐先将数据转化为 tsv 格式， 内部参考上传的数据即可， 这是因为仓库本身实现了对 tsv 文件的支持， 提前转化数据有助于对数据进行简单预处理以及可以更加轻松的实现 Processor 。