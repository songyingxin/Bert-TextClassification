## Introduction

本仓库专注于 Bert  在文本分类领域的应用， 探索在 Bert 之上如何提高文本分类上的表现。

## Requirements

下面命令还未经过完整测试， 可以参考。

推荐使用 Anconda 来管理包环境， 我采用的是 Anconda python 3.7，其余 3.0 以上应该都可以， 推荐新建一个环境来做测试。

```
conda create -n BertText  # 创建新环境
conda activate BertText   # 激活指定环境

Pytorch ： [conda install pytorch torchvision cudatoolkit=9.0 -c pytorch](https://pytorch.org/get-started/locally/)

scikit-learn： conda install scikit-learn

pytorch-pretrained-BERT： pip install pytorch-pretrained-bert

numpy： conda install numpy

tensorboardx： pip install tensorboardX

tensorflow： pip install tensorflow
```

## 数据集

仓库采用了三个数据集，分别是 SST-2 情感分类， Yelp多标签分类， THUCNews 多标签分类。 

其中，  [THUCNews](http://thuctc.thunlp.org/)  只选取了一个子集， 该子集中包括了10个分类，每个分类6500条数据。

```
sst-2: 链接：https://pan.baidu.com/s/1ax9uCjdpOHDxhUhpdB0d_g  提取码：rxbi 
cnews: 链接：https://pan.baidu.com/s/19sOrAxSKn3jCIvbVoD_-ag  提取码：rstb 
```

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
python3 run_CNews.py --max_seq_length=512 --num_train_epochs=5.0 --do_train --gpu_ids="4 5 6 7" --gradient_accumulation_steps=8 --print_step=500  # gpu_ids 选择 gpu， 如果是单gpu， 选择 max_seq_length 为150较为合适(1080ti)
python3 run_CNews.py --max_seq_length=512
```

model_name | loss | acc | f1 
--- |--- | --- | --- 
BertOrigin(base) | 0.088 | 97.40 | 97.39 
 |       |  |  
 |       |  |  
 |       |  |  
 BertHAN | 0.103 | 97.49 | 97.48 
 |       |  |  

### SST-2

```
python3 run_SST2.py --max_seq_length=65 --num_train_epochs=5.0 --do_train --gpu_ids="1" --gradient_accumulation_steps=8 --print_step=100  # train and test
python3 run_SST2.py --max_seq_length=65   # test
```

| 模型                 | loss  | acc    | f1     |
| -------------------- | ----- | ------ | ------ |
| BertOrigin(base)     | 0.170 | 94.458 | 94.458 |
| BertCNN (5,6) (base) | 0.148 | 94.607 | 94.62  |
| BertATT (base)       | 0.162 | 94.211 | 94.22  |
| BertRCNN (base)      | 0.145 | 95.151 | 95.15  |
| BertCNNPlus (base)   | 0.160 | 94.508 | 94.51  |



### Yelp



## 如何适配自己的数据集

如果你想要适配自己的数据集，你只需要添加一个Processor即可， 然后在`run_**.py` 中更改`__main__` 下的几个相关文件夹以及`Processor`即可，简直不要太方便。

注意： 推荐先将数据转化为 tsv 格式， 内部参考上传的数据即可， 这是因为仓库本身实现了对 tsv 文件的支持， 提前转化数据有助于对数据进行简单预处理以及可以更加轻松的实现 Processor 。

## 如何定义新模型

