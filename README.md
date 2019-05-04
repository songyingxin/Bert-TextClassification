## 数据集

仓库采用了三个数据集，分别是 SStT-2 情感分类， Yelp多标签分类， THUCNews 多标签分类。 

其中，  [THUCNews](http://thuctc.thunlp.org/)  只选取了一个子集， 该子集中包括了10个分类，每个分类6500条数据：
> - train： 5000 * 10
> - dev: 500 * 10
> - test： 1000 * 10

## 关于 Bert 

这里，使用了 [pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT) 来加载 Bert 模型， 考虑到国内网速问题，推荐先将相关的 Bert 文件下载，主要有两种文件：
> - vocab.txt: 记录了Bert中所用词表
> - 模型参数： 主要包括预训练模型的相关参数

```
# vocab 文件下载
'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt",
'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt",
'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-vocab.txt",
'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-vocab.txt",
'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-vocab.txt",
'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-vocab.txt",
'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt",

# 预训练模型参数下载
'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz",
'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz",
'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz",
'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz",
'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz",
'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz",
'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz",
```

## 实验设置

本实验均在单1080ti上运行，后续如果有计算资源会加入多GPU分布式训练的逻辑。


## Results

### THUCNews

model_name | ACC | F1 | 
--- |--- | ---
BertOrigin | | 


### SST-2


### Yelp

