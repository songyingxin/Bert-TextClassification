# imdb results

## 0. 长度分析

|                      | 最小长度           | 平均长度              | 最大长度               |
| -------------------- | ------------------ | --------------------- | ---------------------- |
| imdb                 | train:11    dev: 5 | train: 272   dev: 266 | train: 2789   dev:2640 |
| ag_news              | train: 3  dev:6    | train:36  dev: 35     | train:212   dev: 153   |
| sst_2                | train:1 dev: 2     | train:9  dev: 20      | train: 53   dev: 49    |
| dbpedia              | train: 2 dev: 3    | train:53 dev:53       | train:1499  dev:547    |
| yelp_review_full     | train: 1  dev: 1   | train: 156 dev: 157   | train: 1290 dev: 1196  |
| yelp_review_polarity | train: 1  dev: 1   | train: 155  dev: 154  | train:1202  dev: 1290  |
| THUCNews             | train: 1 dev: 1    | train: 10  dev: 26    | train: 1736 dev: 2268  |
| yahoo_answers        | train:2  dev: 1    | train: 36 dev: 36     | train: 885  dev: 2191  |
|                      |                    |                       |                        |

### 1. imdb

|       | 50   | 100  | 150  | 200  | 250  | 300  | 350  | 400  | 450  | 500  | 500+ |
| ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| train |      |      |      |      |      |      |      |      |      |      | 1    |
| dev   |      |      |      |      |      |      |      |      |      |      | 1    |

### 2. ag_news





### 1. 长度实验

| 模型                 | 50     | 100    | 150    | 200    | 250    | 300    | 350    | 400    | 450    | 500    |
| -------------------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| imdb                 | 81.572 | 86.792 | 89.828 | 91.176 | 92.148 | 92.808 | 93.280 | 93.672 | 93.772 | 93.996 |
| ag_news              |        |        | 94.224 | 94.14  | 94.335 | 94.276 | 94.224 | 94.197 | 94.263 | 94.263 |
| sst_2                | 92.546 |        |        |        | 93.005 | 93.463 | 93.005 | 93.119 | 93.119 | 93.005 |
| dbpedia              |        |        |        |        |        |        |        |        |        | 99.25  |
| yelp_review_full     |        |        |        |        |        |        |        |        |        |        |
| yelp_review_polarity |        |        |        |        |        |        |        |        |        |        |
| THUCNews             | 96.1   |        |        |        |        |        |        |        |        | 97.35  |
| yahoo_answers        |        |        |        |        |        |        |        |        |        |        |



```
python3 run_ag_news.py --max_seq_length=500 --num_train_epochs=5.0 --do_train --gpu_ids="4 5 6 7" --gradient_accumulation_steps=8 --print_step=100
```



## 模型试验

