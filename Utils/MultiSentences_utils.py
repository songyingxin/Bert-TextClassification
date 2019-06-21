# coding=utf-8

import torch
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)


class InputExample(object):
    """单句子分类的 Example 类"""

    def __init__(self, guid, texts, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.texts = texts
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, example_id, texts_features, label):
        self.example_id = example_id
        self.texts_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids
            }
            for _,input_ids, input_mask, segment_ids in texts_features
        ]
        self.label = label


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
    """Loads a data file into a list of `InputBatch`s.
    Args:
        examples: InputExample, 表示样本集
        label_list: 标签列表
        max_seq_length: 句子最大长度
        tokenizer： 分词器
    Returns:
        features: InputFeatures, 表示样本转化后信息 
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            print("Writing example {} of {}".format(ex_index, len(examples)))

        texts_features = []

        true_sencentes = 0

        for text_index, text in enumerate(example.texts):
            text_tokens = tokenizer.tokenize(text)
        
            _truncate_seq_pair(text_tokens, max_seq_length-2)

            tokens = ["[CLS]"] + text_tokens  + ["[SEP]"]

            segment_ids = [0] * len(tokens)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            # Padding
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            texts_features.append((tokens, input_ids, input_mask, segment_ids))
        
        label = label_map[example.label]

        features.append(
            InputFeatures(
                example_id=example.guid,
                texts_features=texts_features,
                label=label
            )
        ) 

    return features


def _truncate_seq_pair(tokens, max_length):
    """ 截断句子a和句子b，使得二者之和不超过 max_length """

    while True:
        total_length = len(tokens)
        if total_length <= max_length:
            break
        else:
            tokens.pop()


def select_field(features, field):
    return [
        [
            text[field]
            for text in feature.texts_features
        ]
        for feature in features
    ]

def convert_features_to_tensors(features, batch_size):
    """ 将 features 转化为 tensor，并塞入迭代器
    Args:
        features: InputFeatures, 样本 features 信息
        batch_size: batch 大小
    Returns:
        dataloader: 以 batch_size 为基础的迭代器
    """
    input_ids = select_field(features, 'input_ids')
    all_input_ids = torch.tensor(input_ids, dtype=torch.long)
    all_input_mask = torch.tensor(
        select_field(features, 'input_mask'), dtype=torch.long)
    all_segment_ids = torch.tensor(
        select_field(features, 'segment_ids'), dtype=torch.long)
    all_label_ids = torch.tensor(
        [f.label for f in features], dtype=torch.long)

    data = TensorDataset(all_input_ids, all_input_mask,
                         all_segment_ids, all_label_ids)

    sampler = RandomSampler(data)
    # dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    dataloader = DataLoader(data, sampler=sampler,
                            batch_size=batch_size, drop_last=True)
    return dataloader


def load_data(data_dir, tokenizer, processor, max_length, batch_size, data_type, max_sentence_num):
    """ 导入数据， 并返回对应的迭代器
    Args: 
        data_dir: 原始数据目录
        tokenizer： 分词器
        processor: 定义的 processor
        max_length: 句子最大长度
        batch_size: batch 大小
        data_type: "train" or "dev", "test" ， 表示加载哪个数据
    
    Returns:
        dataloader: 数据迭代器
        examples_len: 样本大小
    """

    label_list = processor.get_labels()

    if data_type == "train":
        examples = processor.get_train_examples(data_dir, max_sentence_num)
    elif data_type == "dev":
        examples = processor.get_dev_examples(data_dir, max_sentence_num)
    elif data_type == "test":
        examples = processor.get_test_examples(data_dir, max_sentence_num)
    else:
        raise RuntimeError("should be train or dev or test")

    features = convert_examples_to_features(examples, label_list, max_length, tokenizer)

    dataloader = convert_features_to_tensors(features, batch_size)

    examples_len = len(examples)

    return dataloader, examples_len
