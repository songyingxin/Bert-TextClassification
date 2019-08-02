# coding=utf-8

import torch
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)


class InputExample(object):
    """单句子分类的 Example 类"""

    def __init__(self, guid, text_a, text_b=None, label=None):
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
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, idx, input_ids, input_mask, segment_ids, label_id):
        self.idx = idx
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


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

        tokens_a = tokenizer.tokenize(example.text_a)  # 分词

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)  # 分词
            # “-3” 是因为句子中有[CLS], [SEP], [SEP] 三个标识，可参见论文
            # [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # "- 2" 是因为句子中有[CLS], [SEP] 两个标识，可参见论文
            # [CLS] the dog is hairy . [SEP]
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # [CLS] 可以视作是保存句子全局向量信息
        # [SEP] 用于区分句子，使得模型能够更好的把握句子信息

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)  # 句子标识，0表示是第一个句子，1表示是第二个句子，参见论文

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)  # 将词转化为对应词表中的id

        # input_mask: 1 表示真正的 tokens， 0 表示是 padding tokens
        input_mask = [1] * len(input_ids)
        padding = [0] * (max_seq_length - len(input_ids))

        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        try:
            label_id = label_map[example.label]
        except:
            print(example.label)
            continue
        idx = int(example.guid)

        features.append(
            InputFeatures(idx=idx, 
                          input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """ 截断句子a和句子b，使得二者之和不超过 max_length """

    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_features_to_tensors(features, batch_size, data_type):
    """ 将 features 转化为 tensor，并塞入迭代器
    Args:
        features: InputFeatures, 样本 features 信息
        batch_size: batch 大小
    Returns:
        dataloader: 以 batch_size 为基础的迭代器
    """
    all_idx_ids = torch.tensor(
        [f.idx for f in features], dtype=torch.long)
    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor(
        [f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor(
        [f.label_id for f in features], dtype=torch.long)

    data = TensorDataset(all_idx_ids, all_input_ids, all_input_mask,
                         all_segment_ids, all_label_ids)

    sampler = RandomSampler(data)
    if data_type == "test":
        dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    else:
        dataloader = DataLoader(data, sampler=sampler,
                                batch_size=batch_size, drop_last=True)

    return dataloader
