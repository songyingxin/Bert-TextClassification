import os
import sys
import csv

from .Classifier_utils import InputExample, convert_examples_to_features, convert_features_to_tensors


def read_tsv(filename):
    with open(filename, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t")
            lines = []
            for line in reader:
                lines.append(line)
            return lines

def load_tsv_dataset(filename, set_type):
    """
    文件内数据格式: sentence  label
    """
    examples = []
    lines = read_tsv(filename)
    for (i, line) in enumerate(lines):
        if i == 0:
            continue
        guid = i
        text_a = line[0]
        label = line[1]
        examples.append(
            InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples


def load_data(data_dir, tokenizer, max_length, batch_size, data_type, label_list, format_type=0):
    if format_type == 0:
        load_func = load_tsv_dataset

    if data_type == "train":
        train_file = os.path.join(data_dir, 'train.tsv')
        examples = load_func(train_file, data_type)
    elif data_type == "dev":
        dev_file = os.path.join(data_dir, 'dev.tsv')
        examples = load_func(dev_file, data_type)
    elif data_type == "test":
        test_file = os.path.join(data_dir, 'test.tsv')
        examples = load_func(test_file, data_type)
    else:
        raise RuntimeError("should be train or dev or test")

    features = convert_examples_to_features(
        examples, label_list, max_length, tokenizer)

    dataloader = convert_features_to_tensors(features, batch_size, data_type)

    examples_len = len(examples)

    return dataloader, examples_len

