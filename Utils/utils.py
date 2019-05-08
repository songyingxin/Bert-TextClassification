import spacy
import time
import matplotlib.pyplot as plt
import csv
from sklearn import metrics

import torch

from torchtext import data
from torchtext import datasets
from torchtext import vocab

NLP = spacy.blank("en")


def word_tokenize(sent):
    """ 分词 """
    doc = NLP(sent)
    return [token.text for token in doc]


def get_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    if torch.cuda.is_available():
        print("device is cuda, # cuda is: ", n_gpu)
    else:
        print("device is cpu, not recommend")
    return device, n_gpu


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def classifiction_metric(preds, labels, label_list):
    """ 分类任务的评价指标， 传入的数据需要是 numpy 类型的 """

    acc = metrics.accuracy_score(labels, preds)

    f1 = metrics.f1_score(labels, preds)

    return acc, f1


def load_data(data_dir, tokenizer, processor, max_length, batch_size, data_type):

    label_list = processor.get_labels()

    if data_type == "train":
        examples = processor.get_train_examples(data_dir)
    elif data_type == "dev":
        examples = processor.get_dev_examples(data_dir)
    elif data_type == "test":
        examples = processor.get_test_examples(data_dir)
    else:
        raise RuntimeError("should be train or dev or test")

    features = convert_examples_to_features(
        examples, label_list, max_length, tokenizer)
    dataloader = convert_features_to_tensors(features, batch_size)

    return dataloader, len(examples)
