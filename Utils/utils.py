# coding=utf-8

import time
from sklearn import metrics

import torch


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

    labels_list = [i for i in range(len(label_list))]

    report = metrics.classification_report(labels, preds, labels=labels_list, target_names=label_list, digits=5, output_dict=True)

    return acc, report



