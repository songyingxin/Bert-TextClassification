import random
import numpy as np 
import time
from tqdm import tqdm
import os

import torch
import torch.nn as nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset) 


from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

from Utils.utils import get_device, load_data
from BertOrigin import args

from Utils.train_evalute import train, evaluate
from Processors.SST2Processor import Sst2Processor, load_data


def main(config):

    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    
    if not os.path.exists(config.cache_dir):
        os.makedirs(config.cache_dir)
        
    output_model_file = os.path.join(config.output_dir, WEIGHTS_NAME) # 模型输出文件
    output_config_file = os.path.join(config.output_dir, CONFIG_NAME)

    device, n_gpu = get_device()  # 设备准备

    config.train_batch_size = config.train_batch_size // config.gradient_accumulation_steps

    """ 设定随机种子 """
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(config.seed)

    """ 数据准备 """
    processor = Sst2Processor()  # 整个文件的代码只需要改此处即可
    tokenizer = BertTokenizer.from_pretrained(
        config.bert_vocab_file, do_lower_case=config.do_lower_case)  # 分词器选择

    label_list = processor.get_labels()
    num_labels = len(label_list)

    if config.do_train:

        train_dataloader, train_examples_len = load_data(config.data_dir, tokenizer, processor, config.max_seq_length, config.train_batch_size, "train")
        dev_dataloader, _ = load_data(
            config.data_dir, tokenizer, processor, config.max_seq_length, config.dev_batch_size, "dev")
        
        num_train_optimization_steps = int(
            train_examples_len / config.train_batch_size / config.gradient_accumulation_steps) * config.num_train_epochs

        """ 模型准备 """
        model = BertForSequenceClassification.from_pretrained(
            config.bert_model_dir, cache_dir=config.cache_dir, num_labels=num_labels)
        
        model.to(device)

        """ 优化器准备 """
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]


        optimizer = BertAdam(optimizer_grouped_parameters,
                            lr=config.learning_rate,
                            warmup=config.warmup_proportion,
                            t_total=num_train_optimization_steps)

        """ 损失函数准备 """ 
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.to(device)

        train(config.num_train_epochs, n_gpu, model, train_dataloader, dev_dataloader, optimizer,
              criterion, config.gradient_accumulation_steps, device, label_list, output_model_file, output_config_file)

    """ Test """
    test_dataloader, _ = load_data(
        config.data_dir, tokenizer, processor, config.max_seq_length, config.test_batch_size, "test")

    bert_config = BertConfig(output_config_file)
    model = BertForSequenceClassification(bert_config, num_labels=num_labels)
    model.load_state_dict(torch.load(output_model_file))
    model.to(device)

    """ 损失函数准备 """
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    # test the model
    test_loss, test_acc, test_f1 = evaluate(
        model, test_dataloader, criterion, device, label_list)
    print("-------------- Test -------------")
    print(f'\t \t Loss: {test_loss: .3f} | Acc: {test_acc*100: .3f} % | F1: {test_f1 * 100: .3f} %')


if __name__ == "__main__":

    data_dir = "/home/songyingxin/datasets/SST-2"
    output_dir = ".output"
    cache_dir = ".cache"

    bert_vocab_file = "/home/songyingxin/datasets/pytorch-bert/vocabs/bert-base-uncased-vocab.txt"
    bert_model_dir = "/home/songyingxin/datasets/pytorch-bert/bert-base-uncased"
    main(args.get_args(data_dir, output_dir, cache_dir, bert_vocab_file, bert_model_dir))
