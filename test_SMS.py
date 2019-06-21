# coding=utf-8

import csv
import random
import numpy as np
import time
from tqdm import tqdm
import os

import torch
import torch.nn as nn


from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.optimization import BertAdam

from Utils.utils import get_device
from Utils.Classifier_utils import InputExample, convert_examples_to_features, convert_features_to_tensors


def read_sms(filename):

    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader((line.replace('\0', '')
                                for line in f), delimiter='\t')
        lines = []

        for line in reader:
            lines.append(line)

        return lines

def create_examples(lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
        if len(line) != 2:
            print(line)
            break
        if i == 0:
            continue
        guid = "%s-%s" % (set_type, i)
        text_a = line[0]
        label = line[1]
        examples.append(
            InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples


def get_test_examples(filename):
    return create_examples(read_sms(filename), 'test')

def load_testdata(filename, tokenizer, max_length, batch_size):

    label_list = ['0', '1']
    examples = get_test_examples(filename)
    features = convert_examples_to_features(examples, label_list, max_length, tokenizer)
    dataloader = convert_features_to_tensors(features, batch_size)
    examples_len = len(examples)

    return dataloader, examples_len


def eval_save(model, dataloader, criterion, device, label_list, filename):
    model.eval()

    all_preds = np.array([], dtype=int)
    all_labels = np.array([], dtype=int)

    epoch_loss = 0

    for input_ids, input_mask, segment_ids, label_ids in tqdm(dataloader, desc="Eval"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, labels=None)
        
        preds = logits.detach().cpu().numpy()
        outputs = np.argmax(preds, axis=1)
        all_preds = np.append(all_preds, outputs)

        label_ids = label_ids.to('cpu').numpy()
        all_labels = np.append(all_labels, label_ids)

    filename = filename + '.csv'

    with open(filename, 'w', encoding='utf-8') as f:
        out_writer = csv.writer(f, delimiter=',')
        out_writer.writerow(['id', 'prediction'])

        for i in range(len(all_preds)):
            out_writer.writerow([all_labels[i], float(all_preds[i])])



def main(config, model_times, myProcessor):

    if not os.path.exists(config.output_dir + model_times):
        os.makedirs(config.output_dir + model_times)

    if not os.path.exists(config.cache_dir + model_times):
        os.makedirs(config.cache_dir + model_times)

    # Bert 模型输出文件
    output_model_file = os.path.join(
        config.output_dir, model_times, WEIGHTS_NAME)
    output_config_file = os.path.join(
        config.output_dir, model_times, CONFIG_NAME)

    # 设备准备
    gpu_ids = [int(device_id) for device_id in config.gpu_ids.split()]
    device, n_gpu = get_device(gpu_ids[0])
    if n_gpu > 1:
        n_gpu = len(gpu_ids)

    config.train_batch_size = config.train_batch_size // config.gradient_accumulation_steps

    # 设定随机种子
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(config.seed)

    # 数据准备
    tokenizer = BertTokenizer.from_pretrained(
        config.bert_vocab_file, do_lower_case=config.do_lower_case)  # 分词器选择
    label_list = ['0', '1']
    num_labels = len(label_list)

    """ Test """

    # test 数据
    input_dir = "/search/hadoop02/suanfa/songyingxin/Processed/test_all/"
    output_dir = "test_output/"

    filenames = os.listdir(input_dir)

    for filename in filenames:
        print(filename)
        file_id = filename.split('.')[0]
        true_filename = input_dir + filename
        if len(open(true_filename, 'rU').readlines()) < 5:
            continue
            
        test_dataloader, _ = load_testdata(
            true_filename, tokenizer, config.max_seq_length, config.test_batch_size)

        # 加载模型
        bert_config = BertConfig(output_config_file)

        if config.model_name == "BertOrigin":
            from BertOrigin.BertOrigin import BertOrigin
            model = BertOrigin(bert_config, num_labels=num_labels)
        elif config.model_name == "BertCNN":
            from BertCNN.BertCNN import BertCNN
            filter_sizes = [int(val) for val in config.filter_sizes.split()]
            model = BertCNN(bert_config, num_labels=num_labels,
                            n_filters=config.filter_num, filter_sizes=filter_sizes)
        elif config.model_name == "BertATT":
            from BertATT.BertATT import BertATT
            model = BertATT(bert_config, num_labels=num_labels)
        elif config.model_name == "BertRCNN":
            from BertRCNN.BertRCNN import BertRCNN
            model = BertRCNN(bert_config, num_labels=num_labels)
        elif config.model_name == "BertCNNPlus":
            from BertCNNPlus.BertCNNPlus import BertCNNPlus
            filter_sizes = [int(val) for val in config.filter_sizes.split()]
            model = BertCNNPlus(bert_config, num_labels=num_labels,
                                n_filters=config.filter_num, filter_sizes=filter_sizes)

        model.load_state_dict(torch.load(output_model_file))
        model.to(device)

        # 损失函数准备
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.to(device)
        output_filename = output_dir + filename
        # test the model
        eval_save(model, test_dataloader, criterion,
                  device, label_list, file_id)




if __name__ == "__main__":

    """ 业务数据集， 不用理会 """

    model_name = "BertOrigin"
    data_dir = "/search/hadoop02/suanfa/songyingxin/Processed/sub"
    output_dir = ".sms_output/"
    cache_dir = ".sms_cache"
    log_dir = ".sms_log/"

    model_times = "model_1/"   # 第几次保存的模型，主要是用来获取最佳结果

    bert_vocab_file = "/search/hadoop02/suanfa/songyingxin/pytorch_Bert/bert-base-chinese-vocab.txt"
    bert_model_dir = "/search/hadoop02/suanfa/songyingxin/pytorch_Bert/bert-base-chinese"

    from Processors.SMSProcessor import SMSProcessor

    if model_name == "BertOrigin":
        from BertOrigin import args

    elif model_name == "BertCNN":
        from BertCNN import args

    elif model_name == "BertATT":
        from BertATT import args

    elif model_name == "BertRCNN":
        from BertRCNN import args

    main(args.get_args(data_dir, output_dir, cache_dir,
                       bert_vocab_file, bert_model_dir, log_dir),
         model_times, SMSProcessor)
