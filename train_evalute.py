# coding=utf-8

import numpy as np
from tqdm import tqdm
import torch
from tensorboardX import SummaryWriter
import time

from Utils.utils import classifiction_metric


def train(epoch_num, n_gpu, model, train_dataloader, dev_dataloader, 
optimizer, criterion, gradient_accumulation_steps, device, label_list, 
output_model_file, output_config_file, log_dir, print_step, early_stop):
    """ 模型训练过程
    Args: 
        epoch_num: epoch 数量
        n_gpu: 使用的 gpu 数量
        train_dataloader: 训练数据的Dataloader
        dev_dataloader: 测试数据的 Dataloader
        optimizer: 优化器
        criterion： 损失函数定义
        gradient_accumulation_steps: 梯度积累
        device: 设备，cuda， cpu
        label_list: 分类的标签数组
        output_model_file: 用于保存 Bert 模型
        output_config_file: 用于 Bert 配置文件
        log_dir: tensorboard 读取的日志目录，用于后续分析
        print_step: 多少步保存一次模型，日志等信息
        early_stop: 提前终止
    """

    

    early_stop_times = 0

    writer = SummaryWriter(
        log_dir=log_dir + '/' + time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime(time.time())))


    best_dev_loss = float('inf')
    best_auc = 0
    best_acc = 0

    global_step = 0
    for epoch in range(int(epoch_num)):

        if early_stop_times >= early_stop:
            break

        print(f'---------------- Epoch: {epoch+1:02} ----------')

        epoch_loss = 0

        train_steps = 0

        all_preds = np.array([], dtype=int)
        all_labels = np.array([], dtype=int)

        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            model.train()
            
            batch = tuple(t.to(device) for t in batch)
            _, input_ids, input_mask, segment_ids, label_ids = batch

            logits = model(input_ids, segment_ids, input_mask, labels=None)
            loss = criterion(logits.view(-1, len(label_list)), label_ids.view(-1))

            """ 修正 loss """
            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            
            train_steps += 1

            loss.backward()

            # 用于画图和分析的数据
            epoch_loss += loss.item()
            preds = logits.detach().cpu().numpy()
            outputs = np.argmax(preds, axis=1)
            all_preds = np.append(all_preds, outputs)
            label_ids = label_ids.to('cpu').numpy()
            all_labels = np.append(all_labels, label_ids)

            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
            
                if global_step % print_step == 0 and global_step != 0:

                    """ 打印Train此时的信息 """
                    train_loss = epoch_loss / train_steps
                    train_acc, train_report, train_auc = classifiction_metric(all_preds, all_labels, label_list)

                    dev_loss, dev_acc, dev_report, dev_auc = evaluate(model, dev_dataloader, criterion, device, label_list)

                    c = global_step // print_step
                    writer.add_scalar("loss/train", train_loss, c)
                    writer.add_scalar("loss/dev", dev_loss, c)

                    writer.add_scalar("acc/train", train_acc, c)
                    writer.add_scalar("acc/dev", dev_acc, c)

                    writer.add_scalar("auc/train", train_auc, c)
                    writer.add_scalar("auc/dev", dev_auc, c)

                    for label in label_list:
                        writer.add_scalar(label + ":" + "f1/train", train_report[label]['f1-score'], c)
                        writer.add_scalar(label + ":" + "f1/dev",
                                        dev_report[label]['f1-score'], c)

                    print_list = ['macro avg', 'weighted avg']
                    for label in print_list:
                        writer.add_scalar(label + ":" + "f1/train",
                                        train_report[label]['f1-score'], c)
                        writer.add_scalar(label + ":" + "f1/dev",
                                        dev_report[label]['f1-score'], c)
                    
                    # # 以损失取优
                    # if dev_loss < best_dev_loss:
                    #     best_dev_loss = dev_loss
                    
                    # 以 acc 取优
                    if dev_acc > best_acc:
                        best_acc = dev_acc
                        
                    # 以 auc 取优
                    # if dev_auc > best_auc:
                    #     best_auc = dev_auc


                        model_to_save = model.module if hasattr(
                            model, 'module') else model
                        torch.save(model_to_save.state_dict(), output_model_file)
                        with open(output_config_file, 'w') as f:
                            f.write(model_to_save.config.to_json_string())

                        early_stop_times = 0
                    else:
                        early_stop_times += 1

    writer.close()
                    

def evaluate(model, dataloader, criterion, device, label_list):

    model.eval()

    all_preds = np.array([], dtype=int)
    all_labels = np.array([], dtype=int)

    epoch_loss = 0

    for _, input_ids, input_mask, segment_ids, label_ids in tqdm(dataloader, desc="Eval"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, labels=None)
        loss = criterion(logits.view(-1, len(label_list)), label_ids.view(-1))

        preds = logits.detach().cpu().numpy()
        outputs = np.argmax(preds, axis=1)
        all_preds = np.append(all_preds, outputs)

        label_ids = label_ids.to('cpu').numpy()
        all_labels = np.append(all_labels, label_ids)

        epoch_loss += loss.mean().item()

    acc, report, auc = classifiction_metric(all_preds, all_labels, label_list)
    return epoch_loss/len(dataloader), acc, report, auc


def evaluate_save(model, dataloader, criterion, device, label_list):

    model.eval()

    all_preds = np.array([], dtype=int)
    all_labels = np.array([], dtype=int)
    all_idxs = np.array([], dtype=int)

    epoch_loss = 0

    for idxs, input_ids, input_mask, segment_ids, label_ids in tqdm(dataloader, desc="Eval"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, labels=None)
        loss = criterion(logits.view(-1, len(label_list)), label_ids.view(-1))

        preds = logits.detach().cpu().numpy()
        outputs = np.argmax(preds, axis=1)
        all_preds = np.append(all_preds, outputs)

        label_ids = label_ids.to('cpu').numpy()
        all_labels = np.append(all_labels, label_ids)

        idxs = idxs.detach().cpu().numpy()
        all_idxs = np.append(all_idxs, idxs)

        epoch_loss += loss.mean().item()

    acc, report, auc = classifiction_metric(all_preds, all_labels, label_list)
    return epoch_loss/len(dataloader), acc, report, auc, all_idxs, all_labels, all_preds
