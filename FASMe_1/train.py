#!/usr/bin/env python3
import argparse
from collections import OrderedDict
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3,2,0,1'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from backbones.fasmodel import FASModel
from detection_layers.modules import MultiBoxLoss
from dataset import FASDataset
from lib.util import load_config, update_learning_rate, my_collate
import time
import logging
import math
from sklearn.metrics import roc_auc_score, roc_curve, auc
import numpy as np
from loss import *
import warnings


warnings.filterwarnings('ignore')


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

def args_func():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='The path to the config.', default='./configs/fasme_train.cfg')
    parser.add_argument('--ckpt', type=str, help='The checkpoint of the pretrained model.', default='./checkpoints/epoch_31_best_0.0.pkl')
    parser.add_argument('--device', type=str, default='0', help='device id, format is like 0,1,2')

    args = parser.parse_args()
    return args


def save_checkpoint(net, opt, save_path, epoch_num, iters=9999999):
    os.makedirs(save_path, exist_ok=True)
    module = net.module
    model_state_dict = OrderedDict()
    for k, v in module.state_dict().items():
        model_state_dict[k] = torch.tensor(v, device="cpu")

    # opt_state_dict = {}
    # opt_state_dict['param_groups'] = opt.state_dict()['param_groups']
    # opt_state_dict['state'] = OrderedDict()
    # for k, v in opt.state_dict()['state'].items():
    #     opt_state_dict['state'][k] = {}
    #     opt_state_dict['state'][k]['step'] = v['step']
    #     if 'exp_avg' in v:
    #         opt_state_dict['state'][k]['exp_avg'] = torch.tensor(v['exp_avg'], device="cpu")
    #     if 'exp_avg_sq' in v:
    #         opt_state_dict['state'][k]['exp_avg_sq'] = torch.tensor(v['exp_avg_sq'], device="cpu")

    checkpoint = {
        'network': model_state_dict,
        # 'opt_state': opt_state_dict,
        'epoch': epoch_num,
    }

    torch.save(checkpoint, f'{save_path}/epoch_{epoch_num}_{iters}.pkl')


def save_best_checkpoint(net, opt, save_path, epoch_num, f_auc):
    os.makedirs(save_path, exist_ok=True)
    module = net.module
    model_state_dict = OrderedDict()
    for k, v in module.state_dict().items():
        model_state_dict[k] = torch.tensor(v, device="cpu")

    # opt_state_dict = {}
    # opt_state_dict['param_groups'] = opt.state_dict()['param_groups']
    # opt_state_dict['state'] = OrderedDict()
    # for k, v in opt.state_dict()['state'].items():
    #     opt_state_dict['state'][k] = {}
    #     opt_state_dict['state'][k]['step'] = v['step']
    #     if 'exp_avg' in v:
    #         opt_state_dict['state'][k]['exp_avg'] = torch.tensor(v['exp_avg'], device="cpu")
    #     if 'exp_avg_sq' in v:
    #         opt_state_dict['state'][k]['exp_avg_sq'] = torch.tensor(v['exp_avg_sq'], device="cpu")

    checkpoint = {
        'network': model_state_dict,
        # 'opt_state': opt_state_dict,
        'epoch': epoch_num,
    }

    torch.save(checkpoint, f'{save_path}/epoch_{epoch_num}_best_{f_auc}.pkl')


def load_checkpoint(ckpt, net, opt, device):
    checkpoint = torch.load(ckpt)

    # gpu_state_dict = OrderedDict()
    # for k, v in checkpoint['network'] .items():
    #     name = "module."+k  # add `module.` prefix
    #     name = k
    #     gpu_state_dict[name] = v.to(device)
    # net.load_state_dict(gpu_state_dict)

    model_state = net.state_dict()
    pretrained_state = checkpoint['network']
    pretrained_state = {k: v for k, v in pretrained_state.items() if
                        k in model_state and v.size() == model_state[k].size()}
    model_state.update(pretrained_state)
    net.load_state_dict(model_state)

    # opt.load_state_dict(checkpoint['opt_state'])
    base_epoch = int(checkpoint['epoch']) + 1

    return net, opt, base_epoch


def train():

    args = args_func()
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # load conifigs
    cfg = load_config(args.cfg)

    logging.basicConfig(filename=cfg['model']['save_path']+"/train.log", level=logging.INFO)

    # init model.
    net = FASModel(3, backbone=cfg['model']['backbone'])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    # loss init
    det_criterion = MultiBoxLoss(
        cfg['det_loss']['num_classes'],
        cfg['det_loss']['overlap_thresh'],
        cfg['det_loss']['prior_for_matching'],
        cfg['det_loss']['bkg_label'],
        cfg['det_loss']['neg_mining'],
        cfg['det_loss']['neg_pos'],
        cfg['det_loss']['neg_overlap'],
        cfg['det_loss']['encode_target'],
        cfg['det_loss']['use_gpu']
    )
    weights = torch.tensor([2.0, 1.0, 1.0])
    criterion = nn.CrossEntropyLoss(weight=weights).cuda(device)
    criterion1 = nn.CrossEntropyLoss()

    # optimizer init.
    optimizer = optim.AdamW(net.parameters(), lr=1e-3, weight_decay=4e-3)

    # load checkpoint if given
    base_epoch = 0
    if args.ckpt:
        net, optimzer, base_epoch = load_checkpoint(args.ckpt, net, optimizer, device)

    net = net.to(device)
    net = nn.DataParallel(net)

    # get training data
    print(f"Load Train deepfake dataset from {cfg['dataset']['img_path']}..")
    train_dataset = FASDataset('train', cfg, cfg['dataset']['info_path'])
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg['train']['batch_size'],
                              shuffle=True, num_workers=16,#zzzz
                              collate_fn=my_collate
                              )

    # get testing data
    print(f"Load Test deepfake dataset from {cfg['test_dataset']['img_path']}..")
    test_dataset = FASDataset('test', cfg, cfg['test_dataset']['info_path'])
    test_loader = DataLoader(test_dataset,
                              batch_size=cfg['train']['batch_size'],
                              shuffle=False, num_workers=16,#zzzz
                              collate_fn=my_collate
                              )

    contra_fun = ContrastLoss()

    # start training.
    warmup_steps = cfg['train']['warmup_epoch'] * len(train_dataset) // cfg['train']['batch_size']
    g_step = 0
    best_auc = 1000
    pre_time = time.time()
    for epoch in range(base_epoch, cfg['train']['epoch_num']):
        acc_record = AvgrageMeter()
        tloss_record = AvgrageMeter()

        net.train()
        for index, (batch_data, batch_labels) in enumerate(train_loader):
            net.train()

            lr = update_learning_rate(epoch, g_step, warmup_steps, cfg['train']['warmup_epoch'])
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            labels, location_labels, confidence_labels, domains = batch_labels
            labels = labels.long().to(device)
            location_labels = location_labels.to(device)
            confidence_labels = confidence_labels.long().to(device)
            domains = domains.long().to(device)

            rand_idx = torch.randperm(batch_data.shape[0])

            optimizer.zero_grad()
            locations, confidence, domain_invariant, outputs, feats1, feats2 = net(batch_data, batch_data[rand_idx, :, :, :])
            locations = locations.to(device)
            confidence = confidence.to(device)
            outputs = outputs.to(device)

            loss_end_cls = criterion(outputs, labels)
            acc = sum(outputs.max(-1).indices == labels).item() / labels.shape[0]

            contrast_label = labels[:].long() == labels[rand_idx].long()
            contrast_label = torch.where(contrast_label == True, 1, -1)
            constra_loss = contra_fun(feats1, feats2, contrast_label)

            loss_l_0, loss_c_0 = det_criterion((locations[:,0,:,:], confidence[:,0,:,:]), confidence_labels[:,0,:], location_labels[:,0,:])
            loss_l_4, loss_c_4 = det_criterion((locations[:,1,:,:], confidence[:,1,:,:]), confidence_labels[:,1,:], location_labels[:,1,:])
            det_loss = 0.1 * (loss_l_0 + loss_c_0 + loss_l_4 + loss_c_4)

            adv_loss = criterion1(domain_invariant, domains.long())

            loss = loss_end_cls + det_loss + constra_loss + adv_loss

            if (math.isinf(loss.item()) and loss.item() > 0) or (math.isinf(loss.item()) and loss.item() < 0):
                print("error: loss is inf : {:.8f}".format(loss.item()))
                logging.error("error: loss is inf : {:.8f}".format(loss.item()))
                continue

            n = labels.shape[0]
            acc_record.update(acc, n)
            tloss_record.update(loss.item(), n)

            loss.backward()

            torch.nn.utils.clip_grad_value_(net.parameters(), 2)
            optimizer.step()
            g_step += 1

            if index % 5 == 0:
                cur_time = time.time()
                outputs = [
                    "e:{},iter:{}".format(epoch, index),
                    "acc:{:.2f}".format(acc),
                    "total loss:{:.2f} ".format(loss.item()),
                    # "loss: {:.8f} ".format(loss_record.avg),
                    "cls loss:{:.4f} ".format(loss_end_cls.item()),
                    "det loss:{:.4f} ".format(det_loss.item()),
                    "cont loss:{:.4f} ".format(constra_loss.item()),
                    "adv loss:{:.4f} ".format(adv_loss.item()),
                    "lr:{:.5g}".format(lr),
                    "time:{:.1f}s".format(cur_time - pre_time),
                ]
                print(" ".join(outputs))
                logging.info(" ".join(outputs))
                pre_time = cur_time

            if index > 0 and index % 5000 == 0: # and epoch >= cfg['train']['warmup_epoch']:
                # save_checkpoint(net, optimizer,
                #                 cfg['model']['save_path'],
                #                 epoch, index)

                print("Starting testing")
                logging.info("Starting testing")
                BPCER = test(net, test_loader)
                print("Finished testing : BPCER = {:.8f} ".format(BPCER))
                logging.info("Finished testing : BPCER = {:.8f} ".format(BPCER))
                if best_auc >= BPCER:
                    best_auc = BPCER
                    save_best_checkpoint(net, optimizer,
                                    cfg['model']['save_path'],
                                    epoch, BPCER)
                    print("Saved best model : best_BPCER = {:.8f}".format(best_auc))
                    logging.info("Saved best model : best_BPCER = {:.8f}".format(best_auc))

        save_checkpoint(net, optimizer,
                        cfg['model']['save_path'],
                        epoch, index)



def get_err_threhold(fpr, tpr, threshold):
    differ_tpr_fpr_1=tpr+fpr-1.0
    right_index = np.argmin(np.abs(differ_tpr_fpr_1))
    best_th = threshold[right_index]
    err = fpr[right_index]
    return err, best_th, right_index


def performances_val(frame_label_list, frame_pred_list):
    val_scores = []
    val_labels = []
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for idx in range(len(frame_label_list)):
        try:
            count += 1
            score = float(frame_pred_list[idx])
            label = float(frame_label_list[idx])
            val_scores.append(score)
            val_labels.append(label)
            data.append({'map_score': score, 'label': label})
            if label == 0:
                num_real += 1
            else:
                num_fake += 1
        except:
            continue

    fpr, tpr, threshold = roc_curve(val_labels, val_scores, pos_label=1)
    auc_test = auc(fpr, tpr)
    val_err, val_threshold, right_index = get_err_threhold(fpr, tpr, threshold)

    type1 = len([s for s in data if s['map_score'] < val_threshold and s['label'] == 1])
    type2 = len([s for s in data if s['map_score'] > val_threshold and s['label'] == 0])

    val_ACC = 1 - (type1 + type2) / count

    FRR = 1 - tpr  # FRR = 1 - TPR

    HTER = (fpr + FRR) / 2.0  # error recognition rate &  reject recognition rate

    print("val_threshold={}".format(val_threshold))

    return val_ACC, fpr[right_index], FRR[right_index], HTER[right_index], auc_test, val_err


def test(net, test_loader):
    frame_pred_list = list()
    frame_label_list = list()
    video_name_list = list()

    net.eval()
    with torch.no_grad():
        for batch_data, batch_labels in test_loader:

            labels, video_name = batch_labels
            labels = labels.long()

            alive_label = labels[:].long() == 0
            alive_label = torch.where(alive_label == True, 1, 0)

            outputs = net(batch_data, batch_data)
            outputs = outputs[:, 0]
            frame_pred_list.extend(outputs.detach().cpu().numpy().tolist())
            frame_label_list.extend(alive_label.detach().cpu().numpy().tolist())
            video_name_list.extend(list(video_name))

            # print(f"{video_name[0]} label:{alive_label.detach().cpu().numpy()[0]} out:{outputs.detach().cpu().numpy()[0]}")

        f_auc = roc_auc_score(frame_label_list, frame_pred_list)
        fpr, tpr, threshold = roc_curve(frame_label_list, frame_pred_list)

        for idx in range(len(tpr)):
            if tpr[idx] >= 0.99:
                break
        if idx == len(tpr):
            BPCER = 1.0
            mythresh = 0.0
        else:
            mythresh = threshold[idx]

            type1 = 0
            type2 = 0
            num_real = 0
            for iii in range(len(frame_label_list)):
                if frame_label_list[iii] == 0:
                    num_real += 1
                if frame_pred_list[iii] < mythresh and frame_label_list[iii] == 1:
                    type1 += 1
                if frame_pred_list[iii] >= mythresh and frame_label_list[iii] == 0:
                    type2 += 1

            BPCER = type2 / num_real  # Bona Fide Presentation Classification Error Rate for 0.01 Attack Presentation Classification Error Rate (APCER)

        print("BPCER={}, val_threshold={}".format(BPCER, mythresh))
        logging.info("BPCER={}, val_threshold={}".format(BPCER, mythresh))

        # val_err, val_threshold, right_index = get_err_threhold(fpr, tpr, threshold)
        # print("val_threshold={}".format(val_threshold))

    return BPCER


if __name__ == "__main__":
    train()

# vim: ts=4 sw=4 sts=4 expandtab
