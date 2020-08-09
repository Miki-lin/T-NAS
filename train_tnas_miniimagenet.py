import numpy as np
import scipy.stats
import argparse
import os
import logging
import glob
import sys
import time
import random


from MiniImagenet_task import MiniImagenet
from meta_nas_train import Meta_decoding
from learner import Network
from utils.utils import infinite_get
import utils.utils as utils
from utils.saver import Saver
from utils.summaries import TensorboardSummary

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from ptflops import get_model_complexity_info

import pdb

parser = argparse.ArgumentParser("mini-imagenet")
parser.add_argument('--dataset', type=str, default='mini-imagenet', help='dataset')
parser.add_argument('--checkname', type=str, default='meta-nas-train', help='checkname')
parser.add_argument('--run', type=str, default='run_meta_nas', help='run_path')
parser.add_argument('--data_path', type=str, default='/data2/dongzelian/datasets/mini-imagenet/', help='path to data')
parser.add_argument('--pretrained_model', type=str, default='/data2/dongzelian/NAS/meta_nas/run_meta_nas/mini-imagenet/meta-nas/experiment_21/model_best.pth.tar', help='path to pretrained model')
parser.add_argument('--seed', type=int, default=222, help='random seed')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epoch', type=int, help='epoch number', default=10)
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--num_workers', type=int, default=8, help='number of workers')
parser.add_argument('--n_way', type=int, help='n way', default=5)
parser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
parser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
parser.add_argument('--task_id', type=int, help='task id', default=0)
parser.add_argument('--batch_size', type=int, default=10000, help='batch size')
parser.add_argument('--test_batch_size', type=int, default=100, help='test batch size')
parser.add_argument('--meta_batch_size', type=int, help='meta batch size, namely task num', default=4)
parser.add_argument('--meta_test_batch_size', type=int, help='meta test batch size', default=1)
parser.add_argument('--report_freq', type=float, default=30, help='report frequency')
parser.add_argument('--test_freq', type=float, default=500, help='test frequency')
parser.add_argument('--img_size', type=int, help='img_size', default=84)
parser.add_argument('--imgc', type=int, help='imgc', default=3)
parser.add_argument('--meta_lr_theta', type=float, help='meta-level outer learning rate (theta)', default=3e-5)
parser.add_argument('--update_lr_theta', type=float, help='task-level inner update learning rate (theta)', default=3e-4)
parser.add_argument('--meta_lr_w', type=float, help='meta-level outer learning rate (w)', default=1e-3)
parser.add_argument('--update_lr_w', type=float, help='task-level inner update learning rate (w)', default=0.01)
parser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
parser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--drop_path_prob', type=float, default=0, help='drop path probability')
parser.add_argument('--arch', type=str, default='AUTO_MAML_1', help='which architecture to use')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')

args = parser.parse_args()


def mean_confidence_interval(accs, confidence=0.95):
    n = accs.shape[0]
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h


def main():
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)
    # np.random.seed(args.seed)

    saver = Saver(args)
    # set log
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p',
                        filename=os.path.join(saver.experiment_dir, 'log.txt'), filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)


    saver.create_exp_dir(scripts_to_save=glob.glob('*.py') + glob.glob('*.sh') + glob.glob('*.yml'))
    saver.save_experiment_config()
    summary = TensorboardSummary(saver.experiment_dir)
    writer = summary.create_summary()
    best_pred = 0

    logging.info(args)

    device = torch.device('cuda')
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    #
    # ''' Compute FLOPs and Params '''
    # maml = Meta(args, criterion)
    # flops, params = get_model_complexity_info(maml.model, (84, 84), as_strings=False, print_per_layer_stat=True)
    # logging.info('FLOPs: {} MMac Params: {}'.format(flops / 10 ** 6, params))
    #
    # maml = Meta(args, criterion).to(device)
    # tmp = filter(lambda x: x.requires_grad, maml.parameters())
    # num = sum(map(lambda x: np.prod(x.shape), tmp))
    # #logging.info(maml)
    # logging.info('Total trainable tensors: {}'.format(num))

    # batch_size here means total episode number
    mini = MiniImagenet(args.data_path, mode='train', n_way=args.n_way, k_shot=args.k_spt,
                        k_query=args.k_qry,
                        batch_size=args.batch_size, resize=args.img_size, task_id=None)
    mini_test = MiniImagenet(args.data_path, mode='test', n_way=args.n_way, k_shot=args.k_spt,
                             k_query=args.k_qry,
                             batch_size=args.test_batch_size, resize=args.img_size, task_id=args.task_id)
    train_loader = DataLoader(mini, args.meta_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(mini_test, args.meta_test_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    ''' Decoding '''
    model = Network(args, args.init_channels, args.n_way, args.layers, criterion, pretrained=True).cuda()
    inner_optimizer_theta = torch.optim.SGD(model.arch_parameters(), lr=args.update_lr_theta)
    #inner_optimizer_theta = torch.optim.SGD(model.arch_parameters(), lr=100)
    inner_optimizer_w = torch.optim.SGD(model.parameters(), lr=args.update_lr_w)

    # load state dict
    pretrained_path = '/data2/dongzelian/NAS/meta_nas/run_meta_nas/mini-imagenet/meta-nas/experiment_21/model_best.pth.tar'
    pretrain_dict = torch.load(pretrained_path)['state_dict_w']
    model_dict = {}
    state_dict = model.state_dict()
    for k, v in pretrain_dict.items():
        if k[6:] in state_dict:
            model_dict[k[6:]] = v
        else:
            print(k)
    state_dict.update(model_dict)
    model.load_state_dict(state_dict)
    #model._arch_parameters = torch.load(pretrained_path)['state_dict_theta']


    for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(test_loader):
        x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                     x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)
        for k in range(args.update_step_test):
            logits = model(x_spt, alphas=model.arch_parameters())
            loss = criterion(logits, y_spt)

            inner_optimizer_w.zero_grad()
            inner_optimizer_theta.zero_grad()
            loss.backward()
            inner_optimizer_w.step()
            inner_optimizer_theta.step()


        genotype = model.genotype()
        logging.info(genotype)
        maml = Meta_decoding(args, criterion, genotype).to(device)
        #exit()
        #print(step)
        #print(genotype)



    for epoch in range(args.epoch):
        logging.info('--------- Epoch: {} ----------'.format(epoch))
        accs_all_train = []
        # # TODO: how to choose batch data to update theta?
        # valid_iterator = iter(train_loader)
        batch_time = utils.AverageMeter()
        data_time = utils.AverageMeter()
        update_w_time = utils.AverageMeter()
        end = time.time()
        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(train_loader):
            data_time.update(time.time() - end)
            x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)
            # (x_search_spt, y_search_spt, x_search_qry, y_search_qry), valid_iterator = infinite_get(valid_iterator, train_loader)
            # x_search_spt, y_search_spt, x_search_qry, y_search_qry = x_search_spt.to(device), y_search_spt.to(device), x_search_qry.to(device), y_search_qry.to(device)
            accs, update_w_time = maml(x_spt, y_spt, x_qry, y_qry, update_w_time)
            accs_all_train.append(accs)
            batch_time.update(time.time() - end)
            end = time.time()
            writer.add_scalar('train/acc_iter', accs[-1], step + len(train_loader) * epoch)
            if step % args.report_freq == 0:
                logging.info('Epoch: [{0}][{1}/{2}]\t'
                             'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                             'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                             'W {update_w_time.val:.3f} ({update_w_time.avg:.3f})\t'
                             'training acc: {accs}'.format(
                        epoch, step, len(train_loader),
                        batch_time=batch_time, data_time=data_time,
                        update_w_time=update_w_time, accs=accs))

            if step % args.test_freq == 0:
                test_accs, test_stds, test_ci95 = meta_test(train_loader, test_loader, maml, device, epoch, writer)
                logging.info('[Epoch: {}]\t Test acc: {}\t Test ci95: {}'.format(epoch, test_accs, test_ci95))

                # Save the best meta model.
                new_pred = test_accs[-1]
                if new_pred > best_pred:
                    is_best = True
                    best_pred = new_pred
                else:
                    is_best = False
                saver.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': maml.module.state_dict() if isinstance(maml, nn.DataParallel) else maml.state_dict(),
                    'best_pred': best_pred,
                }, is_best)

        # accs = np.array(accs_all_train).mean(axis=0).astype(np.float16)
        #
        # return accs


def meta_test(train_loader, test_loader, maml, device, epoch, writer):
    accs_all_test = []
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    update_w_time = utils.AverageMeter()
    end = time.time()
    for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(test_loader):
        data_time.update(time.time() - end)
        x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                     x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

        # len(x_spt.shape)=0, args.meta_test_batch_size=1
        accs, update_w_time = maml.finetunning(x_spt, y_spt, x_qry, y_qry, update_w_time)
        accs_all_test.append(accs)
        batch_time.update(time.time() - end)
        end = time.time()

        if step % args.report_freq == 0:
            logging.info('Epoch: [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'W {update_w_time.val:.3f} ({update_w_time.avg:.3f})\t'
                         'test acc: {accs}'.format(
                    epoch, step, len(test_loader),
                    batch_time=batch_time, data_time=data_time,
                    update_w_time=update_w_time,accs=accs))

    # [b, update_step+1]
    accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)  # accs.shape=11
    stds = np.array(accs_all_test).std(axis=0).astype(np.float16)
    ci95 = 1.96 * stds / np.sqrt(np.array(accs_all_test).shape[0])


    #writer.add_scalar('val/acc', accs[-1], step // 500 + (len(train_loader) // 500 + 1) * epoch)
    #writer.add_scalar('val/acc', accs[-1], epoch)


    return accs, stds, ci95



if __name__ == '__main__':
    main()
