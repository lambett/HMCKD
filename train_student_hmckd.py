from __future__ import print_function
import os
import argparse
import socket
import time
import numpy as np
import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from models import model_dict
from dataset.cifar100_fft import get_cifar100_dataloaders_sample
from helper.util import *
import pandas as pd
from helper.loops import validate as validate
from helper.pretrain import init
from helper.util import AverageMeter, accuracy
from models.wrapper import wrapper
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
from distiller_zoo import DistillKL

import warnings
warnings.filterwarnings("ignore")
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


def parse_option():

    hostname = socket.gethostname()
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--print_freq', type=int,
                        default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int,
                        default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int,
                        default=40, help='save frequency')
    parser.add_argument('--save_dir', type=str,
                        default='./save_hmckd', help='save dir')
    parser.add_argument('--batch_size', type=int,
                        default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int,
                        default=6, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240,
                        help='number of training epochs')
    # optimization
    parser.add_argument('--learning_rate', type=float,
                        default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str,
                        default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float,
                        default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float,
                        default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['cifar100', 'cifar100_aug'], help='dataset')
    parser.add_argument('--ncls', type=int, default=100, help='class number')

    # model
    parser.add_argument('--model_s', type=str, default='resnet8',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'ResNet50',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2'])
    parser.add_argument('--path_t', type=str, default=None,
                        help='teacher model snapshot')

    # distillation
    parser.add_argument('--distill', type=str, default='kd')
    parser.add_argument('--trial', type=str, default='1', help='trial id')
    parser.add_argument('--ce_weight', type=float,
                        default=0.1, help='weight balance for cross entropy losses')
    parser.add_argument('-a', '--alpha', type=float,
                        default=0.9, help='weight balance for KD')
    parser.add_argument('--alpha_aug', type=float,
                        default=0.9, help='weight balance for mixup KD')
    parser.add_argument('--ss_weight', type=float,
                        default=1.0, help='weight balance for conkd losses')
    parser.add_argument('--label_weight', type=float,
                        default=0.75, help='weight balance for label losses')

    # Top-k
    parser.add_argument('--tf-T', type=float, default=4.0,
                        help='temperature in LT')
    parser.add_argument('--ss-T', type=float, default=0.5,
                        help='temperature in SS')
    parser.add_argument('--ratio-tf', type=float, default=0.75,
                        help='keep how many wrong predictions of LT')
    parser.add_argument('--ratio-ss', type=float, default=0.75,
                        help='keep how many wrong predictions of SS')

    # KL distillation
    parser.add_argument('--kd_T', type=float, default=4,
                        help='temperature for KD distillation')
    parser.add_argument('-b', '--beta', type=float,
                        default=0.8, help='weight balance')
    
    # NCE distillation
    parser.add_argument('--feat_dim', default=128,
                        type=int, help='feature dimension')
    parser.add_argument('--mode', default='exact',
                        type=str, choices=['exact', 'relax'])
    parser.add_argument('--nce_k', default=16384, type=int,
                        help='number of negative samples for NCE')
    parser.add_argument('--loader_type', default="HMCKD", type=str,
                        help='sskdloader')
    parser.add_argument('--mixup_num', default=0, type=int,
                        help='number of positive samples for mixup HNRT')
    parser.add_argument('--mixup_rotate', default=3, type=int,
                        help='using rotate mixup pos img')
    parser.add_argument('--mixup_ratio', default=0.0, type=float,
                        help='mixup ratio for pos img')
    parser.add_argument('--nce_m', default=0.5, type=float,
                        help='momentum for non-parametric updates')
    parser.add_argument('--pos_k', default=-1, type=int,
                        help='number of positive samples for NCE')

    # teacher setting
    parser.add_argument('--hint_layer', default=2,
                        type=int, choices=[0, 1, 2, 3, 4])
    parser.add_argument('--t_gamma', type=float, default=0.1)
    parser.add_argument('--t-milestones', type=int,
                        nargs='+', default=[30, 45])
    parser.add_argument('--t_momentum', type=float, default=0.9)
    parser.add_argument('--t_weight-decay', type=float, default=5e-4)
    parser.add_argument('--t-lr', type=float, default=0.05)
    parser.add_argument('--t-epoch', type=int, default=60)
    parser.add_argument('--t_save_folder', type=str, default='save_finetune')
    parser.add_argument('--few-ratio', type=float, default=1.0)
    parser.add_argument('--ops_err_thres', type=float, default=0.1)
    parser.add_argument('--ops_eps', type=float, default=0.1)
    parser.add_argument('--loss_margin', type=float, default=0.0)


    opt = parser.parse_args()
    print(opt)
    # set different learning rate from these 4 models
    if opt.model_s in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

    # set the path according to the environment
    opt.model_path = f'{opt.save_dir}/student_{opt.distill}_model'
    opt.tb_path = f'{opt.save_dir}/student_{opt.distill}_tensorboards'
    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))
    opt.model_t = get_teacher_name(opt.path_t)
    opt.model_name = 'S:{}_T:{}_{}_{}_a:{}_ag:{}_b:{}_{}'.format(opt.model_s,
                                                                opt.model_t, opt.dataset, opt.distill,
                                                                opt.alpha, opt.alpha_aug, opt.beta,
                                                                opt.trial)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    return opt


def get_teacher_name(model_path):
    """parse teacher name"""
    segments = model_path.split('/')[-2].split('_')
    if segments[0] != 'wrn':
        return segments[0]
    else:
        return segments[0] + '_' + segments[1] + '_' + segments[2]


def load_teacher(model_path, n_cls):
    print('==> loading teacher model')
    model_t = get_teacher_name(model_path)
    model = model_dict[model_t](num_classes=n_cls)
    model.load_state_dict(torch.load(model_path, map_location='cpu')['model'])
    print('==> done')
    return model


def save_train_teacher(model_t, optimizer_t, model_path, best_acc):
    state = {
        'epoch': 60,
        'model': model_t.state_dict(),
        't_acc': best_acc,
        'optimizer': optimizer_t.state_dict(),
    }
    print('saving the teacher model!')
    torch.save(state, model_path)
    return

def mixup(input, alpha, share_lam=False):
    if not isinstance(alpha, (list, tuple)):
        alpha = [alpha, alpha]
    beta = torch.distributions.beta.Beta(*alpha)
    tensor_list = [torch.randperm(4, device=input.device) for _ in range(input.shape[0]//4)]
    randinx = torch.randperm(input.shape[0]//4, device=input.device).unsqueeze(1).expand(input.shape[0]//4, 4).reshape(-1) * 4
    concat_tensor = torch.cat(tensor_list, dim=0)
    randind = randinx + concat_tensor

    if share_lam:
        lam = beta.sample().to(device=input.device)
        lam = torch.max(lam, 1. - lam)
        lam_expanded = lam
    else:
        lam = beta.sample([input.shape[0]]).to(device=input.device)
        lam = torch.max(lam, 1. - lam)
        lam_expanded = lam.view([-1] + [1]*(input.dim()-1))
    output = lam_expanded * input + (1. - lam_expanded) * input[randind]
    return output, randinx, lam

def cutmix(input, alpha):
    if not isinstance(alpha, (list, tuple)):
        alpha = [alpha, alpha]
    beta = torch.distributions.beta.Beta(*alpha)
    randind = torch.randperm(input.shape[0], device=input.device)
    lam = beta.sample().to(device=input.device)
    lam = torch.max(lam, 1. - lam)
    (bbx1, bby1, bbx2, bby2), lam = rand_bbox(input.shape[-2:], lam)
    output = input.clone()
    output[..., bbx1:bbx2, bby1:bby2] = output[randind][..., bbx1:bbx2, bby1:bby2]
    return output, randind, lam

def rand_bbox(size, lam):
    W, H = size
    cut_rat = (1. - lam).sqrt()
    cut_w = (W * cut_rat).to(torch.long)
    cut_h = (H * cut_rat).to(torch.long)

    cx = torch.zeros_like(cut_w, dtype=cut_w.dtype).random_(0, W)
    cy = torch.zeros_like(cut_h, dtype=cut_h.dtype).random_(0, H)

    bbx1 = (cx - cut_w // 2).clamp(0, W)
    bby1 = (cy - cut_h // 2).clamp(0, H)
    bbx2 = (cx + cut_w // 2).clamp(0, W)
    bby2 = (cy + cut_h // 2).clamp(0, H)

    new_lam = 1. - (bbx2 - bbx1).to(lam.dtype) * (bby2 - bby1).to(lam.dtype) / (W * H)

    return (bbx1, bby1, bbx2, bby2), new_lam

def cov(train_matrix, test_matrix):
    r_shape, e_shape = train_matrix.shape, test_matrix.shape
    train_matrix, test_matrix = train_matrix.view(r_shape[0], -1), \
                                test_matrix.view(e_shape[0], -1)
    train_mean, test_mean = torch.mean(train_matrix, dim=0), torch.mean(test_matrix, dim=0)
    tct_matrix = train_matrix[r_shape[0]-e_shape[0]: r_shape[0], :]
    n_dim = train_matrix.shape[1]
    cov_abs = []
    tct_matrix = tct_matrix - train_mean
    test_matrix = test_matrix - test_mean
    for i in range(n_dim):
        rsp_matrix = tct_matrix[:, i].view(e_shape[0], 1)
        mul_mt = rsp_matrix * test_matrix
        cov_ins = torch.sum(mul_mt, dim=0) / (e_shape[0] - 1)
        abs_cov = torch.abs(cov_ins)
        cov_abs.append((torch.sum(abs_cov) / abs_cov.shape[0]).cpu().item())
    return np.sum(cov_abs) / (len(cov_abs))

def main():
    best_acc = 0
    opt = parse_option()
    # tensorboard logger
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)
    # dataloader
    train_loader, val_loader, n_data = get_cifar100_dataloaders_sample(batch_size=opt.batch_size,
                                                                       num_workers=opt.num_workers,
                                                                       k=opt.nce_k,
                                                                       mode=opt.mode,
                                                                       loader_type=opt.loader_type,
                                                                       opt=opt
                                                                       )

    str_train_loader, _, _ = get_cifar100_dataloaders_sample(batch_size=opt.batch_size,
                                                             num_workers=opt.num_workers,
                                                             k=opt.nce_k,
                                                             mode=opt.mode,
                                                             loader_type=opt.loader_type,
                                                             opt=opt, strong_transform=True
                                                             )

    n_cls = opt.ncls  # default  100 for cifar100
    # model
    model_t = load_teacher(opt.path_t, n_cls)
    model_s = model_dict[opt.model_s](num_classes=n_cls)
    data = torch.randn(2, 3, 32, 32)  # default (3,32,32) for cifar100
    model_t.eval()
    model_s.eval()
    feat_t, _, _ = model_t(data, is_feat=True)
    feat_s, _, _ = model_s(data, is_feat=True)

    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(opt.kd_T)
    opt.s_dim = feat_s[-1].shape[1]
    opt.t_dim = feat_t[-1].shape[1]
    mixup_num = max(1, opt.mixup_num + 1)
    model_t = wrapper(model_t, opt.feat_dim).cuda()
    model_t_path = f'save_hmckd_t/{get_teacher_name(opt.path_t)}_embed.pth'
    if os.path.exists(model_t_path):
        model_t.load_state_dict(torch.load(
            model_t_path, map_location='cpu')['model'])
        model_t.eval()
    else:
        t_optimizer = optim.SGD([{'params': model_t.backbone.parameters(), 'lr': 0.0},
                                {'params': model_t.proj_head.parameters(),
                                 'lr': opt.t_lr},
                                {'params': model_t.classifier.parameters(), 'lr': opt.t_lr}],
                                momentum=opt.t_momentum,
                                weight_decay=opt.t_weight_decay)
        model_t.eval()
        t_scheduler = MultiStepLR(
            t_optimizer, milestones=opt.t_milestones, gamma=opt.t_gamma)
        # train ssp_head
        for epoch in range(opt.t_epoch):
            model_t.eval()
            loss_record = AverageMeter()
            acc_record = AverageMeter()
            start = time.time()
            for idx, data in enumerate(train_loader):
                x, target, _, _, _ = data
                x = x.cuda() # 64,4,3,32,32
                target = target.cuda()
                t_optimizer.zero_grad()
                c, h, w = x.size()[-3:]
                x = x.view(-1, c, h, w)
                out, feat, proj_x, proj_logit = model_t(x, bb_grad=False)
                batch = int(x.size(0) / mixup_num)
                target = target.unsqueeze(1).expand(
                    batch, mixup_num).reshape(-1)
                loss = F.cross_entropy(proj_logit, target)
                loss.backward()
                t_optimizer.step()
                batch_acc = accuracy(proj_logit, target, topk=(1,))[0]
                loss_record.update(loss.item(), batch)
                acc_record.update(batch_acc.item(), batch)
            run_time = time.time() - start
            info = f'teacher_train_Epoch:{epoch}/{opt.t_epoch}\t run_time:{run_time:.3f}\t t_loss:{loss_record.avg:.3f}\t t_acc:{acc_record.avg:.2f}\t'
            print(info, flush=True)
        save_train_teacher(model_t, t_optimizer, model_t_path, 99)
    opt.t_dim = opt.feat_dim
    opt.n_data = n_data
    # optimizer
    optimizer = optim.SGD(trainable_list.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    # append teacher after optimizer to avoid weight_decay
    module_list.append(model_t)
    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    # classification loss
    criterion_list.append(criterion_div)    # KL-divergence loss
    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        cudnn.benchmark = True

    # validate teacher accuracy
    teacher_acc, _, _, _ = validate(val_loader, model_t, criterion_cls, opt)
    print('teacher accuracy: ', teacher_acc, flush=True)

    # routine
    cov_list = []

    for epoch in range(1, opt.epochs + 2):
        # set modules as train()
        for module in module_list:
            module.train()
        # set teacher as eval()
        module_list[-1].eval()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_cls = AverageMeter()
        losses_div_nor = AverageMeter()
        losses_div_aug = AverageMeter()
        losses_conkd = AverageMeter()
        losses_label = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        end = time.time()
        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")
        time1 = time.time()
        small_cov = 100.0
        train_matrix = None

        # ---------------------------------------------------
        source_iterater = iter(str_train_loader)
        for idx, data in enumerate(train_loader):
            try:
                data1 = next(source_iterater)
            except StopIteration:
                source_iterater = iter(str_train_loader)
                data1 = next(source_iterater)
            data_all = [(data, data1)]
            for s, (data, data1) in enumerate(data_all):
                data_time.update(time.time() - end)
                # From 16384 contrast_idx get 3 items to mixup with index, contrast_idx 16385 mixup_indexes 4
                input, target, index, contrast_idx, mixup_indexes = data
                input = input.cuda()
                target = target.cuda()
                index = index.cuda()
                mixup_indexes = mixup_indexes.cuda()
                if isinstance(contrast_idx, list):
                    contrast_idx = [sub_idx.cuda()
                                    for sub_idx in contrast_idx]
                else:
                    contrast_idx = contrast_idx.cuda()
                c, h, w = input.size()[-3:]  # 256,3,32,32
                input = input.view(-1, c, h, w).cuda()
                target = target.cuda()
                batch = int(input.size(0) // mixup_num)  # 64
                # the no.1 is true, the no.2 to no.4 is false
                nor_index = (torch.arange(mixup_num * batch) %
                             mixup_num == 0).cuda()
                aug_index = (torch.arange(mixup_num * batch) %
                             mixup_num != 0).cuda()

                input1, target1, index1, contrast_idx1, mixup_indexes1 = data1
                input1 = input1.cuda()
                input1 = input1.view(-1, c, h, w).cuda()
                target1 = target1.cuda()
                index1 = index1.cuda()
                mixup_indexes1 = mixup_indexes1.cuda()

                im_q, labels_aux, lam = mixup(input1, 1.0)
                im_q = torch.cat([input, im_q], dim=0)
                labels_aux = torch.round((labels_aux / input1.shape[0]) * 100).to(torch.int64)

                feat_s, logit_s, covout = model_s(im_q, is_feat=True)
                f_s, feat_s_str = feat_s[-1][:batch*4], feat_s[-1][batch*4:]
                logit_s, logit_s_str = logit_s[:batch*4], logit_s[batch*4:]

                loss_label = (lam * nn.CrossEntropyLoss(reduction='none').cuda()(logit_s_str, target1.unsqueeze(1).expand(batch, mixup_num).reshape(-1)) +
                              (1. - lam) * nn.CrossEntropyLoss(reduction='none').cuda()(logit_s_str, labels_aux)).mean()

                if train_matrix is None:
                    train_matrix = covout[:batch]
                else:
                    train_matrix = torch.cat((train_matrix, covout[:batch]), dim=0)

                with torch.no_grad():
                    # (256,64) (256,100) (256,128) (256,100)
                    feat_t, logit_t, proj_x, proj_logit = model_t(input)
                    f_t = proj_x.detach()
                    #---------------------------------------------------
                    aug_knowledge = F.softmax(proj_logit[aug_index] / opt.tf_T, dim=1)

                #---------------------------------------------------
                aug_target = target.unsqueeze(1).expand(-1, 3).contiguous().view(-1).long().cuda()
                rank = torch.argsort(aug_knowledge, dim=1, descending=True)
                rank = torch.argmax(torch.eq(rank, aug_target.unsqueeze(1)).long(), dim=1)
                index_tf = torch.argsort(rank)
                tmp = torch.nonzero(rank, as_tuple=True)[0]
                wrong_num = tmp.numel()
                correct_num = 3 * batch - wrong_num
                wrong_keep = int(wrong_num * opt.ratio_tf)
                index_tf = index_tf[:correct_num + wrong_keep]
                distill_index_tf = torch.sort(index_tf)[0]

                s_nor_feat = f_s[nor_index]
                s_aug_feat = f_s[aug_index]
                s_nor_feat = s_nor_feat.unsqueeze(2).expand(-1, -1, 3 * batch).transpose(0, 2)
                s_aug_feat = s_aug_feat.unsqueeze(2).expand(-1, -1, 1 * batch)
                s_simi = F.cosine_similarity(s_aug_feat, s_nor_feat, dim=1)
                t_nor_feat = f_t[nor_index]
                t_aug_feat = f_t[aug_index]
                t_nor_feat = t_nor_feat.unsqueeze(2).expand(-1, -1, 3 * batch).transpose(0, 2)
                t_aug_feat = t_aug_feat.unsqueeze(2).expand(-1, -1, 1 * batch)
                t_simi = F.cosine_similarity(t_aug_feat, t_nor_feat, dim=1)
                # t_simi = t_simi.detach()
                aug_target = torch.arange(batch).unsqueeze(1).expand(-1, 3).contiguous().view(-1).long().cuda()
                rank = torch.argsort(t_simi, dim=1, descending=True)
                rank = torch.argmax(torch.eq(rank, aug_target.unsqueeze(1)).long(), dim=1)
                index_ss = torch.argsort(rank)
                tmp = torch.nonzero(rank, as_tuple=True)[0]
                wrong_num = tmp.numel()
                correct_num = 3 * batch - wrong_num
                wrong_keep = int(wrong_num * opt.ratio_ss)
                index_ss = index_ss[:correct_num + wrong_keep]
                distill_index_ss = torch.sort(index_ss)[0]
                log_simi = F.log_softmax(s_simi / opt.ss_T, dim=1)
                simi_knowledge = F.softmax(t_simi / opt.ss_T, dim=1)



                # get all loss
                logit_s_nor = logit_s[nor_index]
                logit_t_nor = logit_t[nor_index]
                logit_s_aug = logit_s[aug_index]
                logit_t_aug = logit_t[aug_index]
                target = target.unsqueeze(1).expand(
                    batch, mixup_num).reshape(-1)

                # student's cls loss
                loss_cls = criterion_cls(logit_s, target)

                if opt.alpha > 0:
                    loss_div_nor = criterion_div(logit_s_nor, logit_t_nor)
                else:
                    loss_div_nor = torch.Tensor([0]).cuda()

                if opt.alpha_aug > 0:
                    loss_div_aug = criterion_div(logit_s_aug[distill_index_tf], logit_t_aug[distill_index_tf])

                else:
                    loss_div_aug = torch.Tensor([0]).cuda()

                loss_conkd = F.kl_div(log_simi[distill_index_ss], simi_knowledge[distill_index_ss], \
                                      reduction='batchmean') * opt.ss_T * opt.ss_T

                loss = opt.ce_weight * loss_cls + opt.alpha * loss_div_nor + opt.alpha_aug * loss_div_aug \
                       + opt.ss_weight * loss_conkd + opt.label_weight * loss_label


                acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))
                losses.update(loss.item(), input.size(0))
                losses_cls.update(loss_cls.item(), input.size(0))
                losses_div_nor.update(loss_div_nor.item(), input.size(0))
                losses_div_aug.update(loss_div_aug.item(), input.size(0))
                losses_conkd.update(loss_conkd.item(), input.size(0))
                losses_label.update(loss_label.item(), input.size(0))
                top1.update(acc1[0], input.size(0))
                top5.update(acc5[0], input.size(0))
                # ===================backward=====================
                if epoch > 1:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                # ===================meters=====================
                batch_time.update(time.time() - end)
                end = time.time()

                if idx % opt.print_freq == 0:
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Loss_cls {loss_cls.val:.4f} ({loss_cls.avg:.4f})\t'
                          'Loss_div_nor {loss_div_nor.val:.4f} ({loss_div_nor.avg:.4f})\t'
                          'Loss_div_aug {loss_div_aug.val:.4f} ({loss_div_aug.avg:.4f})\t'
                          'Loss_conkd {loss_conkd.val:.4f} ({loss_conkd.avg:.4f})\t'
                          'Loss_label {loss_label.val:.4f} ({loss_label.avg:.4f})\t'
                          'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                              epoch, idx, len(train_loader), batch_time=batch_time,
                              data_time=data_time, loss=losses, loss_cls=losses_cls,
                              loss_div_nor=losses_div_nor, loss_div_aug=losses_div_aug,
                              loss_conkd=losses_conkd, loss_label=losses_label, top1=top1, top5=top5), flush=True)

        # ---------------------------------------------------
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        time2 = time.time()
        logger.log_value('train_acc', top1.avg, epoch)
        logger.log_value('train_loss', losses.avg, epoch)

        test_acc, tect_acc_top5, test_loss, test_matrix = validate(
            val_loader, model_s, criterion_cls, opt)
        # save the best model
        logger.log_value('test_acc', test_acc, epoch)
        logger.log_value('test_loss', test_loss, epoch)
        logger.log_value('test_acc_top5', tect_acc_top5, epoch)

        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'best_acc': best_acc,
            }
            save_file = os.path.join(
                opt.save_folder, '{}_best.pth'.format(opt.model_s))
            print('saving the best model!')
            torch.save(state, save_file)

        # regular saving
        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'accuracy': test_acc,
            }
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

        cov_start = time.time()
        cov_item = cov(train_matrix, test_matrix)
        cov_end = time.time()
        print('cov_item is: ', cov_item, 'cov cost time is ', cov_end - cov_start)
        if small_cov > cov_item:
            small_cov = cov_item
            print('small_cov:', small_cov)
        cov_list.append(cov_item)

    Pd_data_for_covs = pd.DataFrame(cov_list)
    Pd_data_for_covs.to_csv(
        "log_hmckd_cov/" + "hmckd" + "_" + get_teacher_name(opt.path_t) + "_" + opt.model_s + "_" + str(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())) + ".csv", header=False,
        index=True)

    # This best accuracy is only for printing purpose.
    # The results reported in the paper/README is from the last epoch.
    print('best accuracy:', best_acc)
    state = {
        'opt': opt,
        'model': model_s.state_dict(),
    }
    save_file = os.path.join(
        opt.save_folder, '{}_last.pth'.format(opt.model_s))
    torch.save(state, save_file)

if __name__ == '__main__':
    main()
