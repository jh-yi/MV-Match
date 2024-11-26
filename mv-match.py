"""
Implementation of BMVC 2024 paper "MV-Match: Multi-View Matching for Domain-Adaptive Identification of Plant Nutrient Deficiencies". This code is adapted from the implementation available at: https://github.com/thuml/Transfer-Learning-Library
Author: Jinhui Yi
Contact: jinhui.yi@uni-bonn.de
"""
import sys, os
import pickle as pkl
import shutil, datetime, time
import random
import time
import warnings
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tllib.modules.classifier import Classifier
from tllib.vision.transforms import MultipleApply
from tllib.utils.data import ForeverDataIterator
from tllib.utils.metric import accuracy
from tllib.utils.meter import AverageMeter, ProgressMeter
from tllib.utils.logger import CompleteLogger

import utils

sys.path.append(os.getcwd())

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")

class ImageClassifier(Classifier):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck_dim=1024, **kwargs):
        bottleneck = nn.Sequential(
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
        super(ImageClassifier, self).__init__(backbone, num_classes, bottleneck, bottleneck_dim, **kwargs)

    def forward(self, x: torch.Tensor):
        """"""
        f = self.pool_layer(self.backbone(x))
        f = self.bottleneck(f)
        predictions = self.head(f)
        # return predictions
        return predictions, f


def main(args: argparse.Namespace):

    t_start = time.time()
    log_source(args.log) 
    logger = CompleteLogger(args.log, args.phase)
    print("---------------- start ----------------\n", datetime.datetime.now().strftime("%A %Y-%m-%d %H:%M:%S"), "\n---------------------------------------")
    print('Hostname: ', os.uname()[1])
    print(args)


    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True

    # Data loading code
    weak_augment = utils.get_train_transform(args.train_resizing, scale=args.scale, ratio=args.ratio, ## !
                                             random_horizontal_flip=not args.no_hflip,
                                             random_color_jitter=False, resize_size=args.resize_size,
                                             norm_mean=args.norm_mean, norm_std=args.norm_std)
    strong_augment = utils.get_train_transform(args.train_resizing, scale=args.scale, ratio=args.ratio, ## !
                                               random_horizontal_flip=not args.no_hflip,
                                               random_color_jitter=False, resize_size=args.resize_size,
                                               norm_mean=args.norm_mean, norm_std=args.norm_std,
                                               auto_augment=args.auto_augment)
    train_source_transform = MultipleApply([weak_augment, strong_augment])
    train_target_transform = train_source_transform
    val_transform = utils.get_val_transform(args.val_resizing, resize_size=args.resize_size,
                                            norm_mean=args.norm_mean, norm_std=args.norm_std)

    print("train_source_transform: ", train_source_transform)
    print("train_target_transform: ", train_target_transform)
    print("val_transform: ", val_transform)

    train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, args.class_names = \
        utils.get_dataset(args.data, args.root, args.source, args.target, train_source_transform, val_transform,
                          train_target_transform=train_target_transform, method=args.log.split('/')[-2], sample=args.sample)
    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.unlabeled_batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    # create model
    print("=> using model '{}'".format(args.arch))
    backbone = utils.get_model(args.arch, pretrain=not args.scratch)
    pool_layer = nn.Identity() if args.no_pool else None
    classifier = ImageClassifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim,
                                 pool_layer=pool_layer, finetune=not args.scratch).to(device)
    # print(classifier)

    # define optimizer and lr scheduler
    optimizer = SGD(classifier.get_parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
                    nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    # resume from the best checkpoint
    if args.phase != 'train':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        classifier.load_state_dict(checkpoint)

    if args.phase == 'test':
        acc1 = utils.validate(test_loader, classifier, args, device)
        print(acc1)
        return

    # start training
    best_acc1 = 0.
    for epoch in range(args.epochs):
        print("lr:", lr_scheduler.get_last_lr())
        # train for one epoch
        train(train_source_iter, train_target_iter, classifier, optimizer, lr_scheduler, epoch, args)

        # evaluate on validation set
        acc1 = utils.validate(val_loader, classifier, args, device)

        # optional: evaluate on test set
        acc2 = utils.validate(test_loader, classifier, args, device)

        # remember best acc@1 and save checkpoint
        torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest'))
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
        best_acc1 = max(acc1, best_acc1)

    print("best_acc1 = {:3.1f}".format(best_acc1))

    # evaluate on test set
    classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best'), map_location='cuda:0'))
    acc1 = utils.validate(test_loader, classifier, args, device)
    print("test_acc1 = {:3.1f}".format(acc1))

    logger.close()
    print("---------------- end ----------------\n", datetime.datetime.now().strftime("%A %Y-%m-%d %H:%M:%S"), "\n---------------------------------------")
    print("Elapsed time: {:.1f}h".format((time.time()-t_start)/60))

def train(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator,
          model: ImageClassifier, optimizer: SGD, lr_scheduler: LambdaLR, epoch: int, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':5.2f')
    data_time = AverageMeter('Data', ':5.2f')
    cls_losses = AverageMeter('Cls Loss', ':6.2f')
    self_training_losses = AverageMeter('Self Training Loss', ':6.2f')
    losses = AverageMeter('Loss', ':6.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    pseudo_label_ratios = AverageMeter('Pseudo Label Ratio', ':3.1f')
    pseudo_label_accs = AverageMeter('Pseudo Label Acc', ':3.1f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, cls_losses, self_training_losses, cls_accs, pseudo_label_accs,
         pseudo_label_ratios],
        prefix="Epoch: [{}]".format(epoch))

    self_training_criterion = ConfidenceBasedSelfTrainingLoss(args.threshold, args.use_soft_label).to(device)
    # switch to train mode
    model.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        ((x_s, x_s_strong), (x_sm, x_sm_strong)), labels_s = next(train_source_iter)[:2]
        ((x_t, x_t_strong), (x_tm, x_tm_strong)), labels_t = next(train_target_iter)[:2]

        x_s = x_s.to(device)
        x_t = x_t.to(device)
        labels_s = labels_s.to(device)
        labels_t = labels_t.to(device)

        # x_sm = x_sm.to(device)
        x_sm_strong = x_sm_strong.to(device)
        x_t_strong = x_t_strong.to(device)
        # x_tm = x_tm.to(device)
        x_tm_strong = x_tm_strong.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # clear grad
        optimizer.zero_grad()

        # compute output
        with torch.no_grad():
            y_t, f_t = model(x_t)

        # cross entropy loss
        y_s, f_s = model(x_s)
        cls_loss = F.cross_entropy(y_s, labels_s)
        cls_loss.backward()

        self_training_loss = None
        # self-training loss
        y_t_strong, _ = model(x_t_strong)
        self_training_loss, mask, pseudo_labels = self_training_criterion(y_t_strong, y_t)
        self_training_loss = args.trade_off * self_training_loss
        self_training_loss.backward()

        # y_tm, _ = model(x_tm)
        # self_training_loss_tm, mask, pseudo_labels = self_training_criterion(y_tm, y_t)
        # self_training_loss_tm = args.trade_off * self_training_loss_tm
        # self_training_loss_tm.backward()
        # if self_training_loss is not None:
        #     self_training_loss += self_training_loss_tm
        # else:
        #     self_training_loss = self_training_loss_tm

        y_tm_strong, _ = model(x_tm_strong)
        self_training_loss_tms, mask, pseudo_labels = self_training_criterion(y_tm_strong, y_t)
        self_training_loss_tms = args.trade_off * self_training_loss_tms
        self_training_loss_tms.backward()
        if self_training_loss is not None:
            self_training_loss += self_training_loss_tms
        else:
            self_training_loss = self_training_loss_tms

        # y_sm, _ = model(x_sm)
        # self_training_loss_sm, mask, pseudo_labels = self_training_criterion(y_sm, y_s.detach())
        # self_training_loss_sm = args.trade_off * self_training_loss_sm
        # self_training_loss_sm.backward()
        # self_training_loss += self_training_loss_sm

        y_sm_strong, _ = model(x_sm_strong)
        self_training_loss_sms, mask, pseudo_labels = self_training_criterion(y_sm_strong, y_s.detach())
        # self_training_loss_sms = F.cross_entropy(y_sm_strong, labels_s) # worse
        self_training_loss_sms = args.trade_off * self_training_loss_sms
        self_training_loss_sms.backward()
        self_training_loss += self_training_loss_sms

        # measure accuracy and record loss
        loss = cls_loss + self_training_loss
        losses.update(loss.item(), x_s.size(0))
        cls_losses.update(cls_loss.item(), x_s.size(0))
        self_training_losses.update(self_training_loss.item(), x_s.size(0))

        cls_acc = accuracy(y_s, labels_s)[0]
        cls_accs.update(cls_acc.item(), x_s.size(0))

        if mask is not None:
            # ratio of pseudo labels
            n_pseudo_labels = mask.sum()
            ratio = n_pseudo_labels / x_t.size(0)
            pseudo_label_ratios.update(ratio.item() * 100, x_t.size(0))

            # accuracy of pseudo labels
            if n_pseudo_labels > 0:
                pseudo_labels = pseudo_labels * mask - (1 - mask)
                n_correct = (pseudo_labels == labels_t).float().sum()
                pseudo_label_acc = n_correct / n_pseudo_labels * 100
                pseudo_label_accs.update(pseudo_label_acc.item(), n_pseudo_labels)

        # compute gradient and do SGD step
        optimizer.step()
        lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MV-Match')
    # dataset parameters
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='MiPlo', choices=utils.get_dataset_names(),
                        help='dataset: ' + ' | '.join(utils.get_dataset_names()) +
                             ' (default: MiPlo)')
    parser.add_argument('-s', '--source', help='source domain(s)', nargs='+')
    parser.add_argument('-t', '--target', help='target domain(s)', nargs='+')
    parser.add_argument('--train-resizing', type=str, default='default')
    parser.add_argument('--val-resizing', type=str, default='default')
    parser.add_argument('--resize-size', type=int, default=224,
                        help='the image size after resizing')
    parser.add_argument('--scale', type=float, nargs='+', default=[0.5, 1.0], metavar='PCT',
                        help='Random resize scale (default: 0.5 1.0)')
    parser.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
                        help='Random resize aspect ratio (default: 0.75 1.33)')
    parser.add_argument('--no-hflip', action='store_true',
                        help='no random horizontal flipping during training')
    parser.add_argument('--norm-mean', type=float, nargs='+',
                        default=(0.485, 0.456, 0.406), help='normalization mean')
    parser.add_argument('--norm-std', type=float, nargs='+',
                        default=(0.229, 0.224, 0.225), help='normalization std')
    parser.add_argument('--auto-augment', default='rand-m10-n2-mstd2', type=str,
                        help='AutoAugment policy (default: rand-m10-n2-mstd2)')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=utils.get_model_names(),
                        help='backbone architecture: ' +
                             ' | '.join(utils.get_model_names()) +
                             ' (default: resnet18)')
    parser.add_argument('--bottleneck-dim', default=1024, type=int,
                        help='Dimension of bottleneck')
    parser.add_argument('--no-pool', action='store_true',
                        help='no pool layer after the feature extractor.')
    parser.add_argument('--scratch', action='store_true', help='whether train from scratch.')
    parser.add_argument('--trade-off', default=1., type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    parser.add_argument('--use_soft_label', action='store_true',
                        help='use soft labels instead of hard labels')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('-ub', '--unlabeled-batch-size', default=32, type=int,
                        help='mini-batch size of unlabeled data (target domain) (default: 32)')
    parser.add_argument('--threshold', default=0.9, type=float,
                        help='confidence threshold')
    parser.add_argument('--lr', '--learning-rate', default=0.003, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.0004, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 1e-3)',
                        dest='weight_decay')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=1000, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default='fixmatch',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    
    parser.add_argument("--sample", type=str, default='random', # random, mutual, ...
                        help="Sampling method. Only for saving cache")
    # parser.add_argument("--n_view", type=int, default=1000, 
    #                 help="Number of views")
    args = parser.parse_args()
    main(args)
