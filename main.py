'''
chg1: first change to multi-person pose estimation 
'''
from __future__ import print_function, absolute_import

import argparse
import time
import matplotlib.pyplot as plt

import os

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.datasets as datasets
from torch.autograd import Variable

from pose import Bar
from pose.utils.evaluation import accuracy, final_preds
from pose.utils.misc import save_checkpoint, save_pred

# some functions of setting and adjusting training process
from utils.utils import LRDecay, AverageMeter, mkdir, savefig
from utils.logger import Logger


from pose.utils.osutils import isfile, isdir, join
from pose.utils.imutils import batch_with_heatmap
from pose.utils.transforms import fliplr, flip_back

# model and dataset
import pose.datasets as dtsets
import pose.models as models

from graphviz import Digraph

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))

# joints to calaulate acc
idx = [1,2,3,4,5,6,11,12,15,16]
bestAcc = 0


def main(args):
    global bestAcc

    # create checkpoint dir
    if not isdir(args.checkpoint):
        mkdir(args.checkpoint)

    # create model
    print("==> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](num_classes=16) 

    # multi-GPU
    model = torch.nn.DataParallel(model).cuda()

    # the total number of parameters
    print('    total params num: %.2fM' % (sum(param.numel() for param in model.parameters())/1000000.0))

    # define criterion and optimizer
    criterion = torch.nn.MSELoss().cuda()
    optimizer = torch.optim.RMSprop(model.parameters(),
                                    lr = args.lr,
                                    momentum = args.momentum,
                                    weight_decay = args.wd)

    # resume from a checkpoint
    title = 'mpii-' + args.arch
    if args.resume:
        if isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            bestAcc = checkpoint['bestAcc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            logger = Logger(join(args.checkpoint, 'log.txt'), title=title, resume=True)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        # open the log file
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        # set names of log file
        logger.set_names(['Epoch', 'lr', 'train-loss', 'val-loss', 'train-acc', 'val-acc'])

    # using the fastest algorithm
    cudnn.benchmark = True

    # Data loading code
    train_loader = torch.utils.data.DataLoader(
        dataset=dtsets.Mpii('data/mpii/results.json', args.dataPath),
        batch_size=args.train_batch,
        shuffle=True,
        num_workers=args.jobs,
        pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        dataset=dtsets.Mpii('data/mpii/results.json', args.dataPath, is_train=False),
        batch_size=args.test_batch,
        shuffle=False,
        num_workers=args.jobs,
        pin_memory=True)

    if args.evaluate:
        print('\n    Evaluation:')
        loss, acc, predictions = validate(val_loader, model, criterion, args.debug, args.flip)
        save_pred(predictions, checkpoint=args.checkpoint)
        return

    for epoch in range(args.start_epoch, args.Epochs):
        lr = LRDecay(optimizer, epoch, args.lr, args.schedule, args.gamma)
        print('\nEpoch: %d | lr: %.8f' % (epoch, lr))

        # train for one epoch
        train_loss, train_acc = train(train_loader, model, criterion, optimizer)

        # evaluate on validation set
        valid_loss, valid_acc, predictions = validate(val_loader, model, criterion, args.debug) # , args.flip)

        # append logger file
        logger.append([epoch, lr, train_loss, valid_loss, train_acc, valid_acc])

        # remember best acc and save checkpoint
        is_best = valid_acc > bestAcc
        bestAcc = max(valid_acc, bestAcc)
        save_checkpoint({
            'epoch': epoch,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'bestAcc': bestAcc,
            'optimizer' : optimizer.state_dict(),
        }, predictions, is_best, checkpoint = args.checkpoint)

    logger.close()
    logger.plot()
    plt.savefig(os.path.join(args.checkpoint, 'log.eps'))

def train(train_loader, model, criterion, optimizer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_loss = AverageMeter()
    train_acc = AverageMeter()

    # switch to train mode
    model.train()

    gt_win, pred_win = None, None

    bar = Bar('Processing', max=len(train_loader))

    end = time.time()
    for i, (inp, target, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input_var = torch.autograd.Variable(inp.cuda())
        target_var = torch.autograd.Variable(target.cuda(async=True))

        # compute output
        output = model(input_var)
        print("..>>>>{}.".format(type(output)))

        score_map = output[-1].data.cpu()

        # Calculate intermediate loss
        print("..{}.".format(type(output)))
        loss_pts = criterion(output[0], target_var)
        for j in range(1, len(output)):
            loss_pts += criterion(output[j][:16], target_var)

        acc_pts = accuracy(score_map[:, 0:16, :, :], target, idx)
        acc = acc_pts
        # measure accuracy and save loss

        loss = loss_pts

        train_loss.update(loss.data[0], inp.size(0))
        train_acc.update(acc[0], inp.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.4f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.8f} | acc: {acc: .8f}'.format(
                    batch=i + 1,
                    size=len(train_loader),
                    data=data_time.val,
                    bt=batch_time.val,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=train_loss.avg,
                    acc=train_acc.avg
                    )
        bar.next()

    bar.finish()
    return train_loss.avg, train_acc.avg


def validate(val_loader, model, criterion, debug=False, flip=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    accs = AverageMeter()

    # predictions
    predictions = torch.Tensor(val_loader.dataset.__len__(), 16, 2)

    # switch to evaluate mode
    model.eval()

    gt_win, pred_win = None, None
    end = time.time()
    bar = Bar('Processing', max=len(val_loader))
    print("length of output:{}".format(len(val_loader)))

    for i, (inputs, target, meta) in enumerate(val_loader):
        target = target.cuda(async=True)

        input_var = torch.autograd.Variable(inputs.cuda(), volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        # score_map: 16*64*64
        score_map = output[-1].data.cpu()

        if flip:
            flip_input_var = torch.autograd.Variable(
                    torch.from_numpy(fliplr(inputs.clone().numpy())).float().cuda(),
                    volatile=True
                )
            flip_output_var = model(flip_input_var)
            flip_output = flip_back(flip_output_var[-1].data.cpu())
            score_map += flip_output
        #print("scor")


        loss = 0
        loss_pts = 0
        for o in output:
            loss_pts += criterion(o[:16], target_var)
        # target : 16*64*64
        acc_pts = accuracy(score_map[:16], target.cpu(), idx)

        # generate predictions
        preds = final_preds(score_map[:16], meta['center'], meta['scale'], [64, 64])
        for n in range(score_map[:16].size(0)):
            predictions[meta['index'][n], :, :] = preds[n, :, :]


        if debug:
            gt_batch_img = batch_with_heatmap(inputs, target)
            pred_batch_img = batch_with_heatmap(inputs, score_map[:16])
            if not gt_win or not pred_win:
                plt.subplot(121)
                gt_win = plt.imshow(gt_batch_img)
                plt.subplot(122)
                pred_win = plt.imshow(pred_batch_img)
            else:
                gt_win.set_data(gt_batch_img)
                pred_win.set_data(pred_batch_img)
            plt.pause(.05)
            plt.draw()

        loss = loss_pts
        # measure accuracy and record loss
        losses.update(loss.data[0], inputs.size(0))
        accs.update(acc[0], inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Acc: {acc: .9f}'.format(
                    batch=i + 1,
                    size=len(val_loader),
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    acc=accs.avg
                    )
        bar.next()

    bar.finish()
    return losses.avg, accs.avg, predictions



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # ============================================================
    #                       general options
    # ============================================================
    parser.add_argument('--dataPath',          type=str, default='/data/weigq/mpii/images/', help='path to images data')
    parser.add_argument('--checkpoint',        type=str, default='checkpoint',               help='path to save checkpoint')
    parser.add_argument('--resume',            type=str, default='',                         help='path to latest checkpoint')
    parser.add_argument('-e', '--evaluate',   dest='evaluate', action='store_true',         help='evaluate on val set')
    parser.add_argument('-d', '--debug',      dest='debug',    action='store_true',         help='visualization')
    parser.add_argument('-f', '--flip',       dest='flip',     action='store_true',         help='flip the image')

    # ============================================================
    #                      model options
    # ============================================================
    parser.add_argument('-arch',              default='hg4', choices=model_names,           help='model architecture: '+'/'.join(model_names))
    parser.add_argument('-j', '--jobs',       type=int, default=4,                          help='number of data loading jobs')
    parser.add_argument('--Epochs',           type=int, default=50,                         help='number of total Epochs')
    parser.add_argument('--print-freq',       type=int, default=10,                         help='print frequency')
    

    # ============================================================
    #                    training options
    # ============================================================
    parser.add_argument('--start-epoch',      type=int, default=1,                          help='start epoch')
    parser.add_argument('--train-batch',      type=int, default=6,                          help='train batchsize')
    parser.add_argument('--test-batch',       type=int, default=6,                          help='test batchsize')


    # ============================================================
    #                hyperparemeter options
    # ============================================================
    parser.add_argument('--schedule',         type=int,   default=60,                       help='decrease lr by gamma every schedule')
    parser.add_argument('--gamma',            type=float, default=0.1,                      help='lr decay rate')
    parser.add_argument('--lr',               type=float, default=2.5e-4,                   help='initial learning rate')
    parser.add_argument('--momentum',         type=float, default=0,                        help='momentum')
    parser.add_argument('-wd',                type=float, default=0,                        help='weight decay')


    main(parser.parse_args())
