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
import torchvision.transforms as transforms

from pose import Bar
from pose.utils.logger import Logger
from pose.utils.evaluation import accuracy, AverageMeter, final_preds
from pose.utils.misc import save_checkpoint, save_pred, LRDecay

from pose.utils.osutils import mkdir_p, isfile, isdir, join
from pose.utils.imutils import batch_with_heatmap
from pose.utils.transforms import fliplr, flip_back
import pose.models as models
import pose.datasets as datasets


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

# to calaulate acc
idx = [1,2,3,4,5,6,11,12,15,16]

best_acc = 0



def main(args):
    global best_acc

    # create checkpoint dir
    if not isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # create model
    print("==> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](num_classes=16)

    # multi-GPU
    model = torch.nn.DataParallel(model).cuda()

    # the total number of parameters
    print('    Total params size: %.2fM' % (sum(para.numel() for para in model.parameters())/1000000.0))

    # define criterion and optimizer
    criterion = torch.nn.MSELoss(size_average=True).cuda()
    optimizer = torch.optim.RMSprop(model.parameters(),
                                lr = args.lr,
                                momentum = args.momentum,
                                weight_decay = args.weight_decay)

    # optionally resume from a checkpoint



    # --------
    title = 'mpii-' + args.arch
    if args.resume:
        if isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            logger = Logger(join(args.checkpoint, 'log.txt'), title=title, resume=True)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    # --------


    else:
        # open the log file
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        # set names of log file
        logger.set_names(['train-loss', 'val-loss', 'val-acc'])

    # using the fastest algorithm
    cudnn.benchmark = True

    # Data loading code
    train_loader = torch.utils.data.DataLoader(
        dataset = datasets.Mpii('data/mpii/mpii_annotations.json', args.dataPath),
        batch_size = args.train_batch,
        shuffle = True,
        num_workers = args.workers,
        pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        dataset = datasets.Mpii('data/mpii/mpii_annotations.json', args.dataPath, train=False),
        batch_size = args.test_batch,
        shuffle = False,
        num_workers = args.workers,
        pin_memory=True)

    if args.evaluate:
        print('\nEvaluation only')
        loss, acc, predictions = validate(val_loader, model, criterion, args.debug, args.flip)
        save_pred(predictions, checkpoint=args.checkpoint)
        return

    for epoch in range(args.start_epoch, args.Epochs):
        # lr decay
        lr = LRDecay(optimizer, epoch, args.lr)

        print('\nEpoch: %d | lr: %.8f' % (epoch, lr))

        # train for one epoch
        train_loss = train(train_loader, model, criterion, optimizer, epoch - 1, args.debug)

        # evaluate on validation set
        valid_loss, valid_acc, predictions = validate(val_loader, model, criterion, args.debug, args.flip)

        # append logger file
        logger.append([train_loss, valid_loss, valid_acc])

        # remember best acc and save checkpoint
        is_best = valid_acc > best_acc
        best_acc = max(valid_acc, best_acc)
        save_checkpoint({
            'epoch': epoch,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
        }, predictions, is_best, checkpoint = args.checkpoint)

    logger.close()
    logger.plot()
    plt.savefig(os.path.join(args.checkpoint, 'log.eps'))

def train(train_loader, model, criterion, optimizer, epoch, debug=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    gt_win, pred_win = None, None

    bar = Bar('Processing', max=len(train_loader))
    print("the length of train_loader: {}".format(len(train_loader)))

    for i, (inputs, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        inputs = inputs.cuda()
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(inputs)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)

        # Calculate intermediate loss
        loss = criterion(output[0], target_var)
        for j in range(1, len(output)):
            loss += criterion(output[j], target_var)

        if debug: # visualize groundtruth and predictions
            gt_batch_img = batch_with_heatmap(inputs, target)
            pred_batch_img = batch_with_heatmap(inputs, output[-1].data)
            if not gt_win or not pred_win:
                ax1 = plt.subplot(121)
                ax1.title.set_text('Groundtruth')
                gt_win = plt.imshow(gt_batch_img)
                ax2 = plt.subplot(122)
                ax2.title.set_text('Prediction')
                pred_win = plt.imshow(pred_batch_img)
            else:
                gt_win.set_data(gt_batch_img)
                pred_win.set_data(pred_batch_img)
            plt.pause(.05)
            plt.draw()

        # measure accuracy and record loss
        losses.update(loss.data[0], inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.8f}'.format(
                    batch=i + 1,
                    size=len(train_loader),
                    data=data_time.val,
                    bt=batch_time.val,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    )
        bar.next()

    bar.finish()
    return losses.avg


def validate(val_loader, model, criterion, debug=False, flip=True):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()

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
        for o in output:
            loss += criterion(o, target_var)
        # target : 16*64*64
        acc = accuracy(score_map.cuda(), target, idx)

        # generate predictions
        preds = final_preds(score_map, meta['center'], meta['scale'], [64, 64])
        for n in range(score_map.size(0)):
            predictions[meta['index'][n], :, :] = preds[n, :, :]


        if debug:
            gt_batch_img = batch_with_heatmap(inputs, target)
            pred_batch_img = batch_with_heatmap(inputs, score_map)
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

        # measure accuracy and record loss
        losses.update(loss.data[0], inputs.size(0))
        acces.update(acc[0], inputs.size(0))

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
                    acc=acces.avg
                    )
        bar.next()

    bar.finish()
    return losses.avg, acces.avg, predictions

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='hg_pytorch training')

    ## General options
    parser.add_argument('-dataPath',  default = '/home/guoqiang/hg_train/data/mpii/images/',
                                      help = 'the path to images data')

    ## Model options
    parser.add_argument('-arch',      default = 'hg4', metavar = 'ARCH', choices = model_names,
                                      help = 'model architecture: '+' | '.join(model_names)+' (default: resnet18)')
    parser.add_argument('-j', '--workers', default = 1, type = int, metavar = 'N',
                                      help = 'number of data loading workers (default: 4)')
    parser.add_argument('--Epochs',   default = 50, type = int, metavar='EPOCH',
                                      help = 'number of total Epochs to run')
    parser.add_argument('--start-epoch', default = 1, type = int,
                                      help = 'manual epoch number (useful for continue)')
    parser.add_argument('--train-batch', default = 6, type = int,
                                      help = 'train batchsize')
    parser.add_argument('--test-batch', default = 6, type = int,
                                      help = 'test batchsize')
    parser.add_argument('--lr',       default = 2.5e-4, type = float,
                                      help = 'initial learning rate')
    parser.add_argument('--momentum', default = 0, type = float,
                                      help = 'momentum')
    parser.add_argument('--weight-decay', '--wd', default = 0, type = float,
                                      help = 'weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default = 10, type = int,
                                      help = 'print frequency (default: 10)')
    parser.add_argument('-c', '--checkpoint', default = 'checkpoint', type = str, metavar='PATH',
                                      help = 'path to save checkpoint (default: checkpoint)')
    parser.add_argument('--resume',   default = '', type = str, metavar='PATH',
                                      help = 'path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest = 'evaluate', action = 'store_true',
                                      help = 'evaluate model on validation set')
    parser.add_argument('-d', '--debug', dest = 'debug', action = 'store_true',
                                      help = 'show intermediate results')
    parser.add_argument('-f', '--flip', dest = 'flip', action = 'store_true',
                                      help = 'flip the input during validation')

    main(parser.parse_args())
