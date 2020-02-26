"""Training for yogi dataset."""
import argparse
import os
import sys

import torch
import torch.backends.cudnn
from torch.nn import DataParallel
from torch.optim.rmsprop import RMSprop
from torch.optim.adam import Adam
from torch.utils.data import DataLoader

from yogi import config
from data_handlers.yogi_db import YogiDB
from yogi.models import ImageSet
from stacked_hourglass import hg1, hg2, hg8
from stacked_hourglass.datasets.generic import Generic
from stacked_hourglass.train import do_training_epoch, do_validation_epoch
from stacked_hourglass.utils.logger import Logger, savefig
from stacked_hourglass.utils.misc import save_checkpoint, adjust_learning_rate


def main(args):
    """Train/ Cross validate for data source = YogiDB."""
    # Create data loader
    """Generic(data.Dataset)(image_set, annotations,
                     is_train=True, inp_res=256, out_res=64, sigma=1,
                     scale_factor=0, rot_factor=0, label_type='Gaussian',
                     rgb_mean=RGB_MEAN, rgb_stddev=RGB_STDDEV)."""
    annotations_source = 'basic-thresholder'

    # Get the data from yogi
    db_obj = YogiDB(config.db_url)
    imageset = db_obj.get_filtered(ImageSet,
                                   name=args.image_set_name)
    annotations = db_obj.get_annotations(image_set_name=args.image_set_name,
                                         annotation_source=annotations_source)
    pts = torch.Tensor(annotations[0]['joint_self'])
    num_classes = pts.size(0)
    dataset = Generic(image_set=imageset,
                      inp_res=args.inp_res,
                      out_res=args.out_res,
                      annotations=annotations,
                      mode=args.mode)

    train_dataset = dataset
    train_loader = DataLoader(train_dataset,
                              batch_size=args.train_batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True)

    val_dataset = dataset
    val_dataset.is_train = False
    val_loader = DataLoader(val_dataset,
                            batch_size=args.test_batch, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    # Select the hardware device to use for inference.
    if torch.cuda.is_available():
        device = torch.device('cuda', torch.cuda.current_device())
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    # Disable gradient calculations by default.
    torch.set_grad_enabled(False)

    # create checkpoint dir
    os.makedirs(args.checkpoint, exist_ok=True)

    if args.arch == 'hg1':
        model = hg1(pretrained=False, num_classes=num_classes)
    elif args.arch == 'hg2':
        model = hg2(pretrained=False, num_classes=num_classes)
    elif args.arch == 'hg8':
        model = hg8(pretrained=False, num_classes=num_classes)
    else:
        raise Exception('unrecognised model architecture: ' + args.model)

    model = DataParallel(model).to(device)

    if args.optimizer == "Adam":
        optimizer = Adam(model.parameters(),
                         lr=args.lr,
                         momentum=args.momentum,
                         weight_decay=args.weight_decay)
    else:
        optimizer = RMSprop(model.parameters(),
                            lr=args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)
    best_acc = 0

    # optionally resume from a checkpoint
    title = args.data_identifier + ' ' + args.arch
    if args.resume:
        assert os.path.isfile(args.resume)
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Epoch', 'LR', 'Train Loss', 'Val Loss', 'Train Acc', 'Val Acc'])

    # train and eval
    lr = args.lr
    for epoch in range(args.start_epoch, args.epochs):
        lr = adjust_learning_rate(optimizer, epoch, lr, args.schedule, args.gamma)
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr))

        # train for one epoch
        train_loss, train_acc = do_training_epoch(train_loader, model, device, optimizer)

        # evaluate on validation set
        valid_loss, valid_acc, predictions = do_validation_epoch(val_loader, model, device, False)

        # append logger file
        logger.append([epoch + 1, lr, train_loss, valid_loss, train_acc, valid_acc])

        # remember best acc and save checkpoint
        is_best = valid_acc > best_acc
        best_acc = max(valid_acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, predictions, is_best, checkpoint=args.checkpoint, snapshot=args.snapshot)

    logger.close()
    logger.plot(['Train Acc', 'Val Acc'])
    savefig(os.path.join(args.checkpoint, 'log.eps'))


if __name__ == '__main__':
    # Check the hardware device to use for inference.
    if torch.cuda.is_available():
        try:
            cuda_env = os.environ['CUDA_VISIBLE_DEVICES']
        except KeyError as err:
            print('GPU environment variable CUDA_VISIBLE_DEVICES not set\n{0}'.format(err))
            print('Exiting')
            raise
        gpu_list = [int(x) for x in cuda_env.split(',')]
        if len(gpu_list) > 2:
            print('Warning: You are using more than 2 GPUs. Be Courteous.')
        print('Using ' + str(len(gpu_list)) + ' GPUs')
        print('GPUs: {0}'.format(gpu_list))
    else:
        print('GPU not available or display driver issue. Exiting')
        sys.exit()

    parser = argparse.ArgumentParser(description='Train a stacked hourglass model.')
    # Data identifier setting
    parser.add_argument('--data-identifier', default='', type=str,
                        help='type of image set being used')
    # Dataset setting
    parser.add_argument('-n', '--image-set-name', default='bigpaw', type=str,
                        help='images set name, (default: bigpaw)')
    parser.add_argument('--inp-res', '-i', default=256, type=int,
                        metavar='N',
                        help='resolution of input images (default: 256)')
    parser.add_argument('--out-res', '-o', default=64, type=int,
                        metavar='N',
                        help='resolution of output (default: 64)')

    # Model structure
    parser.add_argument('--arch', '-a', metavar='ARCH', default='hg',
                        choices=['hg1', 'hg2', 'hg8'],
                        help='model architecture')
    # Code version
    parser.add_argument('--mode', default='cv2', metavar='mode',
                        help='use cv2 or original code (default: cv2)')
    # Training strategy
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--train-batch', default=6, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--test-batch', default=6, type=int, metavar='N',
                        help='test batchsize')
    parser.add_argument('--optimizer', '--z', default='RMSprop', type=str,
                        metavar='Z', help='optimizer')
    parser.add_argument('--lr', '--learning-rate', default=2.5e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                        metavar='W', help='weight decay (default: 0)')
    parser.add_argument('--schedule', type=int, nargs='+', default=[60, 90],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='LR is multiplied by gamma on schedule.')
    # Miscs
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str,
                        metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--snapshot', default=0, type=int,
                        help='save models for every #snapshot epochs ' +
                        '(default: 0)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    main(parser.parse_args())
