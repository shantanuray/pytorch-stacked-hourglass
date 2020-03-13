"""Evaluation for yogi dataset."""
import argparse
import os
import sys

import torch
import torch.backends.cudnn
from torch.nn import DataParallel
from torch.utils.data import DataLoader

from yogi import config
from data_handlers.yogi_db import YogiDB
from yogi.models import ImageSet
from stacked_hourglass import hg1, hg2, hg8
from stacked_hourglass.datasets.generic import Generic
from stacked_hourglass.train import do_validation_epoch
from stacked_hourglass.utils.logger import Logger

from datetime import datetime


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
    crop_size = 512
    if args.crop:
        crop_size = args.crop
        crop = True
    else:
        crop = False

    # Using the default RGB mean and std dev as 0
    RGB_MEAN = torch.as_tensor([0.0, 0.0, 0.0])
    RGB_STDDEV = torch.as_tensor([0.0, 0.0, 0.0])

    dataset = Generic(image_set=imageset,
                      inp_res=args.inp_res,
                      out_res=args.out_res,
                      annotations=annotations,
                      mode=args.mode,
                      crop=crop, crop_size=crop_size,
                      rgb_mean=RGB_MEAN, rgb_stddev=RGB_STDDEV)

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

    # create debug_loc dir
    os.makedirs(args.debug_loc, exist_ok=True)

    if args.arch == 'hg1':
        model = hg1(pretrained=False, num_classes=num_classes)
    elif args.arch == 'hg2':
        model = hg2(pretrained=False, num_classes=num_classes)
    elif args.arch == 'hg8':
        model = hg8(pretrained=False, num_classes=num_classes)
    else:
        raise Exception('unrecognised model architecture: ' + args.model)

    model = DataParallel(model).to(device)

    # Load model from a debug_loc
    title = ' '.join(['evaluate',
                     args.data_identifier,
                     args.arch,
                     args.image_set_name,
                     "{0:%F}".format(datetime.now())])

    assert os.path.isfile(args.checkpoint)
    print("=> loading model from '{}'".format(args.checkpoint))
    checkpoint = torch.load(args.checkpoint)
    args.start_epoch = 1
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded model from '{}' ".format(args.checkpoint))
    logger = Logger(os.path.join(args.debug_loc, 'log.txt'), title=title)
    logger.set_names(['Epoch', 'Val Loss', 'Val Acc'])

    # eval
    for epoch in range(args.start_epoch, args.epochs):
        # evaluate on validation set
        valid_loss, valid_acc, predictions, validation_log = do_validation_epoch(val_loader, model, device, False, True, os.path.join(args.debug_loc, title + '.csv'), epoch)
        # append logger file
        logger.append([epoch, valid_loss, valid_acc])

    logger.close()


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

    parser = argparse.ArgumentParser(description='Evaluate a stacked hourglass model.')
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
    parser.add_argument('--crop', default=0, type=int,
                        metavar='N',
                        help='crop size of input image (default: 0 => No cropping)')

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
    parser.add_argument('--test-batch', default=6, type=int, metavar='N',
                        help='test batchsize')
    parser.add_argument('--optimizer', '--z', default='RMSprop', type=str,
                        metavar='Z', help='optimizer')
    # Miscs
    parser.add_argument('-c', '--debug_loc', default='debug_loc', type=str,
                        metavar='PATH',
                        help='path to save debug_loc (default: debug_loc)')
    parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--debug', default='0', type=int,
                        help='debug mode (0,1) (default: 0)')

    main(parser.parse_args())
