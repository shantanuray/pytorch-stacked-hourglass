"""Predict yogi data using stacked hourglass."""
import argparse
import os.path
import sys

import torch
import torch.backends.cudnn
from torch.nn import DataParallel
from torch.utils.data import DataLoader

from stacked_hourglass import hg1, hg2, hg8
from stacked_hourglass.datasets.mpii import print_mpii_validation_accuracy
from stacked_hourglass.datasets.generic import Generic
from stacked_hourglass.train import do_validation_epoch

from yogi import config
from data_handlers.yogi_db import YogiDB
from yogi.models import ImageSet


def main(args):
    """Predict for data source = YogiDB."""
    # Select the hardware device to use for inference.
    if torch.cuda.is_available():
        device = torch.device('cuda', torch.cuda.current_device())
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    # Disable gradient calculations.
    torch.set_grad_enabled(False)

    pretrained = not args.model_file

    if pretrained:
        print('No model weights file specified, exiting.')
        exit()

    # Get the data from yogi
    db_obj = YogiDB(config.db_url)
    imageset = db_obj.get_filtered(ImageSet,
                                   name=args.image_set_name)
    annotations_source = 'basic-thresholder'
    annotations = db_obj.get_annotations(annotation_source=annotations_source)
    pts = torch.Tensor(annotations[0]['joint_self'])
    num_classes = pts.size(0)
    # Initialise the Yogi validation set dataloader.
    val_dataset = Generic(image_set=imageset,
                          inp_res=args.inp_res,
                          out_res=args.out_res,
                          annotations=annotations,
                          mode=args.mode)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    # Create the model, downloading pretrained weights if necessary.
    if args.arch == 'hg1':
        model = hg1(pretrained=False, num_classes=num_classes)
    elif args.arch == 'hg2':
        model = hg2(pretrained=False, num_classes=num_classes)
    elif args.arch == 'hg8':
        model = hg8(pretrained=False, num_classes=num_classes)
    else:
        raise Exception('unrecognised model architecture: ' + args.model)
    model = model.to(device)

    if not pretrained:
        assert os.path.isfile(args.model_file)
        print('Loading model weights from file: {}'.format(args.model_file))
        checkpoint = torch.load(args.model_file)
        state_dict = checkpoint['state_dict']
        if sorted(state_dict.keys())[0].startswith('module.'):
            model = DataParallel(model)
        model.load_state_dict(state_dict)

    # Generate predictions for the validation set.
    _, _, predictions = do_validation_epoch(val_loader, model, device, args.flip)

    # Report PCKh for the predictions.
    print('\nFinal validation PCKh scores:\n')
    print_mpii_validation_accuracy(predictions)


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
    parser.add_argument('-n', '--image-set-name', default='bigpaw', type=str,
                        help='images set name, (default: bigpaw)')
    parser.add_argument('--arch', metavar='ARCH', default='hg1',
                        choices=['hg1', 'hg2', 'hg8'],
                        help='model architecture')
    parser.add_argument('--model-file', default='', type=str, metavar='PATH',
                        help='path to saved model weights')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('--batch-size', default=6, type=int, metavar='N',
                        help='batch size')
    parser.add_argument('--flip', dest='flip', action='store_true',
                        help='flip the input during validation')

    main(parser.parse_args())
