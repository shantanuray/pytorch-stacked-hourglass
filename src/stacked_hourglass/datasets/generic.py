"""Generic dataset."""
import random

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.utils.data as data
from importlib_resources import open_binary
from scipy.io import loadmat
from tabulate import tabulate
import cv2

import stacked_hourglass.res
from stacked_hourglass.utils.imutils import draw_labelmap
from stacked_hourglass.utils.misc import to_torch
from stacked_hourglass.utils.transforms import img_normalize, transform, cv2_crop

# import imgaug.augmenters as iaa
# from imgaug.augmentables import Keypoint, KeypointsOnImage


# TODO: Get joint names from data
# generic_JOINT_NAMES = [
#     'right_ankle', 'right_knee', 'right_hip', 'left_hip',
#     'left_knee', 'left_ankle', 'pelvis', 'spine',
#     'neck', 'head_top', 'right_wrist', 'right_elbow',
#     'right_shoulder', 'left_shoulder', 'left_elbow', 'left_wrist'
# ]


class Generic(data.Dataset):
    """Generic image set."""

    # Using the default RGB mean and std dev used for Mpii
    RGB_MEAN = torch.as_tensor([0.4404, 0.4440, 0.4327])
    RGB_STDDEV = torch.as_tensor([0.2458, 0.2410, 0.2468])

    def __init__(self, image_set, annotations,
                 is_train=True, inp_res=256, out_res=64, sigma=1,
                 scale_factor=0, rot_factor=0, label_type='Gaussian',
                 rgb_mean=RGB_MEAN, rgb_stddev=RGB_STDDEV):
        """Initialize object."""
        self.image_set = image_set  # Image set (array of images)
        self.anno = pd.DataFrame.from_dict(annotations)  # Annotations
        self.is_train = is_train  # training set or test set
        self.inp_res = inp_res
        self.out_res = out_res
        self.sigma = sigma
        self.scale_factor = scale_factor
        self.rot_factor = rot_factor
        self.label_type = label_type

        # create train/val split

        self.train_list, self.valid_list = train_test_split(self.anno,
                                                            test_size=0.2)
        self.mean = rgb_mean
        self.std = rgb_stddev

    def __getitem__(self, index):
        """Get an image referenced by index."""
        sf = self.scale_factor  # Generally from 0 to 0.25
        rf = self.rot_factor
        if self.is_train:
            a = self.train_list.iloc[index]
        else:
            a = self.valid_list.iloc[index]

        img_path = a['img_paths']
        pts = torch.Tensor(a['joint_self'])
        # pts[:, 0:2] -= 1  # Convert pts to zero based

        c = tuple(a['objpos'])
        # In Mpii, scale_provided is the dim of the boundary box wrt 200 px
        # We do not have a boundary box, so we assume a boundary box of inp_res
        s = 1

        # # Adjust scale slightly to avoid cropping limbs
        # if c[0] != -1:
        #     c[1] = c[1] + 15 * s
        #     s = s * 1.25

        # For pose estimation with a centered/scaled figure
        nparts = pts.size(0)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # HxWxC
        rows, cols = img.shape

        if self.is_train:
            # Given sf, choose scale from [1-sf, 1+sf]
            # For sf = 0.25, scale is chosen from [0.75, 1.25]
            s = torch.randn(1).mul_(sf).add_(1).clamp(1 - sf, 1 + sf)[0]
            # Given rf, choose scale from [-rf, rf]
            # For sf = 30, scale is chosen from [-30, 30]
            r = torch.randn(1).mul_(rf).clamp(-rf, rf)[0] if random.random() <= 0.6 else 0

            # Flip
            if random.random() <= 0.5:
                img = cv2.flip(img, 1)
                pts = [[rows - x[0] - 1, x[1]] for x in pts]

            # Color
            img[:, :, 0].mul_(random.uniform(0.8, 1.2)).clamp_(0, 255)
            img[:, :, 1].mul_(random.uniform(0.8, 1.2)).clamp_(0, 255)
            img[:, :, 2].mul_(random.uniform(0.8, 1.2)).clamp_(0, 255)

        # Rotate, scale and crop image
        inp, t = cv2_crop(img, c, s, (self.inp_res, self.inp_res), rot=r)
        # TODO Update color normalize
        inp = img_normalize(inp, self.mean, self.std)

        # Generate ground truth
        tpts = pts.clone()
        target = torch.zeros(nparts, self.out_res, self.out_res)
        target_weight = tpts[:, 2].clone().view(nparts, 1)

        for i in range(nparts):
            # if tpts[i, 2] > 0: # This is evil!!
            if tpts[i, 1] > 0:
                tpts[i, 0:2] = to_torch(transform(tpts[i, 0:2] + 1, c, s, [self.out_res, self.out_res], rot=r))
                target[i], vis = draw_labelmap(target[i], tpts[i] - 1, self.sigma, type=self.label_type)
                target_weight[i, 0] *= vis

        # Meta info
        meta = {'index': index, 'center': c, 'scale': s,
                'pts': pts, 'tpts': tpts, 'target_weight': target_weight}

        return inp, target, meta

    def __len__(self):
        """Number of images in dataset."""
        if self.is_train:
            return len(self.train_list)
        else:
            return len(self.valid_list)


def evaluate_generic_validation_accuracy(preds):
    threshold = 0.5
    sc_bias = 0.6

    dict = loadmat(open_binary(stacked_hourglass.res,
                               'detections_our_format.mat'))
    jnt_missing = dict['jnt_missing']
    pos_gt_src = dict['pos_gt_src']
    headboxes_src = dict['headboxes_src']

    preds = np.array(preds)
    assert preds.shape == (pos_gt_src.shape[2],
                           pos_gt_src.shape[0],
                           pos_gt_src.shape[1])
    pos_pred_src = np.transpose(preds, [1, 2, 0])

    jnt_visible = 1 - jnt_missing
    uv_error = pos_pred_src - pos_gt_src
    uv_err = np.linalg.norm(uv_error, axis=1)
    headsizes = headboxes_src[1, :, :] - headboxes_src[0, :, :]
    headsizes = np.linalg.norm(headsizes, axis=0)
    headsizes *= sc_bias
    scale = np.multiply(headsizes, np.ones((len(uv_err), 1)))
    scaled_uv_err = np.divide(uv_err, scale)
    scaled_uv_err = np.multiply(scaled_uv_err, jnt_visible)
    jnt_count = np.sum(jnt_visible, axis=1)
    less_than_threshold = np.multiply((scaled_uv_err < threshold), jnt_visible)
    PCKh = np.divide(100. * np.sum(less_than_threshold, axis=1), jnt_count)

    PCKh = np.ma.array(PCKh, mask=False)
    PCKh.mask[6:8] = True

    return PCKh


def print_generic_validation_accuracy(preds):
    PCKh = evaluate_generic_validation_accuracy(preds)

    head = generic_JOINT_NAMES.index('head_top')
    lsho = generic_JOINT_NAMES.index('left_shoulder')
    lelb = generic_JOINT_NAMES.index('left_elbow')
    lwri = generic_JOINT_NAMES.index('left_wrist')
    lhip = generic_JOINT_NAMES.index('left_hip')
    lkne = generic_JOINT_NAMES.index('left_knee')
    lank = generic_JOINT_NAMES.index('left_ankle')
    rsho = generic_JOINT_NAMES.index('right_shoulder')
    relb = generic_JOINT_NAMES.index('right_elbow')
    rwri = generic_JOINT_NAMES.index('right_wrist')
    rkne = generic_JOINT_NAMES.index('right_knee')
    rank = generic_JOINT_NAMES.index('right_ankle')
    rhip = generic_JOINT_NAMES.index('right_hip')

    print(tabulate([
        ['Head', 'Shoulder', 'Elbow', 'Wrist', 'Hip', 'Knee', 'Ankle', 'Mean'],
        [PCKh[head], 0.5 * (PCKh[lsho] + PCKh[rsho]), 0.5 * (PCKh[lelb] + PCKh[relb]),
        0.5 * (PCKh[lwri] + PCKh[rwri]), 0.5 * (PCKh[lhip] + PCKh[rhip]),
        0.5 * (PCKh[lkne] + PCKh[rkne]), 0.5 * (PCKh[lank] + PCKh[rank]), np.mean(PCKh)]
    ], headers='firstrow', floatfmt='0.2f'))
