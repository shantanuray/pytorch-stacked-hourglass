import numpy as np
import torch
import cv2

from .imutils import im_to_numpy, im_to_torch
from .misc import to_torch
from .pilutil import imresize, imrotate


def color_normalize(x, mean, std):
    if x.size(0) == 1:
        x = x.repeat(3, 1, 1)

    for t, m, s in zip(x, mean, std):
        t.sub_(m)
    return x


def img_normalize(img, mean, std):
    """Transform the image for torch and normalize."""
    x = im_to_torch(img)
    x = color_normalize(x, mean, std)
    return x


def flip_back(flip_output, dataset='mpii'):
    """
    flip output map
    """
    if dataset ==  'mpii':
        matchedParts = (
            [0,5],   [1,4],   [2,3],
            [10,15], [11,14], [12,13]
        )
    else:
        raise ValueError('Not supported dataset: ' + dataset)

    # flip output horizontally
    flip_output = fliplr(flip_output.numpy())

    # Change left-right parts
    for pair in matchedParts:
        tmp = np.copy(flip_output[:, pair[0], :, :])
        flip_output[:, pair[0], :, :] = flip_output[:, pair[1], :, :]
        flip_output[:, pair[1], :, :] = tmp

    return torch.from_numpy(flip_output).float()


def shufflelr(x, width, dataset='mpii'):
    """
    flip coords
    """
    if dataset ==  'mpii':
        matchedParts = (
            [0,5],   [1,4],   [2,3],
            [10,15], [11,14], [12,13]
        )
    else:
        raise ValueError('Not supported dataset: ' + dataset)

    # Flip horizontal
    x[:, 0] = width - x[:, 0]

    # Change left-right parts
    for pair in matchedParts:
        tmp = x[pair[0], :].clone()
        x[pair[0], :] = x[pair[1], :]
        x[pair[1], :] = tmp

    return x


def fliplr(x):
    if x.ndim == 3:
        x = np.transpose(np.fliplr(np.transpose(x, (0, 2, 1))), (0, 2, 1))
    elif x.ndim == 4:
        for i in range(x.shape[0]):
            x[i] = np.transpose(np.fliplr(np.transpose(x[i], (0, 2, 1))), (0, 2, 1))
    return x.astype(float)


def combine_transformations(t_combined=None, *args):
    """Combine transformation matrices.

    For each matrix:
    1. Convert from 2x3 to 3x3 by appending row [[0,0,1]]
    2. dot product of all matrices
    3. Return 2x3 matrix of dot product

    Used tail recursion below
    """
    if len(args) > 0:
        args = list(args)
        heads = args.pop(0)
        if t_combined is None:
            t_combined = heads
        else:
            t_heads = np.append(heads, [[0, 0, 1]], axis=0)
            t_combined = np.append(t_combined, [[0, 0, 1]], axis=0)
            t_combined = np.dot(t_combined, t_heads)
            t_combined = t_combined[0:2, 0:3]
        t_combined = combine_transformations(t_combined, *args)
    return t_combined


def cv2_resize(img, res):
    """Resize and return image as well as transformation matrix."""
    rows, cols, _ = img.shape
    # Resize the image to res
    img = cv2.resize(img, res)
    # Create the transformation matrix for resize
    # Resize only affects [0,0] * [1,1]
    t_resize = np.zeros((2, 3))
    t_resize[0, 0] = res[0] / rows
    t_resize[1, 1] = res[1] / cols
    return img, t_resize


def cv2_crop(img, center, scale, res, rot=0, crop=False, crop_size=512):
    """Scale, rotate and crop wrt to objpos using cv2."""
    rows, cols, _ = img.shape
    # To keep consistency with original code written for Mpii
    obj_bbox_size = int(200 * scale)
    # Scale the image so that a scaled obj_bbox_size fits inside res
    w = int(cols * res[1] / obj_bbox_size)
    h = int(rows * res[0] / obj_bbox_size)
    img, t_bbox_res = cv2_resize(img, (w, h))
    # Get new shape
    rows, cols, _ = img.shape
    # Get new center position
    center = tuple(transform(torch.tensor(center), t=t_bbox_res))
    # Rotate around center
    t = cv2.getRotationMatrix2D(center, rot, 1)

    if crop is True:
        # Translate the image to the center
        t[0, 2] += cols / 2 - center[0]
        t[1, 2] += rows / 2 - center[1]
        # Actually rotate the image and translate
        img = cv2.warpAffine(img, t, (rows, cols))
        # To include the entire rotated box in the cropped image,
        # compute a box that will contain the rotated box of size res
        # For example, to include a 256x256 that has been rotated 30 degrees,
        # one will need a 349x349 box
        abs_cos = abs(t[0, 0])
        abs_sin = abs(t[0, 1])
        # find the new width and height bounds
        crop_boundary_w = int(res[0] * abs_sin + res[1] * abs_cos)
        crop_boundary_h = int(res[0] * abs_cos + res[1] * abs_sin)
        # Crop image around 'center' bounded by new box bounds
        # Max is size of the image
        crop_x_bounds = torch.Tensor([-1, 1]).mul_(crop_boundary_w / 2).add_(cols / 2).clamp(0, cols)
        crop_y_bounds = torch.Tensor([-1, 1]).mul_(crop_boundary_h / 2).add_(rows / 2).clamp(0, rows)
        img = img[int(crop_x_bounds[0]):int(crop_x_bounds[1]),
                  int(crop_y_bounds[0]):int(crop_y_bounds[1])]
        # Resize the image to res
        img, t_resize = cv2_resize(img, res)
        t = combine_transformations(t, t_resize)
    else:
        # Rotate the image
        img = cv2.warpAffine(img, t, (rows, cols))

    # Combine the transformation matrices
    t_combined = combine_transformations(t_bbox_res, t)
    return img, t_combined



def get_transform(center, scale, res, rot=0):
    """
    General image processing functions
    """
    # Generate transformation matrix
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot # To match direction of rotation from cropping
        rot_mat = np.zeros((3,3))
        rot_rad = rot * np.pi / 180
        sn,cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0,:2] = [cs, -sn]
        rot_mat[1,:2] = [sn, cs]
        rot_mat[2,2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0,2] = -res[1]/2
        t_mat[1,2] = -res[0]/2
        t_inv = t_mat.copy()
        t_inv[:2,2] *= -1
        t = np.dot(t_inv,np.dot(rot_mat,np.dot(t_mat,t)))
    return t


def transform(pt, center=None, scale=None, res=None, invert=0, rot=0, t=None):
    """Transform pixel location to different reference."""
    if t is None:
        t = get_transform(center, scale, res, rot=rot)
        if invert:
            t = np.linalg.inv(t)
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int)


def transform_preds(coords, center,
                    scale,
                    out_res, inp_res,
                    rot):
    """Transform predicted coordinates to original image specifications."""
    # Old code
    # This was commented by original author
    # size = coords.size()
    # coords = coords.view(-1, coords.size(-1))
    # print(coords.size())

    # Generic dataset does not assume scale_provided wrt 200 px
    # Code below replaces scale_provided wrt 200 px with inp_res
    if inp_res is not None:
        scale = scale * inp_res / 200
    for p in range(coords.size(0)):
        # TODO: How come no rotation back and why invert?

        coords[p, 0:2] = to_torch(transform(coords[p, 0:2], center,
                                            scale,
                                            out_res, 1, 0))
    # TODO: New code - Switch to using cv2 transformations below
        # t_rot = cv2.getRotationMatrix2D(tuple(coords), -rot, 1)
        # t_resize = np.zeros((2, 3))
        # t_resize[0, 0] = inp_res / out_res
        # t_resize[1, 1] = inp_res / out_res

        # t_translate = np.diag((1, 1, 1))
        # t_translate[0, 2] = -inp_res / 2 + int(center[0])
        # t_translate[1, 2] = -inp_res / 2 + int(center[1])
        # t_translate = t_translate[0:2, 0:3]
        # # Resize and "un"-rotate
        # coords[p, 0:2] = transform(coords[p, 0:2],
        #                            t=combine_transformations(t_resize, t_rot))
        # # Translate to original coordinate system
        # coords[p, 0:2] = transform(coords[p, 0:2],
        #                            t=t_translate)
    return coords



def crop(img, center, scale, res, rot=0):
    img = im_to_numpy(img)

    # Preprocessing for efficient cropping
    ht, wd = img.shape[0], img.shape[1]
    sf = scale * 200.0 / res[0]
    if sf < 2:
        sf = 1
    else:
        new_size = int(np.math.floor(max(ht, wd) / sf))
        new_ht = int(np.math.floor(ht / sf))
        new_wd = int(np.math.floor(wd / sf))
        if new_size < 2:
            return torch.zeros(res[0], res[1], img.shape[2]) \
                        if len(img.shape) > 2 else torch.zeros(res[0], res[1])
        else:
            img = imresize(img, [new_ht, new_wd])
            center = center * 1.0 / sf
            scale = scale / sf

    # Upper left point
    ul = np.array(transform([0, 0], center, scale, res, invert=1))
    # Bottom right point
    br = np.array(transform(res, center, scale, res, invert=1))

    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
    if not rot == 0:
        ul -= pad
        br += pad

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(img.shape[1], br[0])
    old_y = max(0, ul[1]), min(img.shape[0], br[1])
    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]

    if not rot == 0:
        # Remove padding
        new_img = imrotate(new_img, rot)
        new_img = new_img[pad:-pad, pad:-pad]

    new_img = im_to_torch(imresize(new_img, res))
    return new_img
