import os
from math import factorial

import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
import torchvision.transforms.functional as TF
from scipy.signal import savgol_filter
from svgpathtools import svg2paths2, Path, polynomial2bezier, QuadraticBezier, CubicBezier, wsvg, Line
from torchvision.io.image import read_image

from beautiful_tensors.rendering.utils import get_half_circle


DEFAULT_STROKE_FP = '/data/estorrs/beautiful-tensors/data/sandbox/concepts/New Drawing 4 (2).png'
DEFAULT_DISTANCES = (.2, .5)

def read_line(fp, pad=20, background=254):
    img = read_image(fp)
    if img.shape[0] == 3:
        img = TF.rgb_to_grayscale(img)[0]
    rs, cs = torch.where(img<background)
    img = img[torch.min(rs) - pad:torch.max(rs) + pad, torch.min(cs):torch.max(cs)]
    img = img < background

    return img


def coordinates_from_mask(mask, step=100, pad=20, window_size=11):
    """
    mask - (h, w) # mask of line to extract pts from
    
    returns pts - (n, 3) # where n is number of steps and 3 is top, middle, and bottom y coords of line respectively
    """
    pts = [] # (top, middle, bottom)
    for i in range(pad, mask.shape[1] - pad, step):
        x = mask[:, i]
        idxs = torch.where(x)[0]
        min_idx, max_idx, median_idx = idxs.min(), idxs.max(), idxs.median()
        pts.append([min_idx, median_idx, max_idx])

    pts = torch.tensor(pts, dtype=torch.float32)
    # scale y by step
    pts /= step

    if window_size is not None:
        smoothed = torch.zeros_like(pts)
        for i in range(pts.shape[1]):
            smoothed[:, i] = torch.tensor(savgol_filter(pts[:, i], window_size, 3))
        pts = smoothed

    return pts


def get_cap(arr, positive=True):
    min_h, max_h = arr.min(0).values[0], arr.max(0).values[2]
    if positive:
        min_h = arr[-1, 0]
        max_h = arr[-1, 2]
    else:
        min_h = arr[0, 0]
        max_h = arr[0, 2]
    r = (max_h - min_h) / 2.
    print(r)
    cap = get_half_circle(r, positive=positive)
    cap[:, 0] += np.asarray([min_h + r] * cap.shape[0])
    return cap

def get_cap_simple(top_xy, bottom_xy, positive=True):
    if positive:
        top_pt = top_xy[-1]
        bottom_pt = bottom_xy[0]
    else:
        top_pt = top_xy[0]
        bottom_pt = bottom_xy[-1]
    min_h = top_pt[1]
    max_h = bottom_pt[1]
    r = (max_h - min_h) / 2.
    cap = get_half_circle(r, positive=positive)
    cap[:, 0] += top_pt[0]
    cap[:, 1] += min_h + r
    return cap

def rescale_x(xy, w, length):
    xy[:, 0] /= w
    xy[:, 0] *= length
    return xy

def rescale_y(xy, h, height):
    xy[:, 1] /= h
    xy[:, 1] *= height
    return xy


def convert_to_obj_pts(pts, start=None, stop=None, length=10., height=1., center=True):
    start = start if start is not None else 0
    stop = stop if stop is not None else pts.shape[0]
    size = stop - start

    top = np.asarray([[i - start, pts[i, 0]] for i in range(start, stop, 1)])
    midline = np.asarray([[i - start, pts[i, 1]] for i in range(start, stop, 1)])
    bottom = np.asarray([[i - start, pts[i, 2]] for i in range(stop - 1, start - 1, -1)])

    collected = np.concatenate((top, bottom), axis=0)
    w, h = collected.max(0) - collected.min(0)
    ratio = h / w
    height = ratio * length if height is None else height

    if length is not None:
        top, bottom, midline = [rescale_x(pts, w, length) for pts in [top, bottom, midline]]
        top, bottom, midline = [rescale_y(pts, h, height) for pts in [top, bottom, midline]]

    cap_right = np.asarray(get_cap_simple(top, bottom, positive=True))

    cap_left = np.asarray(get_cap_simple(top, bottom, positive=False))

    # plt.axis('equal')
    xy = np.concatenate((top, cap_right, bottom, cap_left), axis=0)
    # plt.scatter(xy[100:170, 0], xy[100:170, 1])
    if center:
        idxs = np.where(xy[:, 0] == 0)[0]
        assert len(idxs)==2
        v1, v2 = xy[idxs[0], 1],  xy[idxs[1], 1]
        y_center = abs((v1 - v2) / 2.) + min(v1, v2)
        xy[:, 1] -= y_center
        midline[:, 1] -= y_center
  
    return xy, midline


class ImageStroke(object):
    def __init__(self, filepath=None, background=254., step=100, window_size=51):
        if filepath is None:
            filepath = DEFAULT_STROKE_FP
        self.img = read_line(filepath)
        self.background = background
        self.window_size = window_size
        self.step = step

        self.pts = coordinates_from_mask(
            self.img, step=step, window_size=window_size)

    def __get_obj_pts(self, start=None, stop=None, length=1., height=None, flip=False):
        return convert_to_obj_pts(
            self.pts if not flip else torch.flip(self.pts, dims=(0,)),
            start=start, stop=stop, length=length, height=height)
    
    def get_path(self, start=None, stop=None, length=1., height=None, flip=False):
        start, stop = int(start * self.pts.shape[0]), int(stop * self.pts.shape[0])
        xy, midline = self.__get_obj_pts(start, stop, length=length, height=height, flip=flip)
        lines = [Line(complex(xy[i, 0], xy[i, 1]), complex(xy[i+1, 0], xy[i+1, 1]))
                 for i in range(len(xy) - 1)]
        line_obj = Path(*lines)

        return line_obj, midline

    def sample(self, length, stroke_width=None, distances=(.2, .5), allow_flip=True):
        distance = np.random.randint(
            int(distances[0] * 100), int(distances[1] * 100), 1)[0] / 100
        start_max = .9999 - distance
        start = np.random.randint(0, int(start_max * 100), 1)[0] / 100
        stop = start + distance
        flip = np.random.rand() < .5 if allow_flip else False
        return self.get_path(
            start=start, stop=stop, length=length, height=stroke_width, flip=flip)
