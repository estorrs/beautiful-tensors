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
    cap = get_half_circle(r, positive=positive)
    cap[:, 0] += np.asarray([min_h + r] * cap.shape[0])
    return cap


def convert_to_obj_pts(pts, start=None, stop=None, length=None, height=None, center=True):
    start = start if start is not None else 0
    stop = stop if stop is not None else pts.shape[0]
    size = stop - start

    xy = [[i - start, pts[i, 0]] for i in range(start, stop, 1)]
    cap = get_cap(pts[start:stop], positive=True)
    xy += [[x + size, y] for y, x in cap]

    xy += [[i - start, pts[i, 2]] for i in range(stop - 1, start - 1, -1)]
    cap = get_cap(pts[start:stop], positive=False)
    xy += [[x, y] for y, x in cap]

    xy = np.asarray(xy)

    w, h = xy.max(0) - xy.min(0)
    ratio = h / w
    if length is not None:
        xy[:, 0] /= w
        xy[:, 0] *= length

        xy[:, 1] /= h
        if height is None:
            xy[:, 1] *= ratio * length
        else:
            xy[:, 1] *= height
 
    if center:
        idxs = np.where(xy[:, 0] == 0)[0]
        assert len(idxs)==2
        v1, v2 = xy[idxs[0], 1],  xy[idxs[1], 1]
        y_center = abs((v1 - v2) / 2.) + min(v1, v2)
        xy[:, 1] -= y_center
  
    return xy


class ImageStroke(object):
    def __init__(self, filepath, background=254., step=100, window_size=201):
        self.img = read_line(filepath)
        self.background = background
        self.window_size = window_size
        self.step = step

        self.pts = coordinates_from_mask(
            self.img, step=step, window_size=window_size)

    def __get_obj_pts(self, start=None, stop=None, length=1., height=None, flip=False):
        return convert_to_obj_pts(
            self.pts if not flip else torch.flip(self.pts, dims=(0,)),
            start=start, stop=stop, length=length)
    
    def get_path(self, start=None, stop=None, length=1., height=None, flip=False):
        start, stop = int(start * self.pts.shape[0]), int(stop * self.pts.shape[0])
        xy = self.__get_obj_pts(start, stop, length=length, height=height, flip=flip)
        lines = [Line(complex(xy[i, 0], xy[i, 1]), complex(xy[i+1, 0], xy[i+1, 1]))
                 for i in range(len(xy) - 1)]
        line_obj = Path(*lines)

        return line_obj, xy

    def sample(self, length, height=None, distances=(.2, .5), allow_flip=True):
        distance = np.random.randint(
            int(distances[0] * 100), int(distances[1] * 100), 1)[0] / 100
        start_max = .9999 - distance
        start = np.random.randint(0, int(start_max * 100), 1)[0] / 100
        stop = start + distance
        flip = np.random.rand() < .5 if allow_flip else False
        return self.get_path(
            start=start, stop=stop, length=length, height=height, flip=flip)


        
        

