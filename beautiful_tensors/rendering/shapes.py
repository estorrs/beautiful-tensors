import numpy as np
from svgpathtools import Path, Line

from beautiful_tensors.rendering.strokes import ImageStroke
from beautiful_tensors.rendering.utils import rotate_pts

DEFAULT_STROKE_FP = '/data/estorrs/beautiful-tensors/data/sandbox/concepts/New Drawing 4 (2).png'

def make_rectangle(stroke, height, width):
    top, top_xy = stroke.sample(width)
    
    bottom, bottom_xy = stroke.sample(width)
    dist = height - (bottom_xy[:, 1].max() - bottom_xy[:, 1].min())
    bottom = bottom.translated(complex(0, dist))
    bottom_xy[:, 1] += dist
    
    left, left_xy = stroke.sample(height)
    left = left.rotated(90, left.point(0.))
    left_xy = rotate_pts(left_xy, 90)
    
    right, right_xy = stroke.sample(height)
    right = right.rotated(90, right.point(0.)).translated(complex(width, 0))
    right_xy = rotate_pts(right_xy, 90)
    right_xy[:, 0] += width
    
    xy = np.concatenate((top_xy, right_xy, bottom_xy, left_xy), axis=0)
    paths = [top, right, bottom, left]
    
    return paths, xy

class Rectangle(object):
    def __init__(self, height, width, top_left=(0, 0), stroke=None):
        self.r1, self.c1 = [int(x) for x in top_left]
        self.height = height
        self.width = width
        
        self.stroke = stroke if stroke is not None else ImageStroke(DEFAULT_STROKE_FP)
        
        self.paths, self.xy = make_rectangle(self.stroke, height, width)

    def copy(self):
        new = Rectangle(
            self.height, self.width, top_left=(self.r1, self.c1), stroke=self.stroke)
        new.paths = [p for p in self.paths]
        new.xy = self.xy.copy()
        return new

    def rotate(self, deg):
        origin = self.paths[0].point(0.)
        self.paths = [p.rotated(deg, origin) for p in self.paths]
        self.xy = rotate_pts(self.xy, deg, origin=origin)

    def translate(self, offset):
        self.paths = [p.translated(complex(offset[0], offset[1])) for p in self.paths]
        self.xy += np.asarray(offset)