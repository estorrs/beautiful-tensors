import numpy as np
from svgpathtools import Path, Line

from beautiful_tensors.rendering.strokes import ImageStroke
from beautiful_tensors.rendering.fills import PathFill
from beautiful_tensors.rendering.utils import rotate_pts, path_from_pts

DEFAULT_STROKE_FP = '/data/estorrs/beautiful-tensors/data/sandbox/concepts/New Drawing 4 (2).png'
DEFAULT_FILL_FP = '/data/estorrs/beautiful-tensors/data/sandbox/concepts/New Drawing 5.svg'


def make_rectangle(stroke, height, width, stroke_width=1., deg=90):
    top, top_xy = stroke.sample(width, stroke_width=stroke_width)
    top = top.translated(complex(-stroke_width / 2, 0))
    top_xy += np.asarray([-stroke_width / 2, 0])

    left, left_xy = stroke.sample(height, stroke_width=stroke_width)
    left = left.rotated(deg + 180, left.point(0.))
    left_xy = rotate_pts(top_xy, deg + 180, left.point(0.))
    pt = top.point(0) - left.point(.5) + complex(stroke_width / 2, stroke_width / 2)
    left = left.translated(pt)
    left_xy += np.asarray([pt.real, pt.imag])
    
    right, right_xy = stroke.sample(height, stroke_width=stroke_width)
    right = right.rotated(deg, right.point(0))
    right_xy = rotate_pts(right_xy, deg, right.point(0))
    right = right.translated(complex(width - (stroke_width / 2), stroke_width / 4))
    right_xy += np.asarray([width - (stroke_width / 2), stroke_width / 4])
    
    bottom, bottom_xy = stroke.sample(width, stroke_width=stroke_width)
    bottom = bottom.rotated(180, bottom.point(0))
    bottom_xy = rotate_pts(bottom_xy, 180, bottom.point(0))

    pt = right.point(.5)
    bottom = bottom.translated(pt)
    bottom_xy += np.asarray([pt.real, pt.imag])
    bottom = bottom.translated(complex(0, stroke_width))
    bottom_xy += np.asarray([0, stroke_width])
    
    xy = np.concatenate((top_xy, right_xy, bottom_xy, left_xy), axis=0)
    paths = [top, right, bottom, left]
    
    return paths, xy


# def make_cube(stroke, height, width, stroke_width=1., deg=135):
#     (top_center, right_center, bottom_center, left_center), xy_center = make_rectangle(
#         stroke, height, width, stroke_width=1., deg=90)
#     (top_upper, right_upper, bottom_upper, left_upper), xy_upper = make_rectangle(
#         stroke, height, width, stroke_width=1., deg=deg)
#     pt = top_center.point(0) - left_upper.point(0)

#     (top_side, right_side, bottom_side, left_side), xy_side = make_rectangle(
#         stroke, height, width, stroke_width=1., deg=deg - 90)
    


class Shape(object):
    def __init__(self):
        self.c1, self.r1 = 0, 0
        self.stroke_paths = []
        self.stroke_xy = np.asarray([[]])
        self.boundary = None
        self.fill_paths = []
        self.stroke_width = 1

    def rotate(self, deg):
        origin = self.stroke_paths[0].point(0.)
        self.stroke_paths = [p.rotated(deg, origin) for p in self.stroke_paths]
        self.fill_paths = [p.rotated(deg, origin) for p in self.fill_paths]
        self.boundary = self.boundary.rotated(deg, origin)
        self.stroke_xy = rotate_pts(self.stroke_xy, deg, origin=origin)

    def translate(self, offset):
        self.stroke_paths = [p.translated(complex(offset[0], offset[1])) for p in self.stroke_paths]
        self.fill_paths = [p.translated(complex(offset[0], offset[1])) for p in self.fill_paths]
        self.boundary = self.boundary.translated(complex(offset[0], offset[1]))
        self.stroke_xy += np.asarray(offset)

    def to_renderable(self,
               fill_stroke_width=None, fill_color='#80a2bd',
               stroke_stroke_width=None, stroke_fill_color='#7e807f',
               stroke_stroke_color='#7e807f'):
        fill_stroke_width = fill_stroke_width if fill_stroke_width is not None else self.stroke_width / 4
        stroke_stroke_width = stroke_stroke_width if stroke_stroke_width is not None else self.stroke_width / 12

        paths, attbs = [], []
        paths += self.fill_paths
        attbs += [{
            'stroke': fill_color, 'stroke-width': fill_stroke_width, 'fill': 'none'
        }] * len(self.fill_paths)

        paths += self.stroke_paths
        attbs += [{
            'stroke': stroke_stroke_color, 'stroke-width': stroke_stroke_width, 'fill': stroke_fill_color
        }] * len(self.stroke_paths)

        return paths, attbs


class Rectangle(Shape):
    def __init__(self, height, width, top_left=(0, 0), deg=90,
                 fill='default', stroke='default', stroke_width=None):
        super().__init__()
        self.c1, self.r1 = [int(x) for x in top_left]
        self.height = height
        self.width = width
        self.stroke_width = stroke_width
        self.deg = deg
        
        self.stroke = stroke if stroke != 'default' else ImageStroke(DEFAULT_STROKE_FP)
        self.fill = fill if fill != 'default' else PathFill(DEFAULT_FILL_FP)
        
        self.stroke_paths, self.stroke_xy = make_rectangle(
            self.stroke, height, width, stroke_width=stroke_width, deg=deg)
        self.boundary = path_from_pts(self.stroke_xy, close=True)

        self.fill_paths = self.fill.random_cropped(self.boundary)

        self.translate((self.c1, self.r1))

    def copy(self):
        new = Rectangle(
            self.height, self.width, top_left=(self.r1, self.c1),
            stroke=self.stroke, stroke_width=self.stroke_width)
        new.stroke_paths = [p for p in self.stroke_paths]
        new.stroke_xy = self.stroke_xy.copy()
        new.boundary = self.boundary
        new.fill_paths = [p for p in self.fill_paths]
        return new


class Cube(Shape):
    def __init__(self, height, width, top_left=(0, 0), deg=90, deg_top=135,
                 fill='default', stroke='default', stroke_width=None):
        super().__init__()
        self.c1, self.r1 = [int(x) for x in top_left]
        self.height = height
        self.width = width
        self.stroke_width = stroke_width
        self.deg = deg
        
        self.stroke = stroke if stroke != 'default' else ImageStroke(DEFAULT_STROKE_FP)
        self.fill = fill if fill != 'default' else PathFill(DEFAULT_FILL_FP)

        (top_center, right_center, bottom_center, left_center), xy_center = make_rectangle(
            self.stroke, height, width, stroke_width=stroke_width, deg=deg)
        
        (top_upper, right_upper, bottom_upper, left_upper), xy_upper = make_rectangle(
            self.stroke, height, width, stroke_width=stroke_width, deg=deg_top)
        pt = top_center.point(0) - left_upper.point(0) + complex(0, stroke_width / 2)
        top_upper, right_upper, bottom_upper, left_upper = [
            p.translated(pt) for p in [top_upper, right_upper, bottom_upper, left_upper]]
        xy_upper += np.asarray([pt.real, pt.imag])
        top_upper = top_upper.translated(complex(0, -stroke_width / 2))

        (top_side, right_side, bottom_side, left_side), xy_side = make_rectangle(
            self.stroke, height, width, stroke_width=stroke_width, deg=deg_top - 90)
        top_side, right_side, bottom_side, left_side = [
            p.rotated(90, left_side.point(0))
            for p in [top_side, right_side, bottom_side, left_side]]
        xy_side = rotate_pts(xy_side, 90, left_side.point(0))
        pt = right_center.point(0) - left_side.point(0) + complex(-stroke_width / 2, 0)
        top_side, right_side, bottom_side, left_side = [
            p.translated(pt) for p in [top_side, right_side, bottom_side, left_side]]
        xy_side += np.asarray([pt.real, pt.imag])
        right_side = right_side.translated(complex(0, -stroke_width / 2))
        

        self.stroke_paths_center = [top_center, right_center, bottom_center, left_center]
        self.stroke_xy_center = xy_center
        self.boundary_center = path_from_pts(self.stroke_xy_center, close=True)
        self.fill_paths_center = self.fill.random_cropped(self.boundary_center)

        self.stroke_paths_upper = [top_upper, right_upper, left_upper]
        self.stroke_xy_upper = xy_upper
        self.boundary_upper = path_from_pts(self.stroke_xy_upper, close=True)
        self.fill_paths_upper = self.fill.random_cropped(self.boundary_upper)

        self.stroke_paths_side = [top_side, right_side]
        self.stroke_xy_side = xy_side
        self.boundary_side = path_from_pts(self.stroke_xy_side, close=True)
        self.fill_paths_side = self.fill.random_cropped(self.boundary_side)

        self.stroke_paths = [p for paths in [
            self.stroke_paths_center, self.stroke_paths_upper, self.stroke_paths_side]
                             for p in paths]
        self.stroke_xy = np.concatenate((xy_center, xy_upper, xy_side))
        self.boundary =  path_from_pts(self.stroke_xy, close=True)
        self.fill_paths = [p for paths in [
            self.fill_paths_center, self.fill_paths_upper, self.fill_paths_side]
                           for p in paths]

        self.translate((self.c1, self.r1))

    def copy(self):
        new = Rectangle(
            self.height, self.width, top_left=(self.r1, self.c1),
            stroke=self.stroke, stroke_width=self.stroke_width)
        new.stroke_paths = [p for p in self.stroke_paths]
        new.stroke_xy = self.stroke_xy.copy()
        new.boundary = self.boundary
        new.fill_paths = [p for p in self.fill_paths]
        return new
