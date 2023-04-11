import uuid
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib.pyplot as plt
from svgpathtools import Path, Line

from beautiful_tensors.rendering.strokes import ImageStroke, DEFAULT_DISTANCES
from beautiful_tensors.rendering.fills import PathFill
from beautiful_tensors.rendering.text import Text, get_square_text, get_cube_text
from beautiful_tensors.rendering.utils import rotate_pts, path_from_pts, flatten_paths, get_bezier_arc, show_svg

DEFAULT_STROKE_FP = '/data/estorrs/beautiful-tensors/data/sandbox/concepts/New Drawing 4 (2).png'
DEFAULT_FILL_FP = '/data/estorrs/beautiful-tensors/data/sandbox/concepts/New Drawing 5.svg'

DEFAULT_STROKE = ImageStroke(DEFAULT_STROKE_FP)
DEFAULT_FILL = PathFill(DEFAULT_FILL_FP)


def make_rectangle(stroke, height, width, stroke_width=1., deg=90):
    top, top_xy = stroke.sample(width, stroke_width=stroke_width)

    left, left_xy = stroke.sample(height, stroke_width=stroke_width)
    origin = complex(*left_xy[0])
    left = left.rotated(deg + 180, origin)
    left_xy = rotate_pts(left_xy, deg + 180, origin)
    
    right, right_xy = stroke.sample(height, stroke_width=stroke_width)
    origin = complex(*right_xy[0])
    right = right.rotated(deg, origin)
    right_xy = rotate_pts(right_xy, deg, origin)
    right = right.translated(complex(width, 0))
    right_xy += np.asarray([width, 0])
    
    bottom, bottom_xy = stroke.sample(width, stroke_width=stroke_width)
    origin = complex(*bottom_xy[0])
    bottom = bottom.rotated(180, origin)
    bottom_xy = rotate_pts(bottom_xy, 180, origin)

    pt = complex(*right_xy[-1])
    bottom = bottom.translated(pt)
    bottom_xy += np.asarray([pt.real, pt.imag])

    pt = complex(*bottom_xy[-1])
    left = left.translated(pt)
    left_xy += np.asarray([pt.real, pt.imag])
    
    xy = np.concatenate((top_xy, right_xy, bottom_xy, left_xy), axis=0)
    paths = [top, right, bottom, left]
    
    return paths, xy


def make_rounded_rectangle(stroke, height, width, stroke_width=1., radius=None,
                           top_left=complex(0, 0), fidelity=200):
    if radius is None:
        radius = height * .1
    
    top_start = top_left + complex(radius, 0)
    top_stop = top_left + complex(width - radius, 0)
    right_start = top_left + complex(width, radius)
    right_stop = top_left + complex(width, height - radius)
    bottom_start = top_left + complex(width - radius, height)
    bottom_stop = top_left + complex(radius, height)
    left_start = top_left + complex(0, height - radius)
    left_stop = top_left + complex(0, radius)
    
    tl_arc = get_bezier_arc(left_stop, top_start, top_left + complex(radius, radius))
    tr_arc = get_bezier_arc(top_stop, right_start, top_left + complex(width - radius, radius))
    br_arc = get_bezier_arc(right_stop, bottom_start, top_left + complex(width - radius, height - radius))
    bl_arc = get_bezier_arc(bottom_stop, left_start, top_left + complex(radius, height - radius))
    top = Line(top_start, top_stop)
    right = Line(right_start, right_stop)
    bottom = Line(bottom_start, bottom_stop)
    left = Line(left_start, left_stop)
    
    path = Path(tl_arc, top, tr_arc, right, br_arc, bottom, bl_arc, left)

    # unfortunately need to get rid of the beziers so downstream intersect operation dont break
    # find a way to fix this eventually
    path = path_from_pts([[path.point(x).real, path.point(x).imag]
                          for x in np.linspace(0, .999, fidelity)], close=False)

    rect_fit, rect_fit_xy = stroke.fit_to_path(path, stroke_width=stroke_width)
    
    return rect_fit, rect_fit_xy


def make_arrow(stroke, length, stroke_width=1., deg=90,
               head_scale=.1):
    """
    ----->
    """
    head_distances = (head_scale * DEFAULT_DISTANCES[0], head_scale * DEFAULT_DISTANCES[1])
    head_length = length * head_scale
    head_left, head_left_xy = stroke.sample(
        head_length, stroke_width=stroke_width, distances=head_distances)
    head_left = head_left.rotated(deg // 2, head_left.point(.5))
    head_left_xy = rotate_pts(head_left_xy, deg // 2, head_left.point(.5))

    # return [head_left], head_left_xy

    head_right, head_right_xy = stroke.sample(
        head_length, stroke_width=stroke_width, distances=head_distances)
    head_right = head_right.rotated(-deg // 2, head_right.point(.5))
    head_right_xy = rotate_pts(head_right_xy, -deg // 2, head_right.point(.5))
    head_right = head_right.translated(complex(stroke_width / 2, 0))
    head_right_xy += np.asarray([stroke_width / 2, 0])

    axis, axis_xy = stroke.sample(length, stroke_width=stroke_width)

    pt1 = axis.point(.5)
    pt2 = head_left.point(.5)
    delta = pt1 - pt2
    head_right = head_right.translated(delta)
    head_right_xy += np.asarray([delta.real, delta.imag])
    head_left = head_left.translated(delta)
    head_left_xy += np.asarray([delta.real, delta.imag])

    xy = np.concatenate((axis_xy, head_left_xy, head_right_xy), axis=0)
    paths = [axis, head_left, head_right]

    return paths, xy


class Shape(object):
    def __init__(self):
        self.id = str(uuid.uuid4())
        self.c1, self.r1 = 0, 0
        self.stroke_paths = []
        self.stroke_xy = np.asarray([[]])
        self.boundary = None
        self.fill_paths = []
        self.stroke_width = 1
        self.texts = []

    def rotate(self, deg):
        origin = self.stroke_paths[0].point(0.)
        self.stroke_paths = [p.rotated(deg, origin) for p in self.stroke_paths]
        self.fill_paths = [p.rotated(deg, origin) for p in self.fill_paths]
        self.boundary = self.boundary.rotated(deg, origin)
        self.stroke_xy = rotate_pts(self.stroke_xy, deg, origin=origin)

        for t in self.texts:
            t.rotate(deg, origin)

    def translate(self, offset):
        if isinstance(offset, complex) or isinstance(offset, np.complex128):
            offset = (offset.real, offset.imag)
        self.stroke_paths = [p.translated(complex(offset[0], offset[1])) for p in self.stroke_paths]
        self.fill_paths = [p.translated(complex(offset[0], offset[1])) for p in self.fill_paths]
        self.stroke_xy += np.asarray(offset)

        if self.boundary is not None:
            self.boundary = self.boundary.translated(complex(offset[0], offset[1]))

        for t in self.texts:
            t.translate(complex(offset[0], offset[1]))

    def to_renderable(self,
               fill_stroke_width=None, fill_stroke_color='#80a2bd',
               fill_fill_color='none',
               stroke_stroke_width=None, stroke_fill_color='#7e807f',
               stroke_stroke_color='#7e807f'):
        fill_stroke_width = fill_stroke_width if fill_stroke_width is not None else self.stroke_width / 4
        stroke_stroke_width = stroke_stroke_width if stroke_stroke_width is not None else self.stroke_width / 12

        paths, attbs = [], []
        paths += self.fill_paths
        attbs += [{
            'stroke': fill_stroke_color, 'stroke-width': fill_stroke_width, 'fill': fill_fill_color
        }] * len(self.fill_paths)

        paths += self.stroke_paths
        attbs += [{
            'stroke': stroke_stroke_color, 'stroke-width': stroke_stroke_width, 'fill': stroke_fill_color
        }] * len(self.stroke_paths)

        return paths, attbs
    
    def to_xml(self, as_string=False,
               fill_stroke_width=None, fill_stroke_color='#80a2bd',
               fill_fill_color='none',
               stroke_stroke_width=None, stroke_fill_color='#7e807f',
               stroke_stroke_color='#7e807f', opacity=1.):
        paths, attbs = self.to_renderable(
            fill_stroke_width=fill_stroke_width, fill_stroke_color=fill_stroke_color,
            fill_fill_color=fill_fill_color, stroke_stroke_width=stroke_stroke_width,
            stroke_fill_color=stroke_fill_color, stroke_stroke_color=stroke_stroke_color)
    
        g = ET.Element('g')
        g.set('id', self.id)
        g.set('opacity', str(opacity))
        
        for p, a in zip(paths, attbs):
            path = ET.SubElement(g, 'path')
            for k, v in a.items():
                path.set(k, str(v))
            path.set('d', p.d())

        for t in self.texts:
            child = t.to_xml()
            g.append(child)
        
        if as_string:
            return ET.tostring(g).decode('utf-8')

        return g        


class Rectangle(Shape):
    def __init__(self, height, width, top_left=(0, 0), deg=90,
                 fill='default', stroke='default', stroke_width=None,
                 center_text=None, left_text=None, top_text=None, bottom_text=None,
                 center_text_scale=.2, side_text_scale=.1, bottom_text_scale=.2,
                 rotate_left_text=True):
        super().__init__()
        self.c1, self.r1 = [int(x) for x in top_left]
        self.height = height
        self.width = width
        self.stroke_width = stroke_width
        self.deg = deg
        self.stroke = stroke if stroke != 'default' else DEFAULT_STROKE
        self.fill = fill if fill != 'default' else DEFAULT_FILL

        self.texts = get_square_text(
            height, width,
            center_text=center_text, bottom_text=bottom_text, left_text=left_text, top_text=top_text,
            center_text_scale=center_text_scale, bottom_text_scale=bottom_text_scale, side_text_scale=side_text_scale,
            padding=stroke_width, rotate_left_text=rotate_left_text)

        self.stroke_paths, self.stroke_xy = make_rectangle(
            self.stroke, height, width, stroke_width=stroke_width, deg=deg)
        self.boundary = path_from_pts(self.stroke_xy, close=True)

        self.fill_paths = flatten_paths(self.fill.sample(self.boundary))

        self.translate((self.c1, self.r1))
    

class RoundedRectangle(Shape):
    def __init__(self, height, width, top_left=(0, 0), radius=None,
                 fill='default', stroke='default', stroke_width=None,
                 center_text=None, left_text=None, top_text=None, bottom_text=None,
                 center_text_scale=.2, side_text_scale=.1, bottom_text_scale=.2,
                 rotate_left_text=True):
        super().__init__()
        self.c1, self.r1 = [int(x) for x in top_left]
        self.height = height
        self.width = width
        self.stroke_width = stroke_width
        self.radius = radius
        self.stroke = stroke if stroke != 'default' else DEFAULT_STROKE
        self.fill = fill if fill != 'default' else DEFAULT_FILL

        self.texts = get_square_text(
            height, width,
            center_text=center_text, bottom_text=bottom_text, left_text=left_text, top_text=top_text,
            center_text_scale=center_text_scale, bottom_text_scale=bottom_text_scale, side_text_scale=side_text_scale,
            padding=stroke_width, rotate_left_text=rotate_left_text)
        
        self.stroke_paths, self.stroke_xy = make_rounded_rectangle(
            self.stroke, height, width, stroke_width=stroke_width, radius=radius)
        self.stroke_paths = [self.stroke_paths]
        self.boundary = path_from_pts(self.stroke_xy, close=True)

        self.fill_paths = flatten_paths(self.fill.sample(self.boundary))

        self.translate((self.c1, self.r1))


class Cube(Shape):
    def __init__(self, height, width, depth, top_left=(0, 0), deg=90, deg_top=135,
                fill='default', stroke='default', stroke_width=None,
                 center_text=None, left_text=None, top_text=None, bottom_text=None, depth_text=None,
                 center_text_scale=.2, side_text_scale=.1, bottom_text_scale=.2,
                 rotate_left_text=True, rotate_depth_text=True):
        super().__init__()
        self.c1, self.r1 = [int(x) for x in top_left]
        self.height = height
        self.width = width
        self.stroke_width = stroke_width
        self.deg = deg
        self.depth = self.width if depth is None else depth
        
        self.stroke = stroke if stroke != 'default' else DEFAULT_STROKE
        self.fill = fill if fill != 'default' else DEFAULT_FILL

        (top_center, right_center, bottom_center, left_center), xy_center = make_rectangle(
            self.stroke, height, width, stroke_width=stroke_width, deg=deg)
        
        (top_upper, right_upper, bottom_upper, left_upper), xy_upper = make_rectangle(
            self.stroke, depth, width, stroke_width=stroke_width, deg=deg_top)
        pt = top_center.point(0) - left_upper.point(0)
        top_upper, right_upper, bottom_upper, left_upper = [
            p.translated(pt) for p in [top_upper, right_upper, bottom_upper, left_upper]]
        xy_upper += np.asarray([pt.real, pt.imag])

        (top_side, right_side, bottom_side, left_side), xy_side = make_rectangle(
            self.stroke, depth, height, stroke_width=stroke_width, deg=deg_top - 90)
        top_side, right_side, bottom_side, left_side = [
            p.rotated(90, left_side.point(0))
            for p in [top_side, right_side, bottom_side, left_side]]
        xy_side = rotate_pts(xy_side, 90, left_side.point(0))
        pt = right_center.point(0) - left_side.point(0) + complex(-stroke_width / 2, -stroke_width / 2)
        top_side, right_side, bottom_side, left_side = [
            p.translated(pt) for p in [top_side, right_side, bottom_side, left_side]]
        xy_side += np.asarray([pt.real, pt.imag])      

        self.stroke_paths_center = [top_center, right_center, bottom_center, left_center]
        self.stroke_xy_center = xy_center
        self.boundary_center = path_from_pts(self.stroke_xy_center, close=True)
        self.fill_paths_center = self.fill.sample(self.boundary_center)

        self.stroke_paths_upper = [top_upper, right_upper, left_upper]
        self.stroke_xy_upper = xy_upper
        self.boundary_upper = path_from_pts(self.stroke_xy_upper, close=True)
        self.fill_paths_upper = self.fill.sample(self.boundary_upper)

        self.stroke_paths_side = [top_side, right_side]
        self.stroke_xy_side = xy_side
        self.boundary_side = path_from_pts(self.stroke_xy_side, close=True)
        self.fill_paths_side = self.fill.sample(self.boundary_side)

        # get depth text location
        pt = left_upper.point(.25)
        depth_normal = -left_upper.normal(.25)
        self.texts = get_cube_text(
            height, width,
            center_text=center_text, bottom_text=bottom_text, left_text=left_text, top_text=top_text,
            center_text_scale=center_text_scale, bottom_text_scale=bottom_text_scale, side_text_scale=side_text_scale,
            padding=stroke_width, rotate_left_text=rotate_left_text,
            depth_text=depth_text, depth_pt=pt, depth_normal=depth_normal, rotate_depth_text=rotate_depth_text
            )


        self.stroke_paths = flatten_paths([
            self.stroke_paths_center, self.stroke_paths_upper, self.stroke_paths_side
        ])
        self.stroke_xy = np.concatenate((xy_center, xy_upper, xy_side))
        self.boundary =  path_from_pts(self.stroke_xy, close=True)
        self.fill_paths = flatten_paths([
            self.fill_paths_center, self.fill_paths_upper, self.fill_paths_side
        ])

        self.translate((self.c1, self.r1))

    def copy(self):
        pass


class Arrow(Shape):
    def __init__(self, length, top_left=(0, 0), deg=90,
                 stroke='default', stroke_width=None, head_scale=.1):
        super().__init__()
        self.c1, self.r1 = [int(x) for x in top_left]
        self.length = length
        self.stroke_width = stroke_width
        self.deg = deg
        self.stroke = stroke if stroke != 'default' else DEFAULT_STROKE
        self.head_scale = head_scale
        
        self.stroke_paths, self.stroke_xy = make_arrow(
            self.stroke, length, stroke_width=stroke_width, deg=deg, head_scale=head_scale)

        self.translate((self.c1, self.r1))

    def copy(self):
        pass