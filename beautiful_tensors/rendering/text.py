import math
import uuid
import xml.etree.ElementTree as ET

import numpy as np

from beautiful_tensors.rendering.utils import rotate_pts


def get_square_text(
        height, width,
        center_text=None, bottom_text=None, left_text=None, top_text=None,
        center_text_scale=.2, bottom_text_scale=.2, side_text_scale=.1,
        padding=.5, rotate_left_text=True):
    text_padding = padding

    texts = []
    if center_text is not None:
        font_size = center_text_scale * height
        texts.append(Text(
            center_text, complex(width / 2, height / 2),
            font_size=font_size, tag='center_text'))
    if bottom_text is not None:
        font_size = bottom_text_scale * height
        pad = font_size / 2 + text_padding
        texts.append(Text(
            bottom_text, complex(width / 2, height + pad),
            font_size=font_size, tag='bottom_text'))
    if left_text is not None:
        font_size = side_text_scale * height
        pad = font_size / 2 + text_padding
        texts.append(Text(
            left_text, complex(-pad, height / 2),
            font_size=font_size, tag='left_text',
            rotation=-90 if rotate_left_text else 0))
    if top_text is not None:
        font_size = side_text_scale * height
        pad = font_size / 2 + text_padding
        texts.append(Text(
            top_text, complex(width / 2, -pad),
            font_size=font_size, tag='top_text'))
    
    return texts


def get_cube_text(
        height, width,
        center_text=None, bottom_text=None, left_text=None, top_text=None,
        center_text_scale=.2, bottom_text_scale=.2, side_text_scale=.1,
        padding=.5, rotate_left_text=True,
        depth_text=None, depth_pt=None, depth_normal=None, rotate_depth_text=True):
    texts = get_square_text(
        height, width,
        center_text=center_text, bottom_text=bottom_text, left_text=left_text, top_text=top_text,
        center_text_scale=center_text_scale, bottom_text_scale=bottom_text_scale, side_text_scale=side_text_scale,
        padding=padding, rotate_left_text=rotate_left_text)
    
    if depth_text is not None:
        font_size = side_text_scale * height
        pt = depth_pt - (depth_normal * (font_size / 2 + padding))

        axis = complex(1, 0)
        v1 = np.asarray([depth_normal.real, depth_normal.imag])
        v2 = np.asarray([axis.real, axis.imag])
        deg = -math.degrees(math.acos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))

        texts.append(Text(
            depth_text, pt,
            font_size=font_size, tag='depth_text',
            rotation=deg if rotate_depth_text else 0))
    return texts





class Text(object):
    def __init__(self, text, xy, font_size=12, rotation=0,
                 font_family='Caveat', font_weight='normal', font_style='normal',
                 dominant_baseline="middle", text_anchor="middle", tag=None):
        if not isinstance(xy, complex) and not isinstance(xy, np.complex128):
            xy = complex(*xy)
        self.id = str(uuid.uuid4())
        self.text = text
        self.xy = xy
        self.font_size = font_size
        self.rotation = rotation
        self.dominant_baseline = dominant_baseline
        self.text_anchor = text_anchor
        self.font_family = font_family
        self.font_weight = font_weight
        self.font_style = font_style
        self.tag = tag

    def rotate(self, deg, origin=None):
        if origin is None:
            origin = complex(*self.xy)
        self.xy = rotate_pts(
            np.asarray([self.xy.real, self.xy.imag]), deg=deg, origin=origin)[0]
        
        self.rotation = deg

    def translate(self, pt):
        self.xy += pt

    def to_xml(self, as_string=False, fill='#000000', opacity=1.):
        g = ET.Element('g')
        g.set('id', self.id)
        g.set('opacity', str(opacity))
        
        text = ET.SubElement(g, 'text')
        text.text = self.text
        d = {
            'x': self.xy.real,
            'y': self.xy.imag,
            'font-size': self.font_size,
            'transform': f'rotate({self.rotation})', # transform="rotate(-90)"
            'transform-origin': f'{self.xy.real} {self.xy.imag}',
            'font-family': self.font_family,
            'font-weight': self.font_weight,
            'font-style': self.font_style,
            'dominant-baseline': self.dominant_baseline,
            'text-anchor': self.text_anchor,
            'fill': fill,
        }
        for k, v in d.items():
            text.set(k, str(v))
        
        if as_string:
            return ET.tostring(g).decode('utf-8')

        return g        
        