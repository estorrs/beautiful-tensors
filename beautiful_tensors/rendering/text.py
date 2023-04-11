import uuid
import xml.etree.ElementTree as ET

import numpy as np

from beautiful_tensors.rendering.utils import rotate_pts


class Text(object):
    def __init__(self, text, xy, font_size=12, rotation=0,
                 font_family='Helvetica', font_weight='normal', font_style='normal',
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
            'rotation': self.rotation,
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
        