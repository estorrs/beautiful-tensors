import uuid
import xml.etree.ElementTree as ET
from xml.dom import minidom

from beautiful_tensors.rendering.utils import flatten_paths


def get_xml_root_str():
    return f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'

def get_root_svg(width, height, scaled_width=None, scaled_height=None, unit='px'):
    scaled_width = scaled_width if scaled_width is not None else width
    scaled_height = scaled_height if scaled_height is not None else height
    svg =  ET.Element('svg')
    svg.set('xmlns', 'http://www.w3.org/2000/svg')
    svg.set('xmlns:ev', 'http://www.w3.org/2001/xml-events')
    svg.set('xmlns:xlink', 'http://www.w3.org/1999/xlink')
    svg.set('baseProfile', 'full')
    svg.set('width', f'{str(scaled_width)}{unit}')
    svg.set('height', f'{str(scaled_height)}{unit}')
    svg.set('version', '1.1')
    svg.set('viewBox', f'0.0 0.0 {width} {height}')
    return svg

def get_style():
    """    
    <svg xmlns="http://www.w3.org/2000/svg" width="400" height="150" font-size="24" text-anchor="middle">
        <style>
            @import url("https://fonts.googleapis.com/css?family=Roboto+Condensed:400,400i,700,700i");
            @import url("https://fonts.googleapis.com/css?family=Open+Sans:400,400i,700,700i");
            @import url("https://fonts.googleapis.com/css?family=Anonymous+Pro:400,400i,700,700i");
        </style>
        <text font-family="Roboto Condensed" x="190" y="32.92">
            This is a Roboto Condensed font
        </text>
        <text font-family="Open Sans" x="190" y="82.92">
            This is a Open Sans font
        </text>
        <text font-family="Anonymous Pro" x="190" y="132.92">
            This is a Anonymous Pro font
        </text>
    </svg>
    """
    style = ET.Element('style')
    style.text = f'@import url("https://fonts.googleapis.com/css?family=Caveat");'
    return style


class Canvas(object):
    def __init__(self, width, height, scale=None, description='Cosilico tensor canvas'):
        self.id = str(uuid.uuid4())
        self.width = width
        self.height = height

        self.scaled_width = int(width * scale) if scale is not None else int(width)
        self.scaled_height = int(height * scale) if scale is not None else int(height)

        self.svg =  get_root_svg(width, height, scaled_width=self.scaled_width, scaled_height=self.scaled_height)
        style = get_style()
        self.svg.append(style)
        title = ET.SubElement(self.svg, 'title')
        title.text = self.id
        desc = ET.SubElement(self.svg, 'desc')
        desc.text = description
        defs = ET.SubElement(self.svg, 'defs')

        self.shapes = ET.SubElement(self.svg, 'g')
        self.shapes.set('id', 'shapes')


        self.misc_paths = ET.SubElement(self.svg, 'g')
        self.misc_paths.set('id', 'misc_paths')
        self.misc_text = ET.SubElement(self.svg, 'g')
        self.misc_text.set('id', 'misc_text')

    def add_path(self, path, attbs):
        p = ET.SubElement(self.svg, 'path')
        for k, v in attbs.items():
            p.set(k, str(v))
        p.set('d', p.d())
        self.misc_paths.append(p)

    def add_text(self, text):
        self.misc_text.append(text.to_xml())

    def add_shape(self, shape, **kwargs):
        self.shapes.append(shape.to_xml(**kwargs))

    def to_xml(self, as_string=False):
        if as_string:
            # return ET.tostring(self.svg).decode('utf-8')
            return minidom.parseString(
                ET.tostring(self.svg, encoding='utf-8')).toprettyxml(indent="  ")
        return self.svg

    def write_svg(self, filepath):
        f = open(filepath, 'w')
        # f.write(get_xml_root_str() + '/')
        out = self.to_xml(as_string=True)
        f.write(out)
        f.close()
