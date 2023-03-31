import uuid
import xml.etree.ElementTree as ET
from xml.dom import minidom

from beautiful_tensors.rendering.utils import flatten_paths


def get_xml_root_str():
    return f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'

def get_root_svg(width, height, unit='px'):
    svg =  g = ET.Element('svg')
    svg.set('xmlns', 'http://www.w3.org/2000/svg')
    svg.set('xmlns:ev', 'http://www.w3.org/2001/xml-events')
    svg.set('xmlns:xlink', 'http://www.w3.org/1999/xlink')
    svg.set('baseProfile', 'full')
    svg.set('width', f'{str(width)}{unit}')
    svg.set('height', f'{str(height)}{unit}')
    svg.set('version', '1.1')
    svg.set('viewBox', f'0.0 0.0 {width} {height}')
    return svg

class Canvas(object):
    def __init__(self, width, height, description='Cosilico tensor canvas'):
        self.id = str(uuid.uuid4())
        self.width = width
        self.height = height

        self.svg =  get_root_svg(width, height)
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
