import math

import numpy as np
from IPython.display import SVG, display
from svgpathtools import wsvg


def get_half_circle(r, n=100, positive=True):
    pi = math.pi
    s1, s2 = (n // 4), (n // 4) * 3
    result = np.asarray([[math.cos(2*pi/n*x)*r, math.sin(2*pi/n*x)*r] for x in range(s1, s2)])
    if positive:
        result[:, 0] *= -1.
        result = np.flip(result, axis=0)
    
    return result[:, [1, 0]]


def show_svg(paths, colors=None, attributes=None, filename='test.svg'):
    wsvg(paths, colors=colors, attributes=attributes, filename=filename)
    display(SVG(filename=filename))