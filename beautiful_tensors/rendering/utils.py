import math

import numpy as np
from IPython.display import SVG, display
from svgpathtools import wsvg, Line, Path, svg2paths2

def parse_paths_from_svg(filepath, scale=None):
    paths, attributes, _ = svg2paths2(filepath)
    if scale is not None:
        paths = [p.scaled(scale) for p in paths]
    return paths


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


def path_from_pts(xy, close=False):
    path = []
    for i in range(len(xy) - 1):
        path.append(Line(complex(*xy[i]), complex(*xy[i+1])))

    if close:
        path.append((Line(complex(*xy[-1]), complex(*xy[0]))))

    return Path(*path)


def rotate_pts(xy, deg, origin=None):
    lines = []
    for i in range(xy.shape[0] - 1):
        x1, y1 = xy[i]
        x2, y2 = xy[i + 1]
        line = Line(complex(x1, y1), complex(x2, y2))
        lines.append(line)
    lines = Path(*lines)
    lines = lines.rotated(deg, lines.point(0.) if origin is None else origin)
    
    coords = [[lines[0].start.real, lines[0].start.imag]]
    coords += [[l.end.real, l.end.imag] for l in lines]
    return np.asarray(coords)
