import math

import numpy as np
from IPython.display import SVG, display
from svgpathtools import wsvg, Line, Path, svg2paths2, CubicBezier

def parse_paths_from_svg(filepath, scale=None):
    paths, attributes, _ = svg2paths2(filepath)
    if scale is not None:
        paths = [p.scaled(scale) for p in paths]
    return paths


def get_arc(r, n=100, start=180, stop=360):
    pi = math.pi
    x1, x2 = start / 360., stop / 360.
    # x1, x2 = .5, 1.
    s1, s2 = int(x1 * n), int(x2 * n)
    result = np.asarray([[math.cos(2*pi/n*x)*r, math.sin(2*pi/n*x)*r] for x in range(s1, s2)])

    return result


def get_bezier_arc(start, stop, center, as_pts=False, fidelity=100):
    """
    given the start and end points of the arc ([x1, y1] and [x4, y4], respectively) and the the center of the circle
    ([xc, yc]), one can derive the control points for a cubic Bézier curve ([x2, y2] and [x3, y3]) as follows:
    """
    if not isinstance(start, np.complex128) and not isinstance(start, complex):
        start, stop, center = complex(*start), complex(*stop), complex(*center)

    x1, y1 = start.real, start.imag
    x4, y4 = stop.real, stop.imag
    xc, yc = center.real, center.imag
    
    ax = x1 - xc
    ay = y1 - yc
    bx = x4 - xc
    by = y4 - yc
    q1 = ax * ax + ay * ay
    q2 = q1 + ax * bx + ay * by
    k2 = (4/3) * (np.sqrt(2 * q1 * q2) - q2) / (ax * by - ay * bx)

    x2 = xc + ax - k2 * ay
    y2 = yc + ay + k2 * ax
    x3 = xc + bx + k2 * by                                 
    y3 = yc + by - k2 * bx
    
    arc = CubicBezier(start, complex(x2, y2), complex(x3, y3), stop)

    if as_pts:
        return np.asarray([[arc.point(x).real, arc.point(x).imag] for x in np.linspace(0, 1, fidelity)])
    
    return arc

def get_bezier_half_circle(start, stop, center, positive=True, as_pts=False, fidelity=100):
    if not isinstance(start, np.complex128) and not isinstance(start, complex):
        start, stop, center = complex(*start), complex(*stop), complex(*center)

    delta = Line(start, center)
    mag = delta.length()
    direction = delta.normal(.5) * mag
    if not positive:
        direction = -direction
    
    mid = center + direction

    a1 = get_bezier_arc(start, mid, center)
    a2 = get_bezier_arc(mid, stop, center)

    arc = Path(a1, a2)

    if as_pts:
        return np.asarray([[arc.point(x).real, arc.point(x).imag] for x in np.linspace(0, 1, fidelity)])

    return arc
    



def get_half_circle(r, n=100, positive=True):
    pi = math.pi
    s1, s2 = (n // 4), (n // 4) * 3
    result = np.asarray([[math.cos(2*pi/n*x)*r, math.sin(2*pi/n*x)*r] for x in range(s1, s2)])
    if positive:
        result[:, 0] *= -1.
        result = np.flip(result, axis=0)
    
    return result


def show_svg(paths, colors=None, attributes=None, filename='test.svg'):
    wsvg(flatten_paths(paths), colors=colors, attributes=attributes, filename=filename)
    display(SVG(filename=filename))


def path_from_pts(xy, close=False):
    path = []
    for i in range(len(xy) - 1):
        path.append(Line(complex(*xy[i]), complex(*xy[i+1])))

    if close:
        path.append((Line(complex(*xy[-1]), complex(*xy[0]))))

    return Path(*path)


def rotate_pts(xy, deg, origin=None):
    lines = path_from_pts(xy)
    lines = Path(*lines)
    lines = lines.rotated(deg, lines.point(0.) if origin is None else origin)

    coords = [[lines[0].start.real, lines[0].start.imag]]
    coords += [[l.end.real, l.end.imag] for l in lines]
    coords = np.asarray(coords)
    return coords


def flatten_paths(nested_paths, flattened=None):
    """Definitely a more elegant way to do this, but it works for now."""
    if flattened is None:
        flattened = []
        initial = True
    else:
        initial = False

    if not isinstance(nested_paths, list):
        flattened.append(nested_paths)
    else:
        for p in nested_paths:
            flatten_paths(p, flattened=flattened)
        
    if initial:
        return flattened
    

def respace_pts(xy, n):
    path = path_from_pts(xy)
    return np.asarray([[path.point(x).real, path.point(x).imag]
                       for x in np.linspace(0, 1, n)])