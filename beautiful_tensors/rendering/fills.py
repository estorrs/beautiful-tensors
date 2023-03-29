import numpy as np
from scipy.spatial import ConvexHull
from svgpathtools import Line, Path

from beautiful_tensors.rendering.utils import parse_paths_from_svg, path_from_pts, show_svg


def extract_segments(obj, bounds):
    intersections = obj.intersect(bounds)
    locations = sorted([t[0][0] for t in intersections])

    segments = []
    for i in range(0, len(locations) - 1, 2):
        d1, d2 = locations[i], locations[i + 1]
        cropped = obj.cropped(d1, d2)
        segments.append(cropped)

    return segments

class PathFill(object):
    def __init__(self, path):
        if isinstance(path, str):
            self.path = parse_paths_from_svg(path, scale=.02)[0]
        else:
            self.path = path

    def cropped(self, bounds):
        if not isinstance(bounds, Path):
            bounds = path_from_pts(bounds)

        return extract_segments(self.path, bounds)

    def random_cropped(self, bounds, sample_size=1000):
        """
        A random crop for anywhere inside the fill path that is valid for the given bounds
        """
        if not isinstance(bounds, Path):
            bounds = path_from_pts(bounds, close=True)
        # center bounds to the origin
        x1, x2, y1, y2 = bounds.bbox()
        x, y = (x2 - x1) / 2, (y2 - y1) / 2
        bounds = bounds.translated(complex(-x - x1, -y - y1))

        path_pts = [self.path.point(t) for t in np.random.rand(sample_size)]
        path_xy = np.asarray([[pt.real, pt.imag] for pt in path_pts])
        hull = ConvexHull(path_xy)
        hull_xy = path_xy[hull.vertices]
        hull_xy = np.concatenate((hull_xy, hull_xy[:1])) # add closing line
        outer_bounds = path_from_pts(hull_xy)

        crop_bounds = None
        relocation_pt = None
        for pt in path_pts:
            relocated = bounds.translated(pt)
            if relocated.is_contained_by(outer_bounds):
                crop_bounds = relocated
                relocation_pt = pt
                break
        if crop_bounds is None:
            raise RuntimeError('no valid cropping windows found')

        segments = extract_segments(self.path, crop_bounds)
        # recenter
        segments = [s.translated(complex(x + x1, y + y1)) for s in segments]
        # relocation pt
        segments = [s.translated(-relocation_pt) for s in segments]
        return segments
