import beautiful_tensors.rendering.strokes as strokes
import beautiful_tensors.rendering.fills as fills

DEFAULT_STROKE_FP = '/data/estorrs/beautiful-tensors/data/sandbox/concepts/New Drawing 4 (2).png'
DEFAULT_FILL_FP = '/data/estorrs/beautiful-tensors/data/sandbox/concepts/New Drawing 5.svg'

DEFAULT_STROKE = strokes.ImageStroke(DEFAULT_STROKE_FP)
DEFAULT_FILL = fills.SolidFill()
# DEFAULT_FILL = fills.PathFill(DEFAULT_FILL_FP)


DEFAULT_THEME = {
    'stroke': {
        'stroke': DEFAULT_STROKE,
        'stroke_width': .2
    },
    'fill': {
        'fill': DEFAULT_FILL,
    },
    'rendering': {
        'fill': {
            'stroke_width': None,
            'stroke_color': '#80a2bd',
            'fill_color': 'none'
        },
        'stroke': {
            'stroke_width': None,
            'fill_color': '#7e807f',
            'stroke_color': '#7e807f'
        },
    },
    'text': {
        'major_font_size': 2,
        'minor_font_size': 1,
        'rotate_left_text': True,
        'rotate_depth_text': True
    },
}