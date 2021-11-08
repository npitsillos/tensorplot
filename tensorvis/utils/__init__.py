from tensorvis.utils.utils import draw_line, draw_scatter, separate_exps, update_layout

DRAW_FN_MAP = {"line": draw_line, "scatter": draw_scatter}

__all__ = [
    "separate_exps",
    "update_layout",
]
