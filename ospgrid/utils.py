#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 22:35:20 2022

@author: ccaprani
"""

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from typing import List
import numpy as np
from PIL import Image


def save_figs_to_file(
    filename: str,
    transparent: bool = False,
    figs: List[plt.Figure] = None,
    bbox: bool = False,
    pad: int = 20,
    std_names: bool = True,
):
    """
    Saves passed figs, or all open figs to PDF

    Parameters
    ----------
    filename : string
        The file to which the results are saved.
    transparent : bool, optional
        Whether or not the plots should be transparent. The default is False.
    figs : List[matplotlib.pyplot.Figure], optional
        A list of figure objects to save. Defaults to all open figures.
    bbox : bool, optional
        Whether or not to crop the figure to a bounding box of its contents. Only
        applies to image files, e.g., png, jpg, etc (not PDF)
    pad : int, optional
        If applying the bbox cropping to an image, a padding to apply to the
        contents. Defaults to 20 px.
    std_names : bool, optional
        If there are 5 figures open, and this is true, then meaningful strings
        (e.g. "_bmd") are added to the figure name in the order in which they
        are created in the function :meth:`grid.Grid.plot_results()`.
    """
    std_ends = [
        "mdl",
        "dsd",
        "bmd",
        "tmd",
        "sfd",
    ]
    use_std_ends = False

    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]

    if std_names and len(figs) == 5:
        use_std_ends = True

    ext = ".pdf"
    if "." in filename:
        stem, ext = filename.split(".")
        if ext == ".pdf":
            pp = PdfPages(filename)
            for fig in figs:
                fig.savefig(pp, format="pdf", transparent=transparent)

            pp.close()
        else:
            for i, fig in enumerate(figs):
                if use_std_ends:
                    f = stem + "_" + std_ends[i] + "." + ext
                else:
                    f = stem + f"_{i+1}" + "." + ext
                fig.savefig(f, format=ext, transparent=transparent, dpi=500)
                if bbox:
                    crop_to_bbox(f, pad)

    return


def crop_to_bbox(file: str, pad: int = 0):
    """
    Crops the image to the bounding box, but adds a padding. This function
    overwrites the original.
    """
    im = Image.open(file)
    left, top, right, btm = bbox(im)
    left -= pad
    top -= pad
    right += pad
    btm += pad
    im2 = im.crop((left, top, right, btm))
    im2.save(file)


def bbox(im: Image.Image):
    """
    Finds the bounding box of the contents in the image
    """
    a = np.array(im)[:, :, :3]  # keep RGB only
    m = np.any(a != [255, 255, 255], axis=2)
    coords = np.argwhere(m)
    y0, x0, y1, x1 = *np.min(coords, axis=0), *np.max(coords, axis=0)
    return (x0, y0, x1 + 1, y1 + 1)
