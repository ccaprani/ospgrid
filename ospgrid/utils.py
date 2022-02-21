#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 22:35:20 2022

@author: ccaprani
"""

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def save_figs(filename, figs=None):
    """
    Saves passed figs, or all open figs to PDF
    """

    pp = PdfPages(filename)

    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]

    for fig in figs:
        fig.savefig(pp, format="pdf")

    pp.close()
