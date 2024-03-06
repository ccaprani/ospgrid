#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 10:38:15 2022

@author: ccaprani
"""

from .grid import Grid


def make_grid(grid_str: str) -> Grid:
    r"""
    This function creates a grid object from a grid string specification. This
    functionality is particularly useful for the bulk generation of grids, and the
    grid specification is useful for the easy storage or communication of a grid.

    .. note:: The grid is taken as lying in the `x`-`y` plane.

    At a high level, the grid string specification has three parts, and looks as
    follows:

        ``"N_L_M"``

    where ``N`` is a string specifying the nodes; ``L`` specifies
    the nodal loads, and; ``M`` specifies the member connectivity. These parts are
    described next.

    The **node string** comprises a comma-separated list of sub-strings, each describing
    its node as follows:

        ``LSX:Y``

    in which ``L`` is the node identifying letter (e.g. "A"); ``S`` is the nodal condition
    which is one of "N" for no support, or one of the single character support descriptors
    acceptable by :py:meth:`~.Grid.add_support`; ``X`` and ``Y`` define the `(x,y)`
    coordinates of the node (note the separating colon ":").

    The **load string** comprises a comma-separated list of substrings, each describing
    the nodal loads as follows:

        ``LFx:Mx:My``

    in which ``L`` is the node identifier to which the loads are applied; ``Fx``, ``Mx``,
    and ``My`` are the nodal loads as per :py:meth:`~.Grid.add_load`. Note the
    separating colons between the nodal load components and that all 3 components
    must be defined (even if zero "0"). Nodes with no loads applied do not need to
    be included here.

    The **member connectivity** substring describes the members, and comprises a
    comma-separated list of two character nodal identifiers, the *i*- and *j*-node of
    each member following :py:meth:`~.Grid.add_member`. These characters must be
    from the list passed in the *node string*.

    For clarity, the hierarchy of separators is:

        - ``"_"`` separates the node, load, and member connectivity sub-strings;
        - ``","`` separates entries in the lists for each of the node, load, and member sub-strings;
        - ``":"`` separates coordinates or load components in a node or load definition.


    **Example**::

        import ospgrid as ospg

        grid_str = "AF-5:0,BX0:-5,CN0:0,DY2:0_C0:-50:-50_AC,CD,BC"
        grid = ospg.make_grid(grid_str)
        grid.analyze()
        grid.plot_results()

    This is a 4-node, 3-member grid. Node `A` is at `(-5,0)` and has a :attr:`~.Support.FIXED`
    support (``AF-5:0``); `B` is at `(0,-5)` and has a :attr:`~.Support.PINNED_X` support
    (``BX0:-5``); `C` is at the origin and is unsupported (``CN0:0``); while `D` is at
    `(2,0)` and has a :attr:`~.Support.PINNED_Y` support (``DY2:0``). The grid is
    loaded at node `C` with :math:`M_x = -50` kNm and :math:`M_y = -50` kNm (``C0:-10:-10``)
    and the members connect as follows: `AC`, `CD`, and `BC` (``AC,CD,BC``).

    .. note::
        This function assigns :math:`EI = 10\times10^3` kNm\ :sup:`2`, and :math:`GJ = 5\times10^3`
        kNm\ :sup:`2`. This should be reasonable for member lengths <10 m and loads of the order
        <10\ :sup:`2` kN or kNm.


    Parameters
    ----------
    grid_str : str
        The string describing the grid, formatted as described here.

    Raises
    ------
    ValueError
        If the string contains inconsistent information.

    Returns
    -------
    Grid
        The ospgrid object.

    """
    (nodes, loads, mbrs) = grid_str.split("_")

    grid = Grid()
    EI = 10e3  # kNm2
    GJ = 5e3

    node = nodes.split(",")
    for n in node:
        letter = n[0]
        cond = n[1]
        coords = n[2:].split(":")

        grid.add_node(letter, float(coords[0]), float(coords[1]))
        if cond != "N":  # N = None support
            grid.add_support(letter, cond)

    mbr = mbrs.split(",")
    for m in mbr:
        try:
            grid.add_member(m[0], m[1], EI, GJ)
        except Exception:
            raise ValueError(f"Node letters {m}, member definition error")

    load = loads.split(",")
    for ld in load:
        letter = ld[0]
        vals = ld[1:].split(":")

        if len(vals) != 3:
            raise ValueError("Insufficient nodal load values")

        grid.add_load(letter, Fz=float(vals[0]), Mx=float(vals[1]), My=float(vals[2]))

    return grid
