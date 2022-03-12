#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 20:25:38 2022

@author: ccaprani
"""

from enum import Enum
import itertools
from typing import Union, Tuple, List
import matplotlib.pyplot as plt
import openseespy.opensees as osp
import opsvis as ospv
import numpy as np
from .utils import save_figs


class Support(Enum):
    """
    Enumerate the possible support types, with definitions correct for OpenSeesPy
    """

    # DX DY DZ RX RY RZ
    PINNED_X = [1, 1, 1, 0, 1, 1]  # rotates about x axis
    PINNED_Y = [1, 1, 1, 1, 0, 1]  # rotates about y axis
    PROP = [1, 1, 1, 0, 0, 1]  # rotates about both
    FIXED = [1, 1, 1, 1, 1, 1]  # fixed
    FIXED_V_ROLLER = [1, 1, 0, 1, 1, 1]  # fixed but vertical roller


class Node:
    """
    Object representing a node of the grid
    """

    def __init__(self, idx: int, label: str, x: float, y: float):
        """
        Initialize the node

        Parameters
        ----------
        idx : int
            The index of the node.
        label : str
            A user-friendly label for the node.
        x : float
            The x-axis coordinate of the node.
        y : float
            The y-axis coordinate of the node.

        Returns
        -------
        None.

        """
        self.idx = idx
        self.label = label
        self.x = x
        self.y = y
        self.Fz = 0
        self.Mx = 0
        self.My = 0
        self.support = None

    def set_load(self, Fz: float = 0, Mx: float = 0, My: float = 0):
        """
        Sets the externally-applied load applied to the node.

        Parameters
        ----------
        Fz : float, optional
            Vertical load. The default is 0.
        Mx : float, optional
            Moment about the x-axis. The default is 0.
        My : float, optional
             Moment about the y-axis. The default is 0.

        Returns
        -------
        None.

        """
        self.Fz = Fz
        self.Mx = Mx
        self.My = My

    def set_support(self, support: Support):
        """
        Sets the support type for a node

        Parameters
        ----------
        support : Support
            The support type, a :class:`ospgrid.grid.Support` object

        Returns
        -------
        None.

        """
        self.support = support


class Member:
    """
    Object encapsulating the proeprties for a grid member
    """

    def __init__(self, idx: int, node_i: Node, node_j: Node, EI: float, GJ: float):
        """
        Initialize the member object

        Parameters
        ----------
        idx : int
            The member index in the grid.
        node_i : Node
            The start node of the member.
        node_j : Node
            The end node of the member.
        EI : float
            The flexural rigitidy.
        GJ : float
            The torsional rigidity.

        Returns
        -------
        None.

        """
        self.idx = idx
        self.node_i = node_i
        self.node_j = node_j
        self.EI = EI
        self.GJ = GJ
        self.delta_x = self.node_j.x - self.node_i.x
        self.delta_y = self.node_j.y - self.node_i.y
        self.L = np.sqrt(self.delta_x ** 2 + self.delta_y ** 2)

    def get_local_stiffness(self) -> np.ndarray:
        """
        Returns the member stiffness matrix in local coordinates with nodal DOFs
        in the order DZ ('vertical' force), RX (torsion), RZ (bending moment).

        Returns
        -------
        K : np.ndarray
            Member stiffness matrix in local coordinates.

        """

        k11 = 12 * self.EI / self.L ** 3
        k13 = 6 * self.EI / self.L ** 2
        k22 = self.GJ / self.L
        k33 = 4 * self.EI / self.L
        k36 = 2 * self.EI / self.L

        K = np.array(
            [
                [k11, 0, k13, -k11, 0, k13],
                [0, k22, 0, 0, -k22, 0],
                [k13, 0, k33, -k13, 0, k36],
                [-k11, 0, -k13, k11, 0, -k13],
                [0, -k22, 0, 0, k22, 0],
                [k13, 0, k36, -k13, 0, k33],
            ]
        )
        return K

    def get_transformation_matrix(self) -> np.ndarray:
        """
        Gives the transformation matrix relating the member dgrees of freedom from
        local to global coordinate system.

        It considers only the grid member DOFs, per :func:`get_local_stiffness`

        Returns
        -------
        T : np.ndarray
            Tranformation matrix

        """
        c = self.delta_x / self.L
        s = self.delta_y / self.L

        T = np.kron(np.eye(2, dtype=int), np.array([[1, 0, 0], [0, c, s], [0, -s, c]]))
        return T

    def get_global_stiffness(self) -> np.ndarray:
        """
        Returns the member stiffness matrix in global coordinates with nodal DOFs
        in the order DZ ('vertical' force), RX (torsion), RZ (bending moment). This
        shows the stiffness
        contributions of the member to the global DOFs.

        Returns
        -------
        Kg : np.ndarray
            Member stiffness matrix in global coordinates.

        """
        K = self.get_local_stiffness()
        T = self.get_transformation_matrix()
        Kg = T.T @ K @ T
        return Kg


class Grid:
    """
    A class that provides a user-friendly interface to OpenSeesPy for the analysis
    of plane elastic grids.
    """

    FIGSIZE = (6.0, 6.0)

    def __init__(self):
        """
        Initialize the grid

        Returns
        -------
        None.

        """
        self.clear()
        pass

    def clear(self):
        """
        Clears any nodes or members from the grid.

        Returns
        -------
        None.

        """
        self.nodes = []
        self.members = []
        self.no_nodes = 0
        self.no_members = 0

    def add_node(self, label: str, x: float, y: float):
        """
        Adds a node to the grid

        Parameters
        ----------
        label : str
            A user-friendly label for the node, e.g. "A".
        x : float
            The x-axis coordinate of the node.
        y : float
            The y-axis coordinate of the node.

        Returns
        -------
        node : TYPE
            DESCRIPTION.

        """
        self.no_nodes += 1
        node = Node(self.no_nodes, label, x, y)
        self.nodes.append(node)
        return node

    def add_member(
        self,
        node_i: Union[Node, str, int],
        node_j: Union[Node, str, int],
        EI: float,
        GJ: float,
    ):
        """
        Adds a member to the grid.

        Parameters
        ----------
        node_i : Union[Node,str,int]
            The starting node for the member.
        node_j : Union[Node,str,int]
            The ending node for the member.
        EI : float
            The flexural rigidity.
        GJ : float
            The torsional rigidity.

        Returns
        -------
        member : Member
            The member object instance.

        """
        self.no_members += 1
        the_node_i = self.get_node(node_i)
        the_node_j = self.get_node(node_j)
        member = Member(self.no_members, the_node_i, the_node_j, EI, GJ)
        self.members.append(member)
        return member

    def add_load(
        self,
        node: Union[Node, str, int],
        Fz: float = 0,
        Mx: float = 0,
        My: float = 0,
    ):
        """
        Add a load to the grid.

        Parameters
        ----------
        node : Union[Node,str,int]
            Node object, label, or id
        Fz : float, optional
            Vertical load. The default is 0.
        Mx : float, optional
            Moment about the x-axis. The default is 0.
        My : float, optional
             Moment about the y-axis. The default is 0.

        Raises
        ------
        ValueError
            If node object, label, or id not passed, or is multiple nodes match the
            label, or no node found.

        Returns
        -------
        None.

        """
        the_node = self.get_node(node)
        the_node.set_load(Fz, Mx, My)

    def add_support(
        self, node: Union[Node, str, int], support: Union[Support, str] = None
    ):
        """
        Add a support to a node in the grid.

        Parameters
        ----------
        node : Union[Node,str,int]
            Node object, label, or id
        support : Union[Support,str]
            The support type, a :class:`ospgrid.grid.Support` object. Alternatively,
            a single character support descriptor as a :class:`str` can be used as
            follows:

                - `X` = :attr:`ospgrid.grid.Support.PINNED_X`
                - `Y` = :attr:`ospgrid.grid.Support.PINNED_Y`
                - `F` = :attr:`ospgrid.grid.Support.FIXED`
                - `V` = :attr:`ospgrid.grid.Support.FIXED_V_ROLLER`
                - `P` = :attr:`ospgrid.grid.Support.PROP`

        Returns
        -------
        None.

        """
        the_node = self.get_node(node)
        if isinstance(support, str):
            if support == "F":
                the_node.set_support(Support.FIXED)
            elif support == "X":
                the_node.set_support(Support.PINNED_X)
            elif support == "Y":
                the_node.set_support(Support.PINNED_Y)
            elif support == "P":
                the_node.set_support(Support.PROP)
            elif support == "V":
                the_node.set_support(Support.FIXED_V_ROLLER)
        else:
            the_node.set_support(support)

    def get_node(self, node=Union[Node, str, int]):
        """
        Gets the node from an node object, id or label.

        Parameters
        ----------
        node : Union[Node,str,int]
            Node object, label, or id

        Raises
        ------
        ValueError
            If node object, label, or id not passed, or is multiple nodes match the
            label, or no node found.

        Returns
        -------
        Node object.

        """

        if isinstance(node, Node):
            return node

        if isinstance(node, str):
            node_match = [n for n in self.nodes if n.label == node]
            if len(node_match) > 1:
                raise ValueError(f"More than one node has label: '{node}'")
            elif len(node_match) == 0:
                raise ValueError(f"Cannot find node '{node}' - is it defined?")
            return node_match[0]

        if isinstance(node, int):
            return self.nodes[node]

        # We should never get here
        raise ValueError("Either node object, label, or node id must be passed")

    def get_member(self, member=Union[Member, int, Tuple[str, str]]):
        """
        Gets the member from a member object, id, or a tuple of node labels.

        Parameters
        ----------
        member : Union[Member,int, Tuple[str,str]], optional Member object, id, or
            node labels.

        Raises
        ------
        ValueError
            If member object, or id not passed, or is multiple nodes match the
            label, or no node found.

        Returns
        -------
        None.

        """
        if isinstance(member, Member):
            return member

        if isinstance(member, int):
            return self.members[member]

        if isinstance(member, tuple):
            node_i_lbl = [m.node_i.label for m in self.members]
            node_j_lbl = [m.node_j.label for m in self.members]
            for n_comb in itertools.product(node_i_lbl, node_j_lbl):
                if sorted(member) == sorted(n_comb):
                    # The combination exists, so now let's find the member
                    the_member = [
                        m
                        for m in self.members
                        if (m.node_i.label == member[0] and m.node_j.label == member[1])
                        or (m.node_i.label == member[1] and m.node_j.label == member[0])
                    ]
                    return the_member[0]

            raise ValueError(
                f"Member with nodes {member[0]} and {member[1]} could not be found."
            )

        # We should never get here
        raise ValueError("Either member object or id must be passed")

    def analyze(self) -> osp:
        """
        Executes the analysis for the grid object using OpenSeesPy

        Returns
        -------
        osp : OpenSeesPy instance
            The OpenSeesPy instance, which can be used for querying results directly
            or otherwise manipulating the model further.

        """
        # remove any existing model
        osp.wipe()

        # set modelbuilder, 3 dims, 6 DOF
        osp.model("basic", "-ndm", 3, "-ndf", 6)

        # create nodes & add support, load, if any
        for n in self.nodes:
            osp.node(n.idx, n.x, n.y, 0.0)
            if n.support is not None:
                osp.fix(n.idx, *n.support.value)

        # Nominal E and G
        E = 200e9  # GPa
        G = 80e9  # GPa

        # define materials
        osp.uniaxialMaterial("Elastic", 1, E)

        # Define geometry transforms
        vecxz = [0, 0, 1]
        osp.geomTransf("Linear", 1, *vecxz)

        # define elements
        # tag   *[ndI ndJ]  A  E  G  Jx  Iy   Iz  transfOBJs
        for m in self.members:
            I = m.EI / E
            J = m.GJ / G
            A = I / 1e6  # rough
            osp.element(
                "elasticBeamColumn",
                m.idx,
                m.node_i.idx,
                m.node_j.idx,
                A,
                E,
                G,
                J,
                I,
                0.1 * I,
                1,
            )

        # create TimeSeries
        osp.timeSeries("Constant", 1)

        # create a plain load pattern
        osp.pattern("Plain", 1, 1)

        # Create the nodal load: nodeID loadvals
        # xForce yForce zForce xMoment yMoment zMoment
        for n in self.nodes:
            if n.Fz != 0 or n.Mx != 0 or n.My != 0:
                loadVals = [0, 0, n.Fz, n.Mx, n.My, 0]
                osp.load(n.idx, *loadVals)

        # ------------------------------
        # Start of analysis generation
        # ------------------------------

        # create SOE
        osp.system("FullGeneral")

        # create DOF number
        osp.numberer("RCM")

        # create constraint handler
        osp.constraints("Plain")

        # create integrator
        osp.integrator("LoadControl", 1.0)

        # create algorithm
        osp.algorithm("Linear")

        # create analysis object
        osp.analysis("Static")

        # perform the analysis
        osp.analyze(1)

        # calculate reactions
        osp.reactions()

        return osp

    def get_displacement(self, node=Union[Node, str, int], dof: int = -1):
        """
        Returns the displacements for a node

        Parameters
        ----------
        node : Union[Node,str,int]
            Node object, label, or id
        dof : int [optional]
            The degree of freedom of interest. Defaults to all.

        Raises
        ------
        ValueError
            If node object, label, or id not passed, or is multiple nodes match the
            label, or no node found.

        Returns
        -------
        None.

        """
        the_node = self.get_node(node)
        return osp.nodeDisp(the_node.idx, dof)

    def get_reactions(self, node=Union[Node, str, int], dof: int = -1):
        """
        Returns the reactions for a node

        Parameters
        ----------
        node : Union[Node,str,int]
            Node object, label, or id
        dof : int [optional]
            The degree of freedom of interest. Defaults to all.

        Raises
        ------
        ValueError
            If node object, label, or id not passed, or is multiple nodes match the
            label, or no node found.

        Returns
        -------
        None.

        """
        the_node = self.get_node(node)
        return osp.nodeReaction(the_node.idx, dof)

    def get_system_stiffness(self) -> np.ndarray:
        """
        Returns the system global stiffness matrix after the imposition of boundary
        conditions.

        Returns
        -------
        K : np.ndarray
            A square numpy array of the reduced global stiffness matrix

        """
        K = osp.printA("-ret")
        n = osp.systemSize()
        return np.reshape(K, (n, n))

    def get_system_force(self) -> np.ndarray:
        """
        Returns the system global force vector after the imposition of boundary
        conditions.

        Returns
        -------
        F : np.ndarray
            A numpy vector of the reduced global system force vector

        """
        #
        raise NotImplementedError("Functionality not yet available")

    def get_member_forces(
        self, member: Union[Member, int, Tuple[str, str]], dof: int = -1
    ) -> List[float]:
        """
        Returns the member end forces for the indicated DOF in the global coordinate
        system.

        Parameters
        ----------
        member : Union[Member, int, Tuple[str,str]]
            Member object, id, or a tuple of node labels.
        dof : int [optional]
            The degree of freedom of interest. Defaults to all.

        Returns
        -------
        F : np.array
            A 12 element list: 6 DOFs for node i, and 6 DOFs for node j, in the order
            Fx, Fy, Fz, Mx, My, Mz.

        """
        the_member = self.get_member(member)
        return np.array(osp.eleForce(the_member.idx, dof))

    def plot_results(
        self,
        save_files: bool = False,
        figsize=None,
        axes_on: bool = True,
        scale_factor: float = 2e2,
    ):
        """
        Plot the results of the grid analysis including:
            - the grid
            - the deflected shape
            - the BMD, SFD, and TMD

        Parameters
        ----------
        save_files : bool, optional
            Whether or not to save the plots to PDF. The default is False.
        axes_on : bool, optional
            Whether or not to have the axes on in the plots. The default is True.
        scale_factor : float, optional
            The scale factor to use for the deflected shape.

        Returns
        -------
        None.

        """

        self.plot_grid(figsize=figsize, axes_on=axes_on)
        self.plot_dsd(scale_factor=scale_factor, figsize=figsize, axes_on=axes_on)
        self.plot_bmd(figsize=figsize, axes_on=axes_on)
        self.plot_tmd(figsize=figsize, axes_on=axes_on)
        self.plot_sfd(figsize=figsize, axes_on=axes_on)

        if save_files:
            save_figs("ospy_plots.pdf")
            plt.close("all")
        else:
            plt.show()

    def plot_grid(self, figsize=None, axes_on: bool = True):
        """
        Plot the grid, showing nodes & members, and their indices.

        Parameters
        ----------
        figsize : TYPE, optional
            The size of the figure in inches. The default is self.FIGSIZE.
        axes_on : bool, optional
            Whether or not to have the axes on in the plots. The default is True.

        Returns
        -------
        None.

        """
        if figsize is None:
            figsize = self.FIGSIZE

        ospv.plot_model(fig_wi_he=figsize)
        fig = plt.gcf()
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        plt.gcf().suptitle("Model")

        if not axes_on:
            plt.gca().set_axis_off()
        else:
            fig.tight_layout()

    def plot_dsd(self, scale_factor: float = 2e2, figsize=None, axes_on: bool = True):
        """
        Plot the deflected shape diagram.

        Parameters
        ----------
        scale_factor : float
            The scale of the deformations to use. The default is 200.
        figsize : TYPE, optional
            The size of the figure in inches. The default is self.FIGSIZE.
        axes_on : bool, optional
            Whether or not to have the axes on in the plots. The default is True.

        Returns
        -------
        None.

        """
        if figsize is None:
            figsize = self.FIGSIZE

        ospv.plot_defo(sfac=scale_factor, endDispFlag=1)
        fig = plt.gcf()
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        plt.gcf().suptitle(f"Displaced Shape\n(Scale: {scale_factor})")
        if not axes_on:
            plt.gca().set_axis_off()
        else:
            fig.tight_layout()

    def plot_bmd(self, scale_factor: float = 1.0, figsize=None, axes_on: bool = True):
        """
        Plot the bending moment diagram.

        Parameters
        ----------
        scale_factor : float
            The scale of the bending moment to use. The default is 1.0. A negative
            sign is then applied to flip the diagram so that it appears on the tension
            face, per convention.
        figsize : TYPE, optional
            The size of the figure in inches. The default is self.FIGSIZE.
        axes_on : bool, optional
            Whether or not to have the axes on in the plots. The default is True.

        Returns
        -------
        None.

        """
        if figsize is None:
            figsize = self.FIGSIZE

        ospv.section_force_diagram_3d("My", Ew={}, sfac=-scale_factor)
        fig = plt.gcf()
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.suptitle(f"Bending Moment Diagram\n(Scale: {scale_factor})")

        if not axes_on:
            plt.gca().set_axis_off()
        else:
            fig.tight_layout()

    def plot_sfd(self, scale_factor: float = 1.0, figsize=None, axes_on: bool = True):
        """
        Plot the shear force diagram.

        Parameters
        ----------
        scale_factor : float
            The scale of the bending moment to use. The default is 1.0. A negative
            sign is then applied to flip the diagram so that it appears per convention.
        figsize : TYPE, optional
            The size of the figure in inches. The default is self.FIGSIZE.
        axes_on : bool, optional
            Whether or not to have the axes on in the plots. The default is True.

        Returns
        -------
        None.

        """
        if figsize is None:
            figsize = self.FIGSIZE

        ospv.section_force_diagram_3d("Vz", Ew={}, sfac=-scale_factor)
        fig = plt.gcf()
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        plt.gcf().suptitle(f"Shear Force Diagram\n(Scale: {scale_factor})")

        if not axes_on:
            plt.gca().set_axis_off()
        else:
            fig.tight_layout()

    def plot_tmd(self, scale_factor: float = 1.0, figsize=None, axes_on: bool = True):
        """
        Plot the torsion moment diagram.

        Parameters
        ----------
        scale_factor : float
            The scale of the bending moment to use. The default is 1.0.  A negative
            sign is then applied to flip the diagram so that it appears per convention.
        figsize : TYPE, optional
            The size of the figure in inches. The default is self.FIGSIZE.
        axes_on : bool, optional
            Whether or not to have the axes on in the plots. The default is True.

        Returns
        -------
        None.

        """
        if figsize is None:
            figsize = self.FIGSIZE

        ospv.section_force_diagram_3d("T", Ew={}, sfac=-scale_factor, dir_plt=2)
        fig = plt.gcf()
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        plt.gcf().suptitle(f"Torsion Moment Diagram\n(Scale: {scale_factor})")

        if not axes_on:
            plt.gca().set_axis_off()
        else:
            fig.tight_layout()


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
