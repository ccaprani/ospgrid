#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 20:25:38 2022

@author: ccaprani
"""

from enum import Enum
import matplotlib.pyplot as plt
import openseespy.opensees as osp
import opsvis as ospv
import numpy as np
from .utils import save_figs_to_file

__all__ = ["Support", "Node", "Member", "Grid"]


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
    Object encapsulating the properties for a grid member
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
            The flexural rigidity.
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
        self.L = np.sqrt(self.delta_x**2 + self.delta_y**2)

    def get_local_stiffness(self) -> np.ndarray:
        """
        Returns the member stiffness matrix in local coordinates with nodal DOFs
        in the order DZ ('vertical' force), RX (torsion), RZ (bending moment).

        Returns
        -------
        K : np.ndarray
            Member stiffness matrix in local coordinates.

        """

        k11 = 12 * self.EI / self.L**3
        k13 = 6 * self.EI / self.L**2
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
        Gives the transformation matrix relating the member degrees of freedom from
        local to global coordinate system.

        It considers only the grid member DOFs, per :func:`get_local_stiffness`

        Returns
        -------
        T : np.ndarray
            Transformation matrix

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

        # Set color preferences
        ospv.fmt_undefo["color"] = "black"
        ospv.fmt_defo["color"] = "red"
        ospv.fmt_model["color"] = "black"
        ospv.fmt_secforce1["color"] = "red"
        ospv.fmt_secforce2["color"] = "red"

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
        node : Node
            The created Node instance.

        """
        self.no_nodes += 1
        node = Node(self.no_nodes, label, x, y)
        self.nodes.append(node)
        return node

    def add_member(
        self,
        node_i: "Node | str | int",
        node_j: "Node | str | int",
        EI: float,
        GJ: float,
    ):
        """
        Adds a member to the grid.

        Parameters
        ----------
        node_i : Node | str | int
            The starting node for the member.
        node_j : Node | str | int
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
        node: "Node | str | int",
        Fz: float = 0,
        Mx: float = 0,
        My: float = 0,
    ):
        """
        Add a load to the grid.

        Parameters
        ----------
        node : Node | str | int
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
        self, node: "Node | str | int", support: "Support | str" = None
    ):
        """
        Add a support to a node in the grid.

        Parameters
        ----------
        node : Node | str | int
            Node object, label, or id
        support : Support | str
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

    def get_node(self, node: "Node | str | int"):
        """
        Gets the node from a node object, id or label.

        Parameters
        ----------
        node : Node | str | int
            Node object, label, or id

        Raises
        ------
        ValueError
            If node object, label, or id not passed, or multiple nodes match the
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

    def get_member(self, member: "Member | int | tuple[str, str]"):
        """
        Gets the member from a member object, id, or a tuple of node labels.

        Parameters
        ----------
        member : Member | int | tuple[str, str]
            Member object, id, or a tuple of the two end-node labels (order
            does not matter).

        Raises
        ------
        ValueError
            If the member cannot be found.

        Returns
        -------
        Member object.

        """
        if isinstance(member, Member):
            return member

        if isinstance(member, int):
            return self.members[member]

        if isinstance(member, tuple):
            target = set(member)
            for m in self.members:
                if {m.node_i.label, m.node_j.label} == target:
                    return m
            raise ValueError(
                f"Member with nodes {member[0]} and {member[1]} could not be found."
            )

        # We should never get here
        raise ValueError("Either member object, id, or node label tuple must be passed")

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

        # E and G are nominal values that cancel algebraically:
        #   I  = EI / E   →   E * I  = EI   (bending stiffness terms use E*I)
        #   J  = GJ / G   →   G * J  = GJ   (torsion stiffness terms use G*J)
        # Consequently, every entry in the assembled stiffness matrix equals
        # the hand-calculation result (e.g. 12·EI/L³), expressed in the same
        # unit system the caller used for EI/GJ and node coordinates.
        # For example, with EI in kNm² and lengths in m, get_system_stiffness()
        # returns values in kN/m, kN, and kNm — matching a direct stiffness
        # hand calculation. Any non-zero E and G pair would give identical results.
        # Steel values are used here to keep the derived I and J numerically
        # well-conditioned for the linear solver.
        E = 200e9  # nominal (Pa); value does not affect results (see above)
        G = 80e9   # nominal (Pa); value does not affect results (see above)

        # define materials
        osp.uniaxialMaterial("Elastic", 1, E)

        # Define geometry transforms
        vecxz = [0, 0, 1]
        osp.geomTransf("Linear", 1, *vecxz)

        # define elements
        # tag   *[ndI ndJ]  A  E  G  Jx  Iy   Iz  transfTag
        for m in self.members:
            I = m.EI / E          # second moment of area derived from EI
            J = m.GJ / G          # torsion constant derived from GJ
            # A and Iz do not affect the grid solution (no axial/in-plane bending);
            # nominal values are used only to satisfy the OpenSeesPy API.
            A = I / 1e6           # nominal cross-sectional area
            Iz = 0.1 * I          # nominal weak-axis second moment of area
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
                Iz,
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

    def get_displacement(self, node: "Node | str | int", dof: int = -1):
        """
        Returns the displacements for a node

        Parameters
        ----------
        node : Node | str | int
            Node object, label, or id
        dof : int [optional]
            The degree of freedom of interest. Defaults to all.

        Raises
        ------
        ValueError
            If node object, label, or id not passed, or multiple nodes match the
            label, or no node found.

        Returns
        -------
        float or list
            Displacement for the requested DOF, or all DOFs if dof=-1.

        """
        the_node = self.get_node(node)
        return osp.nodeDisp(the_node.idx, dof)

    def get_reactions(self, node: "Node | str | int", dof: int = -1):
        """
        Returns the reactions for a node

        Parameters
        ----------
        node : Node | str | int
            Node object, label, or id
        dof : int [optional]
            The degree of freedom of interest. Defaults to all.

        Raises
        ------
        ValueError
            If node object, label, or id not passed, or multiple nodes match the
            label, or no node found.

        Returns
        -------
        float or list
            Reaction for the requested DOF, or all DOFs if dof=-1.

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

        .. todo::
            OpenSeesPy does not currently expose a direct API for retrieving the
            assembled force vector. This method is reserved for a future release.

        """
        # TODO: OpenSeesPy does not currently provide a public API equivalent to
        # printA for the force vector. Implement once upstream support is available.
        raise NotImplementedError("Functionality not yet available")

    def get_member_forces(
        self, member: "Member | int | tuple[str, str]", dof: int = -1
    ) -> np.ndarray:
        """
        Returns the member end forces for the indicated DOF in the global coordinate
        system.

        Parameters
        ----------
        member : Member | int | tuple[str, str]
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
        figsize=None,
        axes_on: bool = True,
        scale_factor: "float | list[float]" = 0.0,
        axis_title: bool = True,
        save_figs: bool = False,
        filename: str = "ospgrid_results.pdf",
        transparent: bool = False,
        bbox: bool = False,
        pad: int = 20,
        values: bool = True,
    ):
        """
        Plot the results of the grid analysis including:
            - the grid
            - the deflected shape
            - the BMD, SFD, and TMD

        Parameters
        ----------
        axes_on : bool, optional
            Whether or not to have the axes on in the plots. The default is True.
        scale_factor : float | list[float], optional
            If a single float: the scale of the deformations to use. When this value is
            zero, auto-scaling is done. The default is 0.
            If a list of floats of size 4, then the scale factors are applied in the order
            deformations; bending; shear; torsion.
        axis_title : bool, optional
            Whether or not to have the axes title in the plots. The default is True.
        save_figs : bool, optional
            Whether or not to save the plots to PDF. The default is False.
        filename : string
            The file to which the results are saved.
        transparent : bool, optional
            Whether or not the plots should be transparent. The default is False.
        bbox : bool, optional
            Whether or not to crop the figure to a bounding box of its contents. Only
            applies to image files, e.g., png, jpg, etc (not PDF)
        pad : int, optional
            If applying the bbox cropping to an image, a padding to apply to the
            contents. Defaults to 20 px.
        values : bool, optional
            Whether or not to print the salient values on the force diagrams

        Returns
        -------
        None.

        """

        if isinstance(scale_factor, float):
            sf_dsd = scale_factor
            sf_bmd = 1.0
            sf_sfd = 1.0
            sf_tmd = 1.0
        else:
            if len(scale_factor) != 4:
                raise ValueError("The list of scale factors must have length 4.")
            sf_dsd = scale_factor[0]
            sf_bmd = scale_factor[1]
            sf_sfd = scale_factor[2]
            sf_tmd = scale_factor[3]

        self.plot_grid(figsize=figsize, axes_on=axes_on, axis_title=axis_title)
        self.plot_dsd(
            scale_factor=sf_dsd,
            figsize=figsize,
            axes_on=axes_on,
            axis_title=axis_title,
        )
        self.plot_bmd(
            figsize=figsize,
            scale_factor=sf_bmd,
            axes_on=axes_on,
            axis_title=axis_title,
            values=values,
        )
        self.plot_sfd(
            figsize=figsize,
            scale_factor=sf_sfd,
            axes_on=axes_on,
            axis_title=axis_title,
            values=values,
        )
        self.plot_tmd(
            figsize=figsize,
            scale_factor=sf_tmd,
            axes_on=axes_on,
            axis_title=axis_title,
            values=values,
        )

        if save_figs:
            save_figs_to_file(filename, transparent=transparent, bbox=bbox, pad=pad)
            plt.close("all")
        else:
            plt.show()

    def plot_grid(
        self,
        figsize=None,
        axes_on: bool = True,
        axis_title: bool = True,
    ):
        """
        Plot the grid, showing nodes & members, and their indices.

        Parameters
        ----------
        figsize : tuple, optional
            The size of the figure in inches. The default is self.FIGSIZE.
        axes_on : bool, optional
            Whether or not to have the axes on in the plots. The default is True.
        axis_title : bool, optional
            Whether or not to have the axes title in the plots. The default is True.

        Returns
        -------
        None.

        """
        if figsize is None:
            figsize = self.FIGSIZE

        ospv.plot_model(fig_wi_he=figsize,
                        node_supports=False)
        fig = plt.gcf()
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        if axis_title:
            plt.gcf().suptitle("Model")

        if not axes_on:
            plt.gca().set_axis_off()

        fig.tight_layout()

    def _plot_model(self, ax):
        """
        Plots the background model for the section force diagrams.
        """

        ospv.plot_model(ax=ax,
                        node_labels=False,
                        element_labels=False,
                        node_supports=False,
                        local_axes=False)

    def plot_dsd(
        self,
        scale_factor: float = 0,
        figsize=None,
        axes_on: bool = True,
        axis_title: bool = True,
    ):
        """
        Plot the deflected shape diagram.

        Parameters
        ----------
        scale_factor : float
            The scale of the deformations to use. When this value is zero, auto-scaling
            is done. The default is 0.
        figsize : tuple, optional
            The size of the figure in inches. The default is self.FIGSIZE.
        axes_on : bool, optional
            Whether or not to have the axes on in the plots. The default is True.
        axis_title : bool, optional
            Whether or not to have the axes title in the plots. The default is True.

        Returns
        -------
        None.

        """
        if figsize is None:
            figsize = self.FIGSIZE

        if scale_factor == 0:
            disps = {n.label: self.get_displacement(n, 3) for n in self.nodes}
            max_disp = max(abs(min(disps.values())), abs(max(disps.values())))
            x = [n.x for n in self.nodes]
            y = [n.y for n in self.nodes]
            grid_size = max(max(x) - min(x), max(y) - min(y))

            # in case of very small nodal values
            max_disp = ospv.max_u_abs_from_beam_defo_interp_3d(max_disp, nep=21)

            # target about 1/4 the dimension of the grid
            sf = 0.25 * grid_size / max_disp
            # But round to some sensible values
            mag = 10 ** int(np.ceil(np.log10(sf)))
            scale_factor = round(10 * sf / mag) * mag / 10

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        self._plot_model(ax)
        ospv.plot_defo(sfac=scale_factor,
                       unDefoFlag=False,
                       ax=ax,
                       endDispFlag=False,
                       node_supports=False
                       )
        fig = plt.gcf()
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        if axis_title:
            plt.gcf().suptitle(f"Displaced Shape\n(Scale: {scale_factor})")
        if not axes_on:
            plt.gca().set_axis_off()

        fig.tight_layout()

    def plot_bmd(
        self,
        scale_factor: float = 1.0,
        figsize=None,
        axes_on: bool = True,
        axis_title: bool = True,
        values: bool = True,
    ):
        """
        Plot the bending moment diagram.

        Parameters
        ----------
        scale_factor : float
            The scale of the bending moment to use. The default is 1.0.
        figsize : tuple, optional
            The size of the figure in inches. The default is self.FIGSIZE.
        axes_on : bool, optional
            Whether or not to have the axes on in the plots. The default is True.
        axis_title : bool, optional
            Whether or not to have the axes title in the plots. The default is True.
        values : bool optional
            Whether or not to print the values at the member ends

        Returns
        -------
        None.

        """
        if figsize is None:
            figsize = self.FIGSIZE

        _, _, ax = ospv.section_force_diagram_3d("My",
                                                 sfac=scale_factor,
                                                 end_max_values=values,
                                                 node_supports=False,
                                                 alt_model_plot=2)
        self._plot_model(ax)
        fig = plt.gcf()
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        plt.gca().set_box_aspect(None)
        if axis_title:
            fig.suptitle(f"Bending Moment Diagram\n(Scale: {scale_factor})")

        if not axes_on:
            plt.gca().set_axis_off()

        fig.tight_layout()

    def plot_sfd(
        self,
        scale_factor: float = 1.0,
        figsize=None,
        axes_on: bool = True,
        axis_title: bool = True,
        values: bool = True,
    ):
        """
        Plot the shear force diagram.

        Parameters
        ----------
        scale_factor : float
            The scale of the shear force to use. The default is 1.0. A negative
            sign is then applied to flip the diagram so that it appears per convention.
        figsize : tuple, optional
            The size of the figure in inches. The default is self.FIGSIZE.
        axes_on : bool, optional
            Whether or not to have the axes on in the plots. The default is True.
        axis_title : bool, optional
            Whether or not to have the axes title in the plots. The default is True.
        values : bool optional
            Whether or not to print the values at the member ends

        Returns
        -------
        None.

        """
        if figsize is None:
            figsize = self.FIGSIZE

        _, _, ax = ospv.section_force_diagram_3d("Vz",
                                                 sfac=-scale_factor,
                                                 end_max_values=values,
                                                 node_supports=False,
                                                 alt_model_plot=2)
        self._plot_model(ax)
        fig = plt.gcf()
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        plt.gca().set_box_aspect(None)
        if axis_title:
            plt.gcf().suptitle(f"Shear Force Diagram\n(Scale: {scale_factor})")

        if not axes_on:
            plt.gca().set_axis_off()

        fig.tight_layout()

    def plot_tmd(
        self,
        scale_factor: float = 1.0,
        figsize=None,
        axes_on: bool = True,
        axis_title: bool = True,
        values: bool = True,
    ):
        """
        Plot the torsion moment diagram.

        Parameters
        ----------
        scale_factor : float
            The scale of the torsion moment to use. The default is 1.0. A negative
            sign is then applied to flip the diagram so that it appears per convention.
        figsize : tuple, optional
            The size of the figure in inches. The default is self.FIGSIZE.
        axes_on : bool, optional
            Whether or not to have the axes on in the plots. The default is True.
        axis_title : bool, optional
            Whether or not to have the axes title in the plots. The default is True.
        values : bool optional
            Whether or not to print the values at the member ends

        Returns
        -------
        None.

        """
        if figsize is None:
            figsize = self.FIGSIZE

        _, _, ax = ospv.section_force_diagram_3d("T",
                                                 sfac=-scale_factor,
                                                 dir_plt=2,
                                                 end_max_values=values,
                                                 node_supports=False,
                                                 alt_model_plot=2)
        self._plot_model(ax)
        fig = plt.gcf()
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        plt.gca().set_box_aspect(None)
        if axis_title:
            plt.gcf().suptitle(f"Torsion Moment Diagram\n(Scale: {scale_factor})")

        if not axes_on:
            plt.gca().set_axis_off()

        fig.tight_layout()
