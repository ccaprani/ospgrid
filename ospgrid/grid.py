#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 20:25:38 2022

@author: ccaprani
"""

from enum import Enum
from typing import Union
import matplotlib.pyplot as plt
import openseespy.opensees as ops
import opsvis as opsv
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
            The support type.

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


class Grid:
    """
    A class that provides a user-friendly interface to OpenSeesPy for the analysis
    of plane elastic grids.
    """

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
        the_node_i = self._get_node(node_i)
        the_node_j = self._get_node(node_j)
        member = Member(self.no_members, the_node_i, the_node_j, EI, GJ)
        self.members.append(member)
        return member

    def add_load(
        self, node: Union[Node, str, int], Fz: float = 0, Mx: float = 0, My: float = 0,
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
        the_node = self._get_node(node)
        the_node.set_load(Fz, Mx, My)

    def add_support(self, node: Union[Node, str, int], support: Support = None):
        """
        Add a support to a node in the grid.

        Parameters
        ----------
        node : Union[Node,str,int]
            Node object, label, or id
        support : Support
            The support type for the node.

        Returns
        -------
        None.

        """
        the_node = self._get_node(node)
        the_node.set_support(support)

    def _get_node(self, node=Union[Node, str, int]):
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

    def analyze(self):
        """
        Executes the analysis for the grid object using OpenSeesPy

        Returns
        -------
        ops : OpenSeesPy instance
            The OpenSeesPy instance, which can be used for querying results directly
            or otherwise manipulating the model further.

        """
        # remove any existing model
        ops.wipe()

        # set modelbuilder, 3 dims, 6 DOF
        ops.model("basic", "-ndm", 3, "-ndf", 6)

        # create nodes & add support, load, if any
        for n in self.nodes:
            ops.node(n.idx, n.x, n.y, 0.0)
            if n.support is not None:
                ops.fix(n.idx, *n.support.value)

        # Nominal E and G
        E = 200e9  # GPa
        G = 80e9  # GPa

        # define materials
        ops.uniaxialMaterial("Elastic", 1, E)

        # Define geometry transforms
        vecxz = [0, 0, 1]
        ops.geomTransf("Linear", 1, *vecxz)

        # define elements
        # tag   *[ndI ndJ]  A  E  G  Jx  Iy   Iz  transfOBJs
        for m in self.members:
            I = m.EI / E
            J = m.GJ / G
            A = I / 1e6  # rough
            ops.element(
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
        ops.timeSeries("Constant", 1)

        # create a plain load pattern
        ops.pattern("Plain", 1, 1)

        # Create the nodal load: nodeID loadvals
        # xForce yForce zForce xMoment yMoment zMoment
        for n in self.nodes:
            if n.Fz != 0 or n.Mx != 0 or n.My != 0:
                loadVals = [0, 0, n.Fz, n.Mx, n.My, 0]
                ops.load(n.idx, *loadVals)

        # ------------------------------
        # Start of analysis generation
        # ------------------------------

        # create SOE
        ops.system("FullGeneral")

        # create DOF number
        ops.numberer("RCM")

        # create constraint handler
        ops.constraints("Plain")

        # create integrator
        ops.integrator("LoadControl", 1.0)

        # create algorithm
        ops.algorithm("Linear")

        # create analysis object
        ops.analysis("Static")

        # perform the analysis
        ops.analyze(1)

        # calculate reactions
        ops.reactions()

        return ops

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
        the_node = self._get_node(node)
        return ops.nodeDisp(the_node.idx, dof)

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
        the_node = self._get_node(node)
        return ops.nodeReaction(the_node.idx, dof)

    def plot_results(
        self, save_files: bool = False, axes_on: bool = True, sfac: float = 2e2
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
        sfac : float, optional
            The scale factor to use for the deflected shape.

        Returns
        -------
        None.

        """

        # Plotting
        fig_wi_he = (10.0, 10.0)

        opsv.plot_model(fig_wi_he=fig_wi_he)
        plt.gcf().suptitle("Model")

        opsv.plot_defo(sfac, endDispFlag=1, fig_wi_he=fig_wi_he)
        plt.pause(1)  # We must wait for plot_defo to finish with the canvas
        plt.gcf().suptitle("Displaced Shape (mm)")

        opsv.fig_wi_he = fig_wi_he  # to set sizes for figs below

        opsv.section_force_diagram_3d("My", Ew={}, sfac=-1.0e-3)
        plt.gcf().suptitle("Bending Moment Diagram (My - kNm)")

        opsv.section_force_diagram_3d("T", Ew={}, sfac=1.0e-3, dir_plt=2)
        # opsv.section_force_diagram_3d("T", Ew={}, sfac=1.0e-3)
        plt.gcf().suptitle("Torsion Moment Diagram (kNm)")

        opsv.section_force_diagram_3d("Vz", Ew={}, sfac=1.0e-3)
        plt.gcf().suptitle("Shear Force Diagram (Vz - kN)")

        figs = [plt.figure(n) for n in plt.get_fignums()]
        for fig in figs:
            if not axes_on:
                fig.axes[0].set_axis_off()
            else:
                fig.tight_layout()

        if save_files:
            save_figs("ospy_plots.pdf")
            plt.close("all")
        else:
            plt.show()
