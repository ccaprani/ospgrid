"""
Basic tests for ospgrid operation
"""

import pytest
import numpy as np
import ospgrid as ospg

"""
Based on the S1 2021 Exam, CIV42809, Monash Uni.
"""


def grid_params():
    """
    Gives the grid parameters
    """
    # Define some inputs
    Lae = 2.0  # m
    Lbe = 4.75  # m
    Lce = 4.0  # m
    Lde = 4.0  # m
    EI = 25e3  # kNm2
    GJ = 15e3  # kNm2
    P = -50  # kN
    Mx = 25  # kNm
    My = 15  # kNm

    return [Lae, Lbe, Lce, Lde, EI, GJ, P, Mx, My]


def sysK():
    """
    Returns the restricted global stiffness matrix for the pre-defined
    basic grid topology
    """
    params = grid_params()
    Lae, Lbe, Lce, Lde, EI, GJ, P, Mx, My = params

    # FZ at E
    k11 = 12 * EI / Lde**3 + 12 * EI / Lbe**3 + 12 * EI / Lce**3  # from dZ
    k12 = -6 * EI / Lde**2 + 6 * EI / Lbe**2  # from rX
    k13 = -6 * EI / Lce**2  # from rY

    # RX at E
    k21 = -6 * EI / Lde**2 + 6 * EI / Lbe**2  # from dZ
    k22 = 4 * EI / Lbe + 4 * EI / Lde + GJ / Lce  # from rX
    k23 = 0  # from rY

    # RY at E
    k31 = -6 * EI / Lce**2  # from dZ
    k32 = 0  # from rX
    k33 = 4 * EI / Lce + GJ / Lbe + GJ / Lde  # from rY

    K = np.array([[k11, k12, k13], [k21, k22, k23], [k31, k32, k33]])
    return K


def getF():
    """
    Returns the force vector for the predefined grid
    """
    params = grid_params()

    P = params[6]
    Mx = params[7]
    My = params[8]
    return np.array([P, Mx, My])


def do_ospg_analysis():

    params = grid_params()
    Lae, Lbe, Lce, Lde, EI, GJ, P, Mx, My = params

    grid = ospg.Grid()

    grid.add_node("A", -Lae, 0.0)
    grid.add_node("B", 0.0, Lbe)
    grid.add_node("C", Lce, 0.0)
    grid.add_node("D", 0.0, -Lde)
    grid.add_node("E", 0.0, 0.0)

    grid.add_member("A", "E", EI, GJ)
    grid.add_member("B", "E", EI, GJ)
    grid.add_member("C", "E", EI, GJ)
    grid.add_member("D", "E", EI, GJ)

    grid.add_load("E", P, Mx, My)

    grid.add_support("B", ospg.Support.FIXED)
    grid.add_support("C", ospg.Support.FIXED)
    grid.add_support("D", ospg.Support.FIXED)

    grid.analyze()

    return grid


def test_basic_grid():
    """
    Execute a two-span beam analysis and check the reaction results
    """

    grid = do_ospg_analysis()

    delta_E = grid.get_displacement("E", 3)
    theta_E = grid.get_displacement("E", 4)
    phi_E = grid.get_displacement("E", 5)

    grid_disps = [delta_E, theta_E, phi_E]

    # Direct stiffness method
    K = sysK()
    F = getF()
    D = np.linalg.solve(K, F)

    assert grid_disps == pytest.approx(D)
