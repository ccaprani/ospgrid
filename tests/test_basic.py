"""
Tests for ospgrid operation.

Based on exam problems from CIV4280 Bridge Design & Assessment, Monash University.
"""

import pathlib
import pytest
import matplotlib.pyplot as plt
import numpy as np
import ospgrid as ospg


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def grid_params():
    """Return parameters for the standard 4-member star grid test case."""
    Lae = 2.0  # m
    Lbe = 4.75  # m
    Lce = 4.0  # m
    Lde = 4.0  # m
    EI = 25e3  # kNm²
    GJ = 15e3  # kNm²
    P = -50  # kN
    Mx = 25  # kNm
    My = 15  # kNm
    return [Lae, Lbe, Lce, Lde, EI, GJ, P, Mx, My]


def sysK():
    """
    Return the 3×3 reduced global stiffness matrix at node E for the standard
    test grid (B, C, D fixed; A free with no load → condensed contribution = 0).
    """
    Lae, Lbe, Lce, Lde, EI, GJ, P, Mx, My = grid_params()

    k11 = 12 * EI / Lde**3 + 12 * EI / Lbe**3 + 12 * EI / Lce**3
    k12 = -6 * EI / Lde**2 + 6 * EI / Lbe**2
    k13 = -6 * EI / Lce**2
    k21 = k12
    k22 = 4 * EI / Lbe + 4 * EI / Lde + GJ / Lce
    k23 = 0
    k31 = k13
    k32 = 0
    k33 = 4 * EI / Lce + GJ / Lbe + GJ / Lde

    return np.array([[k11, k12, k13], [k21, k22, k23], [k31, k32, k33]])


def getF():
    """Return the force vector [Fz, Mx, My] at node E."""
    P, Mx, My = grid_params()[6:]
    return np.array([P, Mx, My])


def do_ospg_analysis():
    """Build and analyse the standard 4-member star grid. Returns the Grid."""
    Lae, Lbe, Lce, Lde, EI, GJ, P, Mx, My = grid_params()

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


# ---------------------------------------------------------------------------
# Numerical correctness
# ---------------------------------------------------------------------------


def test_basic_grid():
    """
    Analyse a 4-member star grid and verify node-E displacements match the
    direct stiffness hand calculation (S1 2021 exam, CIV4280, Monash).

    Note: node A is unsupported and unloaded. Its condensed stiffness
    contribution to E is identically zero (proven algebraically), so the
    3×3 hand stiffness matrix at E omits member AE.
    """
    grid = do_ospg_analysis()

    delta_E = grid.get_displacement("E", 3)
    theta_E = grid.get_displacement("E", 4)
    phi_E = grid.get_displacement("E", 5)

    K = sysK()
    F = getF()
    D = np.linalg.solve(K, F)

    assert [delta_E, theta_E, phi_E] == pytest.approx(D)


def test_reactions_equilibrium():
    """Vertical reactions at fixed supports must sum to the applied vertical load."""
    grid = do_ospg_analysis()
    _, _, _, _, _, _, P, Mx, My = grid_params()

    Rz_B = grid.get_reactions("B", 3)
    Rz_C = grid.get_reactions("C", 3)
    Rz_D = grid.get_reactions("D", 3)

    assert Rz_B + Rz_C + Rz_D == pytest.approx(-P)


def test_member_forces_equilibrium():
    """
    Sum of member end forces at the loaded node E must equal the applied load.

    OpenSeesPy's eleResponse(..., 'forces') returns the element *resisting*
    force vector in global coordinates.  At a free (loaded) node in
    equilibrium, the sum of element resisting forces equals the applied
    external load directly (K·D = F_ext ⟹ Σf_elem = F_ext).

    Element end forces are ordered [Fx, Fy, Fz, Mx, My, Mz] for each node;
    the j-node occupies indices 6–11.
    """
    grid = do_ospg_analysis()
    _, _, _, _, _, _, P, Mx, My = grid_params()

    Fz_sum = Mx_sum = My_sum = 0.0
    for m in grid.members:
        f = grid.get_member_forces(m)
        Fz_sum += f[8]
        Mx_sum += f[9]
        My_sum += f[10]

    assert Fz_sum == pytest.approx(P, rel=1e-5)
    assert Mx_sum == pytest.approx(Mx, rel=1e-5)
    assert My_sum == pytest.approx(My, rel=1e-5)


def test_system_stiffness_shape_and_symmetry():
    """
    The reduced system stiffness matrix must be square and symmetric.
    With nodes A and E both free (6 DOFs each), the matrix is 12×12.
    """
    grid = do_ospg_analysis()
    K = grid.get_system_stiffness()

    assert K.ndim == 2
    assert K.shape[0] == K.shape[1]
    assert K.shape[0] == 12
    assert K == pytest.approx(K.T)


# ---------------------------------------------------------------------------
# Member object methods
# ---------------------------------------------------------------------------


def test_member_local_stiffness():
    """Local stiffness matrix entries match the grid beam-element formula."""
    EI = 25e3  # kNm²
    GJ = 15e3  # kNm²
    L = 4.0  # m

    grid = ospg.Grid()
    grid.add_node("i", 0.0, 0.0)
    grid.add_node("j", L, 0.0)  # member along x-axis
    m = grid.add_member("i", "j", EI, GJ)

    k = m.get_local_stiffness()

    assert k[0, 0] == pytest.approx(12 * EI / L**3)
    assert k[1, 1] == pytest.approx(GJ / L)
    assert k[2, 2] == pytest.approx(4 * EI / L)
    assert k[2, 5] == pytest.approx(2 * EI / L)
    assert k[0, 2] == pytest.approx(6 * EI / L**2)

    # Matrix must be symmetric
    assert k == pytest.approx(k.T)


def test_member_transformation_matrix_aligned():
    """For a member along the x-axis (c=1, s=0) the transformation matrix is I."""
    grid = ospg.Grid()
    grid.add_node("i", 0.0, 0.0)
    grid.add_node("j", 4.0, 0.0)
    m = grid.add_member("i", "j", 1.0, 1.0)

    T = m.get_transformation_matrix()
    assert T == pytest.approx(np.eye(6))


def test_member_transformation_matrix_skewed():
    """For a 45-degree member direction cosines are c = s = 1/√2."""
    grid = ospg.Grid()
    grid.add_node("i", 0.0, 0.0)
    grid.add_node("j", 1.0, 1.0)
    m = grid.add_member("i", "j", 1.0, 1.0)

    c = s = 1.0 / np.sqrt(2)
    T_expected = np.kron(np.eye(2), np.array([[1, 0, 0], [0, c, s], [0, -s, c]]))
    assert m.get_transformation_matrix() == pytest.approx(T_expected)


def test_member_global_stiffness_aligned():
    """For an axis-aligned member, global stiffness equals local stiffness."""
    EI = 10e3
    GJ = 5e3
    L = 3.0

    grid = ospg.Grid()
    grid.add_node("i", 0.0, 0.0)
    grid.add_node("j", L, 0.0)
    m = grid.add_member("i", "j", EI, GJ)

    assert m.get_global_stiffness() == pytest.approx(m.get_local_stiffness())


# ---------------------------------------------------------------------------
# Grid helper methods
# ---------------------------------------------------------------------------


def test_get_member_by_tuple():
    """get_member accepts a (label, label) tuple independent of order."""
    grid = do_ospg_analysis()

    m_ae = grid.get_member(("A", "E"))
    m_ea = grid.get_member(("E", "A"))
    assert m_ae is m_ea
    assert {m_ae.node_i.label, m_ae.node_j.label} == {"A", "E"}


def test_get_member_by_index():
    """get_member(int) returns the member at that list position."""
    grid = do_ospg_analysis()
    assert grid.get_member(0) is grid.members[0]


def test_get_member_by_object():
    """get_member(Member) returns the same object unchanged."""
    grid = do_ospg_analysis()
    m = grid.members[1]
    assert grid.get_member(m) is m


def test_get_member_not_found():
    """get_member raises ValueError when no matching member exists."""
    grid = do_ospg_analysis()
    with pytest.raises(ValueError):
        grid.get_member(("A", "C"))  # no direct member between A and C


def test_get_node_errors():
    """get_node raises ValueError for missing or ambiguous labels."""
    grid = ospg.Grid()
    grid.add_node("A", 0, 0)

    with pytest.raises(ValueError, match="Cannot find node"):
        grid.get_node("Z")

    # Duplicate labels should also raise
    grid.add_node("A", 1, 0)
    with pytest.raises(ValueError, match="More than one node"):
        grid.get_node("A")


def test_clear():
    """clear() resets node and member lists and counters."""
    grid = do_ospg_analysis()
    assert len(grid.nodes) == 5
    assert len(grid.members) == 4

    grid.clear()
    assert grid.nodes == []
    assert grid.members == []
    assert grid.no_nodes == 0
    assert grid.no_members == 0


def test_get_system_force_not_implemented():
    """get_system_force raises NotImplementedError (pending upstream support)."""
    grid = do_ospg_analysis()
    with pytest.raises(NotImplementedError):
        grid.get_system_force()


# ---------------------------------------------------------------------------
# Support string shortcuts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "char,enum_val",
    [
        ("F", ospg.Support.FIXED),
        ("X", ospg.Support.PINNED_X),
        ("Y", ospg.Support.PINNED_Y),
        ("P", ospg.Support.PROP),
        ("V", ospg.Support.FIXED_V_ROLLER),
    ],
)
def test_support_string_shortcuts(char, enum_val):
    """Single-character support strings map to the correct Support enum value."""
    grid = ospg.Grid()
    grid.add_node("A", 0, 0)
    grid.add_support("A", char)
    assert grid.nodes[0].support == enum_val


# ---------------------------------------------------------------------------
# make_grid (post.py)
# ---------------------------------------------------------------------------


def test_make_grid_topology():
    """make_grid parses node positions, supports, loads, and connectivity."""
    grid_str = "AF-5:0,BX0:-5,CN0:0,DY2:0_C0:-50:-50_AC,CD,BC"
    grid = ospg.make_grid(grid_str)

    assert len(grid.nodes) == 4
    assert len(grid.members) == 3

    node_A = grid.get_node("A")
    assert node_A.x == pytest.approx(-5.0)
    assert node_A.y == pytest.approx(0.0)
    assert node_A.support == ospg.Support.FIXED

    node_B = grid.get_node("B")
    assert node_B.support == ospg.Support.PINNED_X

    node_C = grid.get_node("C")
    assert node_C.Fz == pytest.approx(0.0)
    assert node_C.Mx == pytest.approx(-50.0)
    assert node_C.My == pytest.approx(-50.0)


def test_make_grid_analyzable():
    """A grid built via make_grid can be analysed without error."""
    grid_str = "AF-5:0,BX0:-5,CN0:0,DY2:0_C0:-50:-50_AC,CD,BC"
    grid = ospg.make_grid(grid_str)
    grid.analyze()  # must not raise


def test_make_grid_invalid_member():
    """make_grid raises ValueError when a member references an undefined node."""
    with pytest.raises(ValueError):
        ospg.make_grid("AF0:0,BX1:0_A0:0:0_AZ")  # Z is not defined


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def test_plotting(monkeypatch):
    """plot_results completes without error (plt.show suppressed)."""
    grid = do_ospg_analysis()
    monkeypatch.setattr(plt, "show", lambda: None)
    grid.plot_results()
    plt.close("all")


def test_individual_plot_methods(monkeypatch):
    """Each individual plot method runs without error."""
    grid = do_ospg_analysis()
    monkeypatch.setattr(plt, "show", lambda: None)

    for method in (
        grid.plot_grid,
        grid.plot_dsd,
        grid.plot_bmd,
        grid.plot_sfd,
        grid.plot_tmd,
    ):
        method()
        plt.close("all")


def test_plot_results_scale_factor_list(monkeypatch):
    """plot_results accepts a 4-element list of scale factors."""
    grid = do_ospg_analysis()
    monkeypatch.setattr(plt, "show", lambda: None)
    grid.plot_results(scale_factor=[0.0, 1.0, 1.0, 1.0])
    plt.close("all")


def test_plot_results_scale_factor_list_wrong_length(monkeypatch):
    """plot_results raises ValueError when the scale-factor list is not length 4."""
    grid = do_ospg_analysis()
    monkeypatch.setattr(plt, "show", lambda: None)
    with pytest.raises(ValueError):
        grid.plot_results(scale_factor=[1.0, 1.0])


def test_plot_results_save_pdf(monkeypatch, tmp_path):
    """plot_results saves a single multi-page PDF when filename ends in .pdf."""
    grid = do_ospg_analysis()
    monkeypatch.setattr(plt, "show", lambda: None)

    out = tmp_path / "results.pdf"
    grid.plot_results(save_figs=True, filename=str(out))

    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_results_save_png(monkeypatch, tmp_path):
    """plot_results saves five named PNG files when filename ends in .png."""
    grid = do_ospg_analysis()
    monkeypatch.setattr(plt, "show", lambda: None)

    out = tmp_path / "results.png"
    grid.plot_results(save_figs=True, filename=str(out))

    for suffix in ("mdl", "dsd", "bmd", "sfd", "tmd"):
        assert (tmp_path / f"results_{suffix}.png").exists()
