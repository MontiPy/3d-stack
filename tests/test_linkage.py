"""Tests for the 3D linkage model and analysis."""

import math

import numpy as np
import pytest

from tolerance_stack.linkage import Joint, JointType, Link, Linkage
from tolerance_stack.linkage_analysis import (
    linkage_worst_case,
    linkage_rss,
    linkage_monte_carlo,
    analyze_linkage,
    _compute_jacobian,
)
from tolerance_stack.models import Distribution


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_planar_linkage() -> Linkage:
    """Single link along X with a revolute joint at the base.

    Base (revolute_z 0deg) -> Link (100mm, +X) -> Tip (fixed)
    """
    linkage = Linkage(name="Simple")
    linkage.add_joint(Joint("Base", JointType.REVOLUTE_Z, nominal=0.0,
                            plus_tol=1.0, minus_tol=1.0))
    linkage.add_link(Link("Arm", length=100.0, plus_tol=0.1, minus_tol=0.1,
                          direction=(1, 0, 0)))
    linkage.add_joint(Joint("Tip", JointType.FIXED))
    return linkage


def _two_bar_linkage() -> Linkage:
    """Two-bar planar linkage in the XY plane."""
    linkage = Linkage(name="TwoBar")
    linkage.add_joint(Joint("Base", JointType.REVOLUTE_Z, nominal=30.0))
    linkage.add_link(Link("Link1", length=100.0, plus_tol=0.1, minus_tol=0.1,
                          direction=(1, 0, 0)))
    linkage.add_joint(Joint("Elbow", JointType.REVOLUTE_Z, nominal=45.0,
                            plus_tol=0.5, minus_tol=0.5))
    linkage.add_link(Link("Link2", length=80.0, plus_tol=0.08, minus_tol=0.08,
                          direction=(1, 0, 0)))
    linkage.add_joint(Joint("Tip", JointType.FIXED))
    return linkage


# ---------------------------------------------------------------------------
# Transform tests
# ---------------------------------------------------------------------------

class TestTransforms:
    def test_rotx_90(self):
        j = Joint("J", JointType.REVOLUTE_X, nominal=90.0)
        T = j.transform()
        expected_y = np.array([0, 0, 1, 0])  # Y maps to Z
        np.testing.assert_allclose(T @ [0, 1, 0, 0], expected_y, atol=1e-12)

    def test_roty_90(self):
        j = Joint("J", JointType.REVOLUTE_Y, nominal=90.0)
        T = j.transform()
        # X maps to -Z
        result = T @ np.array([1, 0, 0, 0])
        np.testing.assert_allclose(result, [0, 0, -1, 0], atol=1e-12)

    def test_rotz_90(self):
        j = Joint("J", JointType.REVOLUTE_Z, nominal=90.0)
        T = j.transform()
        # X maps to Y
        result = T @ np.array([1, 0, 0, 0])
        np.testing.assert_allclose(result, [0, 1, 0, 0], atol=1e-12)

    def test_prismatic_x(self):
        j = Joint("J", JointType.PRISMATIC_X, nominal=5.0)
        T = j.transform()
        np.testing.assert_allclose(T[:3, 3], [5, 0, 0], atol=1e-12)

    def test_prismatic_y(self):
        j = Joint("J", JointType.PRISMATIC_Y, nominal=3.0)
        T = j.transform()
        np.testing.assert_allclose(T[:3, 3], [0, 3, 0], atol=1e-12)

    def test_link_translation(self):
        lk = Link("L", length=10.0, direction=(0, 1, 0))
        T = lk.transform()
        np.testing.assert_allclose(T[:3, 3], [0, 10, 0], atol=1e-12)

    def test_link_diagonal(self):
        lk = Link("L", length=10.0, direction=(1, 1, 0))
        T = lk.transform()
        d = np.array([1, 1, 0]) / np.sqrt(2) * 10
        np.testing.assert_allclose(T[:3, 3], d, atol=1e-12)

    def test_fixed_joint_identity(self):
        j = Joint("J", JointType.FIXED)
        T = j.transform()
        np.testing.assert_allclose(T, np.eye(4), atol=1e-12)


# ---------------------------------------------------------------------------
# Linkage construction tests
# ---------------------------------------------------------------------------

class TestLinkageConstruction:
    def test_alternating_order(self):
        linkage = Linkage(name="Test")
        linkage.add_joint(Joint("J0", JointType.FIXED))
        with pytest.raises(ValueError, match="Expected a link"):
            linkage.add_joint(Joint("J1", JointType.FIXED))

    def test_link_before_joint(self):
        linkage = Linkage(name="Test")
        with pytest.raises(ValueError, match="Expected a joint"):
            linkage.add_link(Link("L", length=10))

    def test_incomplete_validation(self):
        linkage = Linkage(name="Test")
        linkage.add_joint(Joint("J0", JointType.FIXED))
        linkage.add_link(Link("L", length=10))
        with pytest.raises(ValueError, match="incomplete"):
            linkage.validate()

    def test_complete_linkage(self):
        linkage = _simple_planar_linkage()
        assert linkage.is_complete
        linkage.validate()  # should not raise


# ---------------------------------------------------------------------------
# Forward kinematics tests
# ---------------------------------------------------------------------------

class TestForwardKinematics:
    def test_straight_link(self):
        """Base at 0 degrees, link along X = tip at (100, 0, 0)."""
        linkage = _simple_planar_linkage()
        pos = linkage.end_effector_position()
        np.testing.assert_allclose(pos, [100, 0, 0], atol=1e-10)

    def test_rotated_90(self):
        """Base at 90 degrees -> tip at (0, 100, 0)."""
        linkage = Linkage(name="Test")
        linkage.add_joint(Joint("Base", JointType.REVOLUTE_Z, nominal=90.0))
        linkage.add_link(Link("Arm", length=100.0, direction=(1, 0, 0)))
        linkage.add_joint(Joint("Tip", JointType.FIXED))

        pos = linkage.end_effector_position()
        np.testing.assert_allclose(pos, [0, 100, 0], atol=1e-10)

    def test_two_bar_geometry(self):
        """Verify two-bar linkage tip position analytically."""
        linkage = _two_bar_linkage()
        # Base at 30 deg, link1=100
        # After link1: (100*cos30, 100*sin30, 0)
        # Elbow at 45 deg relative: cumulative angle = 30+45 = 75 deg
        # After link2: prev + (80*cos75, 80*sin75, 0)
        a1 = math.radians(30)
        a2 = math.radians(30 + 45)
        expected_x = 100 * math.cos(a1) + 80 * math.cos(a2)
        expected_y = 100 * math.sin(a1) + 80 * math.sin(a2)
        pos = linkage.end_effector_position()
        np.testing.assert_allclose(pos, [expected_x, expected_y, 0], atol=1e-10)

    def test_joint_value_override(self):
        """Override joint value and check end-effector moves."""
        linkage = _simple_planar_linkage()
        pos0 = linkage.end_effector_position()
        pos45 = linkage.end_effector_position(joint_values={"Base": 45.0})
        assert not np.allclose(pos0, pos45)
        expected = 100 * np.array([math.cos(math.radians(45)),
                                    math.sin(math.radians(45)), 0])
        np.testing.assert_allclose(pos45, expected, atol=1e-10)

    def test_link_length_override(self):
        """Override link length and verify."""
        linkage = _simple_planar_linkage()
        pos = linkage.end_effector_position(link_lengths={"Arm": 200.0})
        np.testing.assert_allclose(pos, [200, 0, 0], atol=1e-10)

    def test_all_joint_positions(self):
        """All joint positions should include base and tip."""
        linkage = _simple_planar_linkage()
        positions = linkage.all_joint_positions()
        assert len(positions) == 2
        assert positions[0][0] == "Base"
        assert positions[1][0] == "Tip"
        np.testing.assert_allclose(positions[0][1], [0, 0, 0], atol=1e-10)
        np.testing.assert_allclose(positions[1][1], [100, 0, 0], atol=1e-10)

    def test_3d_spatial(self):
        """A link along Z after a 90-deg Y rotation should end up along X."""
        linkage = Linkage(name="3D")
        linkage.add_joint(Joint("Base", JointType.REVOLUTE_Y, nominal=90.0))
        linkage.add_link(Link("Arm", length=50.0, direction=(0, 0, 1)))
        linkage.add_joint(Joint("Tip", JointType.FIXED))

        pos = linkage.end_effector_position()
        # Rotating Z-axis 90 deg about Y -> Z maps to -X?
        # Actually: Ry(90) * [0,0,1] = [sin90, 0, cos90] = [1, 0, 0]
        # Wait: Ry(90) rotates X toward Z, so Z toward -X.
        # Let me check: Ry(90) @ [0,0,50] = [50*sin90, 0, 50*cos90] = [50, 0, 0]
        # Hmm, the rotation matrix: Ry = [[c,0,s],[0,1,0],[-s,0,c]]
        # Ry(90) @ [0,0,50] = [0*c+50*s, 0, 0*(-s)+50*c] = [50, 0, 0]
        np.testing.assert_allclose(pos, [50, 0, 0], atol=1e-10)


# ---------------------------------------------------------------------------
# Jacobian tests
# ---------------------------------------------------------------------------

class TestJacobian:
    def test_simple_link_length_jacobian(self):
        """dX/dL should be 1 for a horizontal link, dY/dL should be 0."""
        linkage = _simple_planar_linkage()
        J, names, _ = _compute_jacobian(linkage)
        # Find the link column
        link_idx = names.index("Arm")
        np.testing.assert_allclose(J[0, link_idx], 1.0, atol=1e-4)
        np.testing.assert_allclose(J[1, link_idx], 0.0, atol=1e-4)

    def test_simple_joint_angle_jacobian(self):
        """dY/d(theta) for a 100mm arm at theta=0 should be ~100 * pi/180."""
        linkage = _simple_planar_linkage()
        J, names, _ = _compute_jacobian(linkage)
        joint_idx = names.index("Base")
        # At theta=0, the arm is along X. Rotating by d_theta:
        # dX/dtheta = -L*sin(0)*pi/180 = 0
        # dY/dtheta = L*cos(0)*pi/180 = 100*pi/180 ≈ 1.7453
        expected_dy = 100.0 * math.pi / 180.0
        np.testing.assert_allclose(J[1, joint_idx], expected_dy, atol=0.01)
        np.testing.assert_allclose(J[0, joint_idx], 0.0, atol=0.01)

    def test_jacobian_shape(self):
        linkage = _two_bar_linkage()
        J, names, params = _compute_jacobian(linkage)
        assert J.shape == (3, len(names))
        # Elbow (tol) + Link1 (tol) + Link2 (tol) = 3 params
        assert len(names) == 3


# ---------------------------------------------------------------------------
# Analysis tests
# ---------------------------------------------------------------------------

class TestWorstCase:
    def test_has_tolerance(self):
        linkage = _simple_planar_linkage()
        r = linkage_worst_case(linkage)
        assert np.all(r.plus_tolerance >= 0)
        assert np.all(r.minus_tolerance >= 0)

    def test_range_brackets_nominal(self):
        linkage = _simple_planar_linkage()
        r = linkage_worst_case(linkage)
        np.testing.assert_array_less(r.position_min, r.nominal_position + 1e-10)
        np.testing.assert_array_less(r.nominal_position - 1e-10, r.position_max)

    def test_no_tolerance_zero_range(self):
        """A linkage with no tolerances should have zero tolerance range."""
        linkage = Linkage(name="NoTol")
        linkage.add_joint(Joint("Base", JointType.FIXED))
        linkage.add_link(Link("Arm", length=100.0, direction=(1, 0, 0)))
        linkage.add_joint(Joint("Tip", JointType.FIXED))
        r = linkage_worst_case(linkage)
        np.testing.assert_allclose(r.plus_tolerance, [0, 0, 0], atol=1e-10)
        np.testing.assert_allclose(r.minus_tolerance, [0, 0, 0], atol=1e-10)


class TestRSS:
    def test_rss_smaller_than_wc(self):
        """RSS tolerance should be smaller than worst-case."""
        linkage = _two_bar_linkage()
        wc = linkage_worst_case(linkage)
        rs = linkage_rss(linkage, sigma=3.0)
        # RSS tolerance should be <= WC tolerance per axis
        assert np.all(rs.plus_tolerance <= wc.plus_tolerance + 1e-10)

    def test_sigma_scaling(self):
        linkage = _two_bar_linkage()
        r3 = linkage_rss(linkage, sigma=3.0)
        r6 = linkage_rss(linkage, sigma=6.0)
        # 6-sigma should be 2x of 3-sigma
        np.testing.assert_allclose(r6.plus_tolerance, r3.plus_tolerance * 2.0, atol=1e-10)


class TestMonteCarlo:
    def test_mean_near_nominal(self):
        linkage = _simple_planar_linkage()
        r = linkage_monte_carlo(linkage, n_samples=50_000, seed=42)
        np.testing.assert_allclose(r.mc_mean, r.nominal_position, atol=0.05)

    def test_mc_range_within_wc(self):
        """MC range should generally fit within worst-case range.

        Note: WC uses a linearized Jacobian, so the actual nonlinear MC
        range can slightly exceed it for angular parameters. We use a
        generous margin to account for nonlinear effects.
        """
        linkage = _simple_planar_linkage()
        wc = linkage_worst_case(linkage)
        mc = linkage_monte_carlo(linkage, n_samples=50_000, seed=42)
        # MC range should be in the same ballpark as WC
        margin = 1.5  # generous margin for nonlinear effects
        assert np.all(mc.position_min >= wc.position_min - margin)
        assert np.all(mc.position_max <= wc.position_max + margin)

    def test_seed_reproducibility(self):
        linkage = _simple_planar_linkage()
        r1 = linkage_monte_carlo(linkage, n_samples=1_000, seed=99)
        r2 = linkage_monte_carlo(linkage, n_samples=1_000, seed=99)
        np.testing.assert_array_equal(r1.mc_samples, r2.mc_samples)

    def test_covariance_matrix(self):
        linkage = _simple_planar_linkage()
        r = linkage_monte_carlo(linkage, n_samples=50_000, seed=42)
        assert r.mc_cov.shape == (3, 3)
        # Covariance should be symmetric positive semi-definite
        np.testing.assert_allclose(r.mc_cov, r.mc_cov.T, atol=1e-12)
        eigenvalues = np.linalg.eigvalsh(r.mc_cov)
        assert np.all(eigenvalues >= -1e-10)


class TestAnalyzeLinkage:
    def test_all_methods(self):
        linkage = _simple_planar_linkage()
        results = analyze_linkage(linkage, mc_seed=42)
        assert "wc" in results
        assert "rss" in results
        assert "mc" in results

    def test_single_method(self):
        linkage = _simple_planar_linkage()
        results = analyze_linkage(linkage, methods=["wc"])
        assert "wc" in results
        assert "rss" not in results


# ---------------------------------------------------------------------------
# Serialization tests
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_save_load_roundtrip(self, tmp_path):
        linkage = _two_bar_linkage()
        path = str(tmp_path / "test_linkage.json")
        linkage.save(path)
        loaded = Linkage.load(path)

        assert loaded.name == linkage.name
        assert len(loaded.joints) == len(linkage.joints)
        assert len(loaded.links) == len(linkage.links)

        for orig, load in zip(linkage.joints, loaded.joints):
            assert orig.name == load.name
            assert orig.joint_type == load.joint_type

        for orig, load in zip(linkage.links, loaded.links):
            assert orig.name == load.name
            assert orig.length == load.length

    def test_loaded_fk_matches(self, tmp_path):
        linkage = _two_bar_linkage()
        path = str(tmp_path / "test_linkage.json")
        linkage.save(path)
        loaded = Linkage.load(path)

        pos_orig = linkage.end_effector_position()
        pos_loaded = loaded.end_effector_position()
        np.testing.assert_allclose(pos_orig, pos_loaded, atol=1e-10)


class TestParameterList:
    def test_parameter_count(self):
        linkage = _simple_planar_linkage()
        params = linkage.parameter_list()
        # Base has tolerance, Arm has tolerance, Tip has no tolerance
        assert len(params) == 2
        names = [p[0] for p in params]
        assert "Base" in names
        assert "Arm" in names
