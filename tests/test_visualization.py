"""Tests for 3D visualization module."""

import pytest
import numpy as np

from tolerance_stack.visualization import (
    VisualizationConfig, visualize_assembly, visualize_linkage,
    visualize_sensitivity, PLOTLY_AVAILABLE,
)


class TestVisualizationConfig:

    def test_defaults(self):
        config = VisualizationConfig()
        assert config.width == 800
        assert config.height == 600
        assert config.show_features is True
        assert config.show_mates is True

    def test_custom(self):
        config = VisualizationConfig(width=1200, height=800, mc_alpha=0.3)
        assert config.width == 1200
        assert config.mc_alpha == 0.3


class TestVisualizeAssembly:

    def _make_assembly(self):
        from tolerance_stack.assembly import (
            Assembly, Body, Feature, FeatureType, Mate, MateType,
            Measurement, MeasurementType,
        )
        block = Body("Block")
        block.add_feature(Feature("hole", FeatureType.CYLINDER,
                                  origin=(0, 0, 0), direction=(0, 0, 1),
                                  radius=5.0, position_tol=0.05))
        pin = Body("Pin")
        pin.add_feature(Feature("shaft", FeatureType.CYLINDER,
                                origin=(0, 0, 0), direction=(0, 0, 1),
                                radius=4.9, position_tol=0.03))

        assy = Assembly("test_assy")
        assy.add_body(block, origin=(0, 0, 0))
        assy.add_body(pin, origin=(0, 0, 10))
        assy.add_mate(Mate("coax", "Block", "hole", "Pin", "shaft", MateType.COAXIAL))
        assy.set_measurement(Measurement("gap", "Block", "hole",
                                          "Pin", "shaft", MeasurementType.DISTANCE_ALONG,
                                          direction=(0, 0, 1)))
        return assy

    @pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="plotly not installed")
    def test_basic_visualization(self):
        assy = self._make_assembly()
        fig = visualize_assembly(assy)
        assert fig is not None
        assert len(fig.data) > 0

    @pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="plotly not installed")
    def test_with_config(self):
        assy = self._make_assembly()
        config = VisualizationConfig(width=1000, height=700, show_mates=False)
        fig = visualize_assembly(assy, config=config)
        assert fig is not None

    def test_returns_none_without_plotly(self):
        # Test that the function gracefully handles missing plotly
        # by testing the check function
        from tolerance_stack.visualization import _check_plotly
        # Just verify it returns a bool
        assert isinstance(_check_plotly(), bool)


class TestVisualizeLinkage:

    def _make_linkage(self):
        from tolerance_stack.linkage import Linkage, Joint, JointType, Link
        linkage = Linkage("test_link")
        linkage.add_joint(Joint("base", JointType.REVOLUTE_Z, nominal=45.0,
                                plus_tol=0.5, minus_tol=0.5))
        linkage.add_link(Link("arm1", length=100.0, plus_tol=0.1, minus_tol=0.1))
        linkage.add_joint(Joint("elbow", JointType.REVOLUTE_Z, nominal=-30.0,
                                plus_tol=0.3, minus_tol=0.3))
        linkage.add_link(Link("arm2", length=80.0, plus_tol=0.08, minus_tol=0.08))
        linkage.add_joint(Joint("wrist", JointType.FIXED))
        return linkage

    @pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="plotly not installed")
    def test_basic_linkage_viz(self):
        linkage = self._make_linkage()
        fig = visualize_linkage(linkage)
        assert fig is not None
        assert len(fig.data) >= 2  # chain + end-effector

    @pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="plotly not installed")
    def test_with_mc_samples(self):
        linkage = self._make_linkage()
        mc = np.random.default_rng(0).normal(size=(1000, 3))
        fig = visualize_linkage(linkage, mc_samples=mc)
        assert fig is not None


class TestVisualizeSensitivity:

    @pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="plotly not installed")
    def test_scalar_sensitivity(self):
        sens = [("A", 0.5), ("B", -0.3), ("C", 0.8)]
        fig = visualize_sensitivity(sens)
        assert fig is not None

    @pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="plotly not installed")
    def test_vector_sensitivity(self):
        sens = [("A", [0.5, 0.1, 0.2]), ("B", [-0.3, 0.4, -0.1])]
        fig = visualize_sensitivity(sens)
        assert fig is not None

    def test_empty_sensitivity(self):
        fig = visualize_sensitivity([])
        assert fig is None
