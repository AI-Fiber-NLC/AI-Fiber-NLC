# SPDX-License-Identifier: MIT
"""Unit tests for the Benchmark Validation Protocol."""

import pytest

from src.benchmark.protocol import (
    SceneParams,
    MVB1,
    MVB2,
    MVB3,
    SCENES,
    ValidationError,
    compute_composite_score,
    compute_normalized_q_factor,
    compute_normalized_efficiency,
    validate_submission,
    scene_to_yaml,
)


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────


@pytest.fixture
def valid_result():
    """A minimal valid benchmark submission."""
    return {
        "q_factor": 12.5,
        "q_baseline": 8.0,
        "q_dbp": 13.2,
        "flops_per_symbol": 1200,
        "flops_dbp_per_symbol": 45000,
        "power_points": [
            {"tx_power_dbm": -3.0, "q_factor": 11.5},
            {"tx_power_dbm": 1.0, "q_factor": 12.5},
            {"tx_power_dbm": 5.0, "q_factor": 10.8},
        ],
        "scene_name": "mvb1",
        "seed": 42,
        "model_name": "test-mlp",
        "author": "test-user",
    }


# ──────────────────────────────────────────────
# Scene Definitions
# ──────────────────────────────────────────────


class TestScenes:
    def test_mvb1_single_pol(self):
        assert MVB1.polarization == "single"
        assert MVB1.modulation == "16QAM"
        assert MVB1.num_spans == 10
        assert MVB1.fiber_length_km == 80.0
        assert MVB1.pmd_ps_per_sqrt_km == 0.0

    def test_mvb2_dual_pol_pmd(self):
        assert MVB2.polarization == "dual"
        assert MVB2.pmd_ps_per_sqrt_km == 0.1
        assert MVB2.modulation == "16QAM"

    def test_mvb3_64qam_pcs(self):
        assert MVB3.modulation == "64QAM"
        assert MVB3.pcs_enabled is True
        assert MVB3.polarization == "single"

    def test_scene_registry(self):
        assert len(SCENES) == 3
        assert set(SCENES.keys()) == {"mvb1", "mvb2", "mvb3"}


# ──────────────────────────────────────────────
# Normalized Q-factor
# ──────────────────────────────────────────────


class TestNormalizedQFactor:
    def test_no_improvement(self):
        r = compute_normalized_q_factor(8.0, 8.0, 13.0)
        assert r == pytest.approx(0.0)

    def test_halfway(self):
        r = compute_normalized_q_factor(10.5, 8.0, 13.0)
        assert r == pytest.approx(0.5)

    def test_equals_dbp(self):
        r = compute_normalized_q_factor(13.0, 8.0, 13.0)
        assert r == pytest.approx(1.0)

    def test_exceeds_dbp_capped(self):
        r = compute_normalized_q_factor(18.0, 8.0, 13.0)
        assert r == pytest.approx(1.5)

    def test_worse_than_baseline(self):
        r = compute_normalized_q_factor(6.0, 8.0, 13.0)
        assert r == pytest.approx(0.0)

    def test_dbp_worse_than_baseline(self):
        r = compute_normalized_q_factor(9.0, 8.0, 7.5)
        assert r >= 0


# ──────────────────────────────────────────────
# Normalized Efficiency
# ──────────────────────────────────────────────


class TestNormalizedEfficiency:
    def test_same_as_dbp(self):
        r = compute_normalized_efficiency(45000, 45000)
        assert r == pytest.approx(1.0)

    def test_ten_times_better(self):
        r = compute_normalized_efficiency(4500, 45000)
        assert r == pytest.approx(1.0)

    def test_much_better(self):
        r = compute_normalized_efficiency(100, 45000)
        assert r > 0.75

    def test_at_max_multiplier(self):
        r = compute_normalized_efficiency(4500000, 45000)
        assert r == pytest.approx(0.0)

    def test_beyond_max(self):
        r = compute_normalized_efficiency(9000000, 45000)
        assert r == pytest.approx(0.0)

    def test_zero_flops(self):
        r = compute_normalized_efficiency(0, 45000)
        assert r == pytest.approx(0.0)


# ──────────────────────────────────────────────
# Composite Score
# ──────────────────────────────────────────────


class TestCompositeScore:
    def test_ideal_model(self):
        s = compute_composite_score(
            q_ai=13.0, q_baseline=8.0, q_dbp=13.0,
            flops_per_symbol=450, flops_dbp_per_symbol=45000,
        )
        assert s["dq_norm"] == pytest.approx(1.0)
        assert s["e_norm"] > 0.5
        assert s["composite_score"] > 0.85
        assert s["rank_category"] == "S"

    def test_no_improvement(self):
        s = compute_composite_score(
            q_ai=8.0, q_baseline=8.0, q_dbp=13.0,
            flops_per_symbol=100, flops_dbp_per_symbol=45000,
        )
        assert s["dq_norm"] == pytest.approx(0.0)
        assert s["composite_score"] < 0.35

    def test_dbp_level_same_cost(self):
        s = compute_composite_score(
            q_ai=13.0, q_baseline=8.0, q_dbp=13.0,
            flops_per_symbol=45000, flops_dbp_per_symbol=45000,
        )
        assert s["dq_norm"] == pytest.approx(1.0)
        assert s["e_norm"] == pytest.approx(1.0)
        assert s["composite_score"] == pytest.approx(1.0)

    def test_output_keys(self):
        s = compute_composite_score(
            q_ai=12.5, q_baseline=8.0, q_dbp=13.2,
            flops_per_symbol=1200, flops_dbp_per_symbol=45000,
        )
        for key in ["composite_score", "delta_q_db", "dq_norm", "e_norm",
                     "flops_ratio", "rank_category"]:
            assert key in s

    def test_custom_weights(self):
        s1 = compute_composite_score(12.0, 8.0, 13.0, 1200, 45000)
        s2 = compute_composite_score(12.0, 8.0, 13.0, 1200, 45000,
                                     effect_weight=0.5, efficiency_weight=0.5)
        assert s1["composite_score"] != s2["composite_score"]


# ──────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────


class TestValidation:
    def test_valid_passes(self, valid_result):
        warns = validate_submission(valid_result, "mvb1")
        assert warns == []

    def test_missing_fields(self):
        with pytest.raises(ValidationError, match="Missing required fields"):
            validate_submission({"q_factor": 10.0})

    def test_q_factor_out_of_range_high(self, valid_result):
        valid_result["q_factor"] = 35.0
        with pytest.raises(ValidationError, match="out of valid range"):
            validate_submission(valid_result)

    def test_q_factor_out_of_range_low(self, valid_result):
        valid_result["q_factor"] = -1.0
        with pytest.raises(ValidationError, match="out of valid range"):
            validate_submission(valid_result)

    def test_flops_negative(self, valid_result):
        valid_result["flops_per_symbol"] = -100
        with pytest.raises(ValidationError, match="positive"):
            validate_submission(valid_result)

    def test_too_few_power_points(self, valid_result):
        valid_result["power_points"] = [
            {"tx_power_dbm": 0.0, "q_factor": 12.0},
        ]
        with pytest.raises(ValidationError, match="At least 3"):
            validate_submission(valid_result)

    def test_q_below_baseline_warns(self, valid_result):
        valid_result["q_factor"] = 7.0
        warns = validate_submission(valid_result, "mvb1")
        assert any("no compensation" in w for w in warns)

    def test_seed_mismatch_warns(self, valid_result):
        valid_result["seed"] = 99
        warns = validate_submission(valid_result, "mvb1")
        assert any("Seed mismatch" in w for w in warns)


# ──────────────────────────────────────────────
# YAML Serialization
# ──────────────────────────────────────────────


class TestSceneYaml:
    def test_contains_fiber_params(self):
        y = scene_to_yaml(MVB1)
        assert "length_km: 80.0" in y
        assert "alpha_db_per_km: 0.2" in y
        assert "D_ps_per_nm_km: 16.0" in y
        assert "gamma_per_W_km: 1.3" in y

    def test_contains_signal_params(self):
        y = scene_to_yaml(MVB1)
        assert "modulation" in y
        assert "baud_rate_GBd: 32.0" in y

    def test_mvb2_has_pmd(self):
        y = scene_to_yaml(MVB2)
        assert "pmd_ps_per_sqrt_km: 0.1" in y

    def test_mvb3_has_pcs(self):
        y = scene_to_yaml(MVB3)
        assert "pcs_enabled: true" in y
        assert "64QAM" in y
