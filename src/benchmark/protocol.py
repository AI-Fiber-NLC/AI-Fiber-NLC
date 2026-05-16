# SPDX-License-Identifier: MIT
"""
Benchmark Validation Protocol for AI-Fiber-NLC.

Defines standard benchmark scenes (MVB-1/2/3), composite scoring formula,
and submission validation for the AI optical fiber nonlinear compensation project.
"""

from __future__ import annotations

import hashlib
import math
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


# ─────────────────────────────────────────────────────────────────────
# 1. Standard Benchmark Scene Definitions (MVB Series)
# ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SceneParams:
    """
    MVB (Minimum Viable Benchmark) scene parameters.

    MVB-1: Single-pol 16QAM, 800 km (initial validation)
    MVB-2: Dual-pol DP-16QAM, 800 km
    MVB-3: 64QAM + PCS, 800 km
    """
    name: str = "MVB-1"
    fiber_length_km: float = 80.0       # span length
    num_spans: int = 10                 # number of spans (total = 800 km)
    alpha_db_per_km: float = 0.2        # attenuation coefficient
    D_ps_per_nm_km: float = 16.0        # dispersion @ 1550 nm
    gamma_per_W_km: float = 1.3         # nonlinear coefficient
    pmd_ps_per_sqrt_km: float = 0.0     # PMD (0 = single-pol, 0.1 = DP)
    modulation: str = "16QAM"
    baud_rate_GBd: float = 32.0
    polarization: str = "single"        # "single" | "dual"
    pcs_enabled: bool = False           # probabilistic constellation shaping
    tx_power_range_dbm: Tuple[float, float] = (-3.0, 5.0)
    tx_power_points: int = 9
    n_symbols: int = 2 ** 16
    seed: int = 42
    fec_threshold_ber: float = 3.8e-3   # hard-decision FEC threshold


# ── Predefined scenes ──

MVB1 = SceneParams(
    name="MVB-1", fiber_length_km=80.0, num_spans=10,
    alpha_db_per_km=0.2, D_ps_per_nm_km=16.0, gamma_per_W_km=1.3,
    pmd_ps_per_sqrt_km=0.0, modulation="16QAM", baud_rate_GBd=32.0,
    polarization="single", pcs_enabled=False,
    tx_power_range_dbm=(-3.0, 5.0), tx_power_points=9,
    n_symbols=2**16, seed=42, fec_threshold_ber=3.8e-3,
)

MVB2 = SceneParams(
    name="MVB-2", fiber_length_km=80.0, num_spans=10,
    alpha_db_per_km=0.2, D_ps_per_nm_km=16.0, gamma_per_W_km=1.3,
    pmd_ps_per_sqrt_km=0.1, modulation="16QAM", baud_rate_GBd=32.0,
    polarization="dual", pcs_enabled=False,
    tx_power_range_dbm=(-3.0, 5.0), tx_power_points=9,
    n_symbols=2**16, seed=42, fec_threshold_ber=3.8e-3,
)

MVB3 = SceneParams(
    name="MVB-3", fiber_length_km=80.0, num_spans=10,
    alpha_db_per_km=0.2, D_ps_per_nm_km=16.0, gamma_per_W_km=1.3,
    pmd_ps_per_sqrt_km=0.0, modulation="64QAM", baud_rate_GBd=32.0,
    polarization="single", pcs_enabled=True,
    tx_power_range_dbm=(-3.0, 5.0), tx_power_points=9,
    n_symbols=2**16, seed=42, fec_threshold_ber=3.8e-3,
)

SCENES: Dict[str, SceneParams] = {
    "mvb1": MVB1,
    "mvb2": MVB2,
    "mvb3": MVB3,
}


# ─────────────────────────────────────────────────────────────────────
# 2. Composite Scoring Formula
# ─────────────────────────────────────────────────────────────────────

DEFAULT_EFFECT_WEIGHT: float = 0.7
DEFAULT_EFFICIENCY_WEIGHT: float = 0.3
FLOPS_MAX_MULTIPLIER: float = 100.0


def compute_normalized_q_factor(
    q_ai: float,
    q_baseline: float,
    q_dbp: float,
) -> float:
    """
    Normalized Q-factor improvement.

    Returns [0, 1.5]: 0 = no compensation, 1.0 = DBP level, >1.0 = beyond DBP.
    """
    if q_dbp <= q_baseline:
        return min(max((q_ai - q_baseline) / 1.0, 0.0), 1.5) if q_ai > q_baseline else 0.0
    ratio = (q_ai - q_baseline) / (q_dbp - q_baseline)
    return max(0.0, min(ratio, 1.5))


def compute_normalized_efficiency(
    flops_per_symbol: float,
    flops_dbp_per_symbol: float,
) -> float:
    """
    Normalized efficiency score.

    Returns [0, 1]: 1 = very efficient, 0 = computationally infeasible.
    """
    if flops_per_symbol <= 0 or flops_dbp_per_symbol <= 0:
        return 0.0
    ratio = flops_per_symbol / flops_dbp_per_symbol
    if ratio >= FLOPS_MAX_MULTIPLIER:
        return 0.0
    if ratio <= 1e-10:
        return 1.0
    log_max = math.log10(FLOPS_MAX_MULTIPLIER)
    e_norm = 1.0 - math.log10(ratio) / log_max
    return max(0.0, min(e_norm, 1.0))


def compute_composite_score(
    q_ai: float,
    q_baseline: float,
    q_dbp: float,
    flops_per_symbol: float,
    flops_dbp_per_symbol: float,
    effect_weight: float = DEFAULT_EFFECT_WEIGHT,
    efficiency_weight: float = DEFAULT_EFFICIENCY_WEIGHT,
) -> Dict[str, Any]:
    """
    Composite score = w_effect * Q_norm + w_efficiency * E_norm.
    """
    dq_norm = compute_normalized_q_factor(q_ai, q_baseline, q_dbp)
    e_norm = compute_normalized_efficiency(flops_per_symbol, flops_dbp_per_symbol)
    score = effect_weight * dq_norm + efficiency_weight * e_norm

    delta_q = q_ai - q_baseline
    flops_ratio = flops_per_symbol / flops_dbp_per_symbol if flops_dbp_per_symbol > 0 else float("inf")

    return {
        "composite_score": round(score, 4),
        "delta_q_db": round(delta_q, 3),
        "dq_norm": round(dq_norm, 4),
        "e_norm": round(e_norm, 4),
        "flops_ratio": round(flops_ratio, 2),
        "rank_category": _rank_label(score),
    }


def _rank_label(score: float) -> str:
    if score >= 1.0:
        return "S"
    if score >= 0.7:
        return "A"
    if score >= 0.4:
        return "B"
    if score >= 0.1:
        return "C"
    return "D"


# ─────────────────────────────────────────────────────────────────────
# 3. Submission Validation
# ─────────────────────────────────────────────────────────────────────

class ValidationError(Exception):
    """Submission does not meet protocol requirements."""
    pass


def validate_submission(
    result: Dict[str, Any],
    scene_name: str = "mvb1",
) -> List[str]:
    """
    Validate a benchmark submission.

    Checks: required fields, Q-factor range, FLOPs positive,
    at least 3 power points, seed consistency.

    Returns: list of warnings (empty = pass).
    Raises: ValidationError for critical failures.
    """
    warnings: List[str] = []

    required = [
        "q_factor", "q_baseline", "q_dbp",
        "flops_per_symbol", "flops_dbp_per_symbol",
        "power_points", "scene_name", "seed",
        "model_name", "author",
    ]
    missing = [f for f in required if f not in result]
    if missing:
        raise ValidationError(f"Missing required fields: {', '.join(missing)}")

    q = result["q_factor"]
    if not (0 < q < 30):
        raise ValidationError(f"Q-factor {q} dB out of valid range (0-30)")

    qb = result["q_baseline"]
    if not (0 < qb < 30):
        raise ValidationError(f"Baseline Q-factor {qb} dB out of valid range")

    qd = result["q_dbp"]
    if not (0 < qd < 30):
        raise ValidationError(f"DBP Q-factor {qd} dB out of valid range")

    if q < qb:
        warnings.append("Q-factor below baseline — model provides no compensation")

    if result["flops_per_symbol"] <= 0:
        raise ValidationError("FLOPs/symbol must be positive")
    if result["flops_dbp_per_symbol"] <= 0:
        raise ValidationError("DBP FLOPs/symbol must be positive")

    ppts = result["power_points"]
    if not isinstance(ppts, list) or len(ppts) < 3:
        raise ValidationError("At least 3 power points required")

    expected_seed = SCENES.get(scene_name)
    if expected_seed and result["seed"] != expected_seed.seed:
        warnings.append(f"Seed mismatch: got {result['seed']}, expected {expected_seed.seed}")

    if result["scene_name"] != scene_name:
        warnings.append(f"Scene name mismatch: got '{result['scene_name']}', expected '{scene_name}'")

    return warnings


# ─────────────────────────────────────────────────────────────────────
# 4. Contributor Identity Signing (Ed25519 → batch on-chain transition)
# ─────────────────────────────────────────────────────────────────────

def generate_contribution_hash(
    model_weights_path: str,
    author: str,
    timestamp: str,
) -> str:
    """
    Generate contribution hash for later on-chain attestation.
    SHA-256 over model weights file + contributor info.
    """
    hasher = hashlib.sha256()
    with open(model_weights_path, "rb") as f:
        hasher.update(f.read())
    hasher.update(author.encode("utf-8"))
    hasher.update(timestamp.encode("utf-8"))
    return hasher.hexdigest()


# ─────────────────────────────────────────────────────────────────────
# 5. Scene config → YAML serialization
# ─────────────────────────────────────────────────────────────────────

def scene_to_yaml(scene: SceneParams) -> str:
    """Serialize scene config to YAML string (no external YAML dependency)."""
    lines = [
        f"# AI-Fiber-NLC Benchmark Scene: {scene.name}",
        "# MIT License",
        "",
        "fiber:",
        f"  length_km: {scene.fiber_length_km}",
        f"  num_spans: {scene.num_spans}",
        f"  alpha_db_per_km: {scene.alpha_db_per_km}",
        f"  D_ps_per_nm_km: {scene.D_ps_per_nm_km}",
        f"  gamma_per_W_km: {scene.gamma_per_W_km}",
        f"  pmd_ps_per_sqrt_km: {scene.pmd_ps_per_sqrt_km}",
        "",
        "signal:",
        f"  modulation: \"{scene.modulation}\"",
        f"  baud_rate_GBd: {scene.baud_rate_GBd}",
        f"  polarization: \"{scene.polarization}\"",
        f"  pcs_enabled: {'true' if scene.pcs_enabled else 'false'}",
        "",
        "test:",
        f"  tx_power_range_dbm: [{scene.tx_power_range_dbm[0]}, {scene.tx_power_range_dbm[1]}]",
        f"  tx_power_points: {scene.tx_power_points}",
        f"  n_symbols: {scene.n_symbols}",
        f"  seed: {scene.seed}",
        f"  fec_threshold_ber: {scene.fec_threshold_ber}",
    ]
    return "\n".join(lines) + "\n"


# ─────────────────────────────────────────────────────────────────────
# CLI: quick validation of a submitted JSON result
# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    """CLI entry — validate a benchmark result file."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python protocol.py <result.json> [scene_name]")
        sys.exit(1)

    result_path = sys.argv[1]
    scene_name = sys.argv[2] if len(sys.argv) > 2 else "mvb1"

    try:
        with open(result_path, "r") as f:
            result = json.load(f)
    except FileNotFoundError:
        print(f"Error: file not found {result_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON {e}")
        sys.exit(1)

    try:
        warns = validate_submission(result, scene_name)
    except ValidationError as e:
        print(f"FAIL: {e}")
        sys.exit(1)

    score = compute_composite_score(
        q_ai=result["q_factor"],
        q_baseline=result["q_baseline"],
        q_dbp=result["q_dbp"],
        flops_per_symbol=result["flops_per_symbol"],
        flops_dbp_per_symbol=result["flops_dbp_per_symbol"],
    )

    print(f"Scene: {result.get('scene_name', scene_name).upper()}")
    print(f"Model: {result.get('model_name', 'N/A')}")
    print(f"Author: {result.get('author', 'N/A')}")
    print("-" * 40)
    print(f"Composite Score: {score['composite_score']}")
    print(f"Rank: {score['rank_category']}")
    print(f"Q-factor improvement: +{score['delta_q_db']} dB")
    print(f"FLOPs ratio: {score['flops_ratio']}x DBP")

    if warns:
        print(f"\nWarnings ({len(warns)}):")
        for w in warns:
            print(f"  - {w}")
    else:
        print("\nPASS — no warnings")


if __name__ == "__main__":
    main()
