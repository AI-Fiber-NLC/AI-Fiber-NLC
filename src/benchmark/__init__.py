# SPDX-License-Identifier: MIT
"""Benchmark validation protocol."""

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
    generate_contribution_hash,
)

__all__ = [
    "SceneParams",
    "MVB1", "MVB2", "MVB3",
    "SCENES",
    "ValidationError",
    "compute_composite_score",
    "compute_normalized_q_factor",
    "compute_normalized_efficiency",
    "validate_submission",
    "scene_to_yaml",
    "generate_contribution_hash",
]
