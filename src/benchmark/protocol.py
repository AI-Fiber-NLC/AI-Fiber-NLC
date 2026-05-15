# SPDX-License-Identifier: MIT
# AI-Fiber-NLC — Benchmark Validation Protocol
# 光纖非線性補償 AI 框架 — 標準 Benchmark 場景定義與評分協議


from __future__ import annotations

import math
import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────
# 1. 標準 Benchmark 場景配置
# ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SceneParams:
    """
    MVB（Minimum Viable Benchmark）場景參數。

    MVB-1：單偏振 16QAM，800km（初期驗證用）
    MVB-2：雙偏振 DP-16QAM，800km
    MVB-3：64QAM + PCS，800km
    """
    name: str = "MVB-1"               # 場景名稱
    # 光纖物理參數（ITU-T G.652 標準單模光纖）
    fiber_length_km: float = 80.0       # 單跨距長度
    num_spans: int = 10                 # 跨距數量（總長 = 800 km）
    alpha_db_per_km: float = 0.2        # 損耗係數
    D_ps_per_nm_km: float = 16.0        # 色散參數 @1550nm
    gamma_per_W_km: float = 1.3         # 非線性係數

    # PMD（偏振模色散）— MVB-1/3 為 0，MVB-2 啟用
    pmd_ps_per_sqrt_km: float = 0.0

    # 信號參數
    modulation: str = "16QAM"
    baud_rate_GBd: float = 32.0
    polarization: str = "single"        # "single" | "dual"
    pcs_enabled: bool = False           # 概率星座整形

    # 測試配置
    tx_power_range_dbm: Tuple[float, float] = (-3.0, 5.0)  # 發射功率掃描範圍
    tx_power_points: int = 9            # 掃描點數（均勻分佈）
    n_symbols: int = 2 ** 16            # 符號數量
    seed: int = 42                      # 固定隨機種子（可驗證性）

    # FEC 閾值
    fec_threshold_ber: float = 3.8e-3   # 標準硬決策 FEC


# ── 預定義場景 ──

MVB1 = SceneParams(
    name="MVB-1",
    fiber_length_km=80.0,
    num_spans=10,
    alpha_db_per_km=0.2,
    D_ps_per_nm_km=16.0,
    gamma_per_W_km=1.3,
    pmd_ps_per_sqrt_km=0.0,
    modulation="16QAM",
    baud_rate_GBd=32.0,
    polarization="single",
    pcs_enabled=False,
    tx_power_range_dbm=(-3.0, 5.0),
    tx_power_points=9,
    n_symbols=2 ** 16,
    seed=42,
    fec_threshold_ber=3.8e-3,
)

MVB2 = SceneParams(
    name="MVB-2",
    fiber_length_km=80.0,
    num_spans=10,
    alpha_db_per_km=0.2,
    D_ps_per_nm_km=16.0,
    gamma_per_W_km=1.3,
    pmd_ps_per_sqrt_km=0.1,
    modulation="16QAM",
    baud_rate_GBd=32.0,
    polarization="dual",
    pcs_enabled=False,
    tx_power_range_dbm=(-3.0, 5.0),
    tx_power_points=9,
    n_symbols=2 ** 16,
    seed=42,
    fec_threshold_ber=3.8e-3,
)

MVB3 = SceneParams(
    name="MVB-3",
    fiber_length_km=80.0,
    num_spans=10,
    alpha_db_per_km=0.2,
    D_ps_per_nm_km=16.0,
    gamma_per_W_km=1.3,
    pmd_ps_per_sqrt_km=0.0,
    modulation="64QAM",
    baud_rate_GBd=32.0,
    polarization="single",
    pcs_enabled=True,
    tx_power_range_dbm=(-3.0, 5.0),
    tx_power_points=9,
    n_symbols=2 ** 16,
    seed=42,
    fec_threshold_ber=3.8e-3,
)

SCENES: Dict[str, SceneParams] = {
    "mvb1": MVB1,
    "mvb2": MVB2,
    "mvb3": MVB3,
}


# ─────────────────────────────────────────────────────────────────────
# 2. 複合評分公式
# ─────────────────────────────────────────────────────────────────────

# 初期權重：效果 70% / 效率 30%
DEFAULT_EFFECT_WEIGHT: float = 0.7
DEFAULT_EFFICIENCY_WEIGHT: float = 0.3

# FLOPs 不可接受上限（相對於 DBP）
FLOPS_MAX_MULTIPLIER: float = 100.0


def compute_normalized_q_factor(
    q_ai: float,
    q_baseline: float,
    q_dbp: float,
) -> float:
    """
    歸一化 Q-factor 提升。

    ΔQ_norm = (Q_ai - Q_baseline) / (Q_dbp - Q_baseline)

    範圍：[0, 1.5]
    - 0 = 無補償（等同 baseline）
    - 1.0 = 達到 DBP 水平
    - >1.0 = 超越 DBP（上限 1.5）
    """
    if q_dbp <= q_baseline:
        # DBP 本身冇提升 — 退化處理
        if q_ai > q_baseline:
            return min((q_ai - q_baseline) / 1.0, 1.5)
        return 0.0

    delta_q = q_ai - q_baseline
    delta_q_max = q_dbp - q_baseline
    ratio = delta_q / delta_q_max

    return max(0.0, min(ratio, 1.5))


def compute_normalized_efficiency(
    flops_per_symbol: float,
    flops_dbp_per_symbol: float,
) -> float:
    """
    歸一化效率評分。

    E_norm = 1 - log₁₀(FLOPs_ai / FLOPs_dbp) / log₁₀(FLOPS_MAX)

    範圍：[0, 1]
    - 1 = 極高效（FLOPs 遠低於 DBP）
    - 0 = 計算量爆炸（≥ FLOPS_MAX_MULTIPLIER × DBP）
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
    計算複合評分。

    Score = w_effect × ΔQ_norm + w_efficiency × E_norm

    返回：
    - composite_score: 總分 [0, 1.35]（理論上限）
    - delta_q_db: Q-factor 絕對提升（dB）
    - flops_ratio: 相對於 DBP 嘅計算量倍數
    - rank_category: 分級標籤
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
        return "S"   # 超越 DBP
    if score >= 0.7:
        return "A"   # 接近 DBP
    if score >= 0.4:
        return "B"   # 部分補償
    if score >= 0.1:
        return "C"   # 微弱補償
    return "D"       # 無效


# ─────────────────────────────────────────────────────────────────────
# 3. 結果驗證
# ─────────────────────────────────────────────────────────────────────

class ValidationError(Exception):
    """提交結果唔符合協議要求."""
    pass


def validate_submission(
    result: Dict[str, Any],
    scene_name: str = "mvb1",
) -> List[str]:
    """
    驗證 Benchmark 提交結果。

    檢查項：
    1. 必需欄位齊全
    2. Q-factor 合理範圍
    3. FLOPs 正數
    4. 至少 3 個功率點嘅結果
    5. 隨機 seed 一致

    返回：警告列表（空列表 = 通過）
    異常：ValidationError（嚴重錯誤，直接拒絕）
    """
    warnings: List[str] = []

    # ── 必需欄位 ──
    required = [
        "q_factor", "q_baseline", "q_dbp",
        "flops_per_symbol", "flops_dbp_per_symbol",
        "power_points", "scene_name", "seed",
        "model_name", "author",
    ]
    missing = [f for f in required if f not in result]
    if missing:
        raise ValidationError(f"缺少必需欄位: {', '.join(missing)}")

    # ── Q-factor 合理性 ──
    q = result["q_factor"]
    if not (0 < q < 30):
        raise ValidationError(f"Q-factor {q} dB 超出合理範圍 (0-30)")

    qb = result["q_baseline"]
    if not (0 < qb < 30):
        raise ValidationError(f"Baseline Q-factor {qb} dB 超出合理範圍")

    qd = result["q_dbp"]
    if not (0 < qd < 30):
        raise ValidationError(f"DBP Q-factor {qd} dB 超出合理範圍")

    if q < qb:
        warnings.append("Q-factor 低過 baseline，模型冇補償效果")

    # ── FLOPs 正數 ──
    if result["flops_per_symbol"] <= 0:
        raise ValidationError("FLOPs/symbol 必須為正數")
    if result["flops_dbp_per_symbol"] <= 0:
        raise ValidationError("DBP FLOPs/symbol 必須為正數")

    # ── 功率點覆蓋 ──
    ppts = result["power_points"]
    if not isinstance(ppts, list) or len(ppts) < 3:
        raise ValidationError("至少需要 3 個功率點嘅測試結果")

    # ── Seed 驗證 ──
    expected_seed = SCENES.get(scene_name)
    if expected_seed and result["seed"] != expected_seed.seed:
        warnings.append(
            f"Seed 唔匹配：提交={result['seed']}, "
            f"預期={expected_seed.seed}"
        )

    # ── 場景名稱 ──
    if result["scene_name"] != scene_name:
        warnings.append(
            f"場景名稱唔匹配：提交='{result['scene_name']}', "
            f"預期='{scene_name}'"
        )

    return warnings


# ─────────────────────────────────────────────────────────────────────
# 4. 貢獻者身份簽名工具（Ed25519 → 批量上鏈過渡方案）
# ─────────────────────────────────────────────────────────────────────

def generate_contribution_hash(
    model_weights_path: str,
    author: str,
    timestamp: str,
) -> str:
    """
    生成貢獻哈希（用於後續鏈上存證）。

    使用 SHA-256 對模型權重文件 + 貢獻者信息計算哈希。
    Phase 0-1：存儲於 GitHub PR 中。
    Phase 1+：批量上鏈（Polygon）。
    """
    hasher = hashlib.sha256()

    # 文件內容哈希
    with open(model_weights_path, "rb") as f:
        hasher.update(f.read())

    # 貢獻者信息
    hasher.update(author.encode("utf-8"))
    hasher.update(timestamp.encode("utf-8"))

    return hasher.hexdigest()


# ─────────────────────────────────────────────────────────────────────
# 5. 場景配置序列化為 YAML
# ─────────────────────────────────────────────────────────────────────

def scene_to_yaml(scene: SceneParams) -> str:
    """將場景配置轉為 YAML 字符串（不依賴外部 YAML 庫）."""
    lines = [
        f"# AI-Fiber-NLC Benchmark Scene: {scene.name}",
        f"# MIT License",
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
# CLI：快速驗證提交的 JSON 結果
# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    """CLI 入口 — 驗證 benchmark 結果文件."""
    import sys

    if len(sys.argv) < 2:
        print("用法: python protocol.py <result.json> [scene_name]")
        print("  result.json  : Benchmark 提交結果文件")
        print("  scene_name   : 場景名稱 (mvb1/mvb2/mvb3, 預設 mvb1)")
        sys.exit(1)

    result_path = sys.argv[1]
    scene_name = sys.argv[2] if len(sys.argv) > 2 else "mvb1"

    try:
        with open(result_path, "r") as f:
            result = json.load(f)
    except FileNotFoundError:
        print(f"錯誤：文件唔存在 {result_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"錯誤：JSON 格式唔啱 {e}")
        sys.exit(1)

    try:
        warns = validate_submission(result, scene_name)
    except ValidationError as e:
        print(f"❌ 驗證失敗: {e}")
        sys.exit(1)

    score = compute_composite_score(
        q_ai=result["q_factor"],
        q_baseline=result["q_baseline"],
        q_dbp=result["q_dbp"],
        flops_per_symbol=result["flops_per_symbol"],
        flops_dbp_per_symbol=result["flops_dbp_per_symbol"],
    )

    print(f"場景: {result.get('scene_name', scene_name).upper()}")
    print(f"模型: {result.get('model_name', 'N/A')}")
    print(f"作者: {result.get('author', 'N/A')}")
    print(f"━" * 40)
    print(f"複合評分: {score['composite_score']}")
    print(f"分級: {score['rank_category']}")
    print(f"Q-factor 提升: +{score['delta_q_db']} dB")
    print(f"計算量比率: {score['flops_ratio']}× DBP")

    if warns:
        print(f"\n⚠️ 警告 ({len(warns)}):")
        for w in warns:
            print(f"  - {w}")
    else:
        print("\n✅ 通過 — 無警告")


if __name__ == "__main__":
    main()
