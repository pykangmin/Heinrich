"""
레짐 정의 및 Track A/D 가중치 설정.

REGIME 환경변수로 현재 레짐 지정 (기본: "관세/무역전쟁").
미등록 레짐은 "정상" config로 폴백.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

SUPPORTED_REGIMES = [
    "관세/무역전쟁",
    "금리 상승",
    "리세션",
    "정상",
]

# 레짐별 Track A / Track D 가중치
REGIME_CONFIG: dict[str, dict] = {
    "관세/무역전쟁": {"w_track_a": 0.55, "w_track_d": 0.45},
    "금리 상승": {"w_track_a": 0.65, "w_track_d": 0.35},
    "리세션": {"w_track_a": 0.50, "w_track_d": 0.50},
    "정상": {"w_track_a": 0.60, "w_track_d": 0.40},
}

_DEFAULT_REGIME = "관세/무역전쟁"


def get_regime() -> str:
    """환경변수 REGIME 우선, 없으면 기본값 반환."""
    return os.getenv("REGIME", _DEFAULT_REGIME).strip()


def get_regime_config(regime: str) -> dict:
    """레짐 config 반환. 미등록 레짐이면 '정상' config로 폴백 후 WARNING."""
    if regime not in REGIME_CONFIG:
        logger.warning("미등록 레짐 '%s' → '정상' config 폴백", regime)
        return REGIME_CONFIG["정상"]
    return REGIME_CONFIG[regime]
