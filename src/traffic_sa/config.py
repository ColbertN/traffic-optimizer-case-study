from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# Anchor all output folders relative to the case-study project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_RESULTS_DIR = PROJECT_ROOT / "model_results"
MODEL_RESULTS_IMAGE_DIR = MODEL_RESULTS_DIR / "plots"


# Each intersection is simulated in four traffic directions.
DIRECTIONS = ("inbound", "outbound", "cross_north_south", "cross_east_west")


@dataclass(frozen=True)
class IntersectionConfig:
    """Small Gauteng-like corridor node used by the simulator."""

    intersection_id: int
    name: str
    corridor: str
    area_type: str
    base_volume: float
    lanes: int
    failure_risk: float


# These nodes represent a small Gauteng commuter corridor from townships to business hubs.
INTERSECTIONS = (
    IntersectionConfig(
        0,
        "Soweto feeder to JHB CBD",
        "M1/CBD",
        "residential_feeder",
        75,
        2,
        0.10,
    ),
    IntersectionConfig(
        1,
        "JHB CBD interchange",
        "M1/CBD",
        "business_core",
        95,
        3,
        0.12,
    ),
    IntersectionConfig(
        2,
        "Sandton/Grayston node",
        "Sandton",
        "business_core",
        105,
        3,
        0.13,
    ),
    IntersectionConfig(
        3,
        "Midrand N1 corridor",
        "Midrand/Pretoria",
        "commuter_corridor",
        90,
        3,
        0.11,
    ),
)


# These factors approximate how much road capacity remains under each robot status.
STATUS_FACTORS = {
    "working": 1.00,
    "pointsman": 0.68,
    "failed": 0.34,
}


# Friendly names used on charts and presentation outputs.
POLICY_LABELS = {
    "fixed": "Fixed robots",
    "adaptive": "Adaptive robots",
    "failure_aware": "Failure-aware adaptive",
}
