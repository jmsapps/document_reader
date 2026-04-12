import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SEMANTIC_RUNS_DIR = Path("local_documents/semantic_deviation_runs")
REPORTS_ROOT = Path("reports/test_semantic_deviation")


class SnapshotRecord(TypedDict):
    record_id: str
    source_name: str
    figure_id: str
    page_number: int | None
    markdown: str
    vector: list[float]


class SnapshotPayload(TypedDict):
    snapshot_file: str
    generated_at: str
    record_count: int
    records: list[SnapshotRecord]


class DistanceRecord(TypedDict):
    key: str
    distance: float


class RunDeviationRecord(TypedDict):
    snapshot_file: str
    distance: float


class PerRecordDeviation(TypedDict):
    record_id: str
    run_count: int
    average_deviation: float
    distances_from_average: list[RunDeviationRecord]


class RunLevelReport(TypedDict):
    run_vector_dimensions: int
    run_centroid: list[float]
    run_distances_from_average: list[RunDeviationRecord]
    average_run_semantic_deviation: float


class DeviationReport(TypedDict):
    snapshot_count: int
    run_vector_dimensions: int
    run_centroid: list[float]
    run_distances_from_average: list[RunDeviationRecord]
    average_run_semantic_deviation: float
    per_record_deviation: list[PerRecordDeviation]


def _load_semantic_snapshots(snapshot_dir: Path) -> list[SnapshotPayload]:
    if not snapshot_dir.exists():
        raise FileNotFoundError(
            f"Semantic snapshot directory not found: {snapshot_dir}"
        )

    payloads: list[SnapshotPayload] = []
    for path in sorted(snapshot_dir.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        records = raw.get("records") or []
        if not isinstance(records, list) or not records:
            continue
        payloads.append(
            {
                "snapshot_file": path.name,
                "generated_at": str(raw.get("generated_at") or ""),
                "record_count": len(records),
                "records": [
                    {
                        "record_id": str(record.get("record_id") or ""),
                        "source_name": str(record.get("source_name") or ""),
                        "figure_id": str(record.get("figure_id") or ""),
                        "page_number": record.get("page_number"),
                        "markdown": str(record.get("markdown") or ""),
                        "vector": [
                            float(value) for value in (record.get("vector") or [])
                        ],
                    }
                    for record in records
                    if record.get("record_id") and record.get("vector")
                ],
            }
        )

    if not payloads:
        raise ValueError(f"No semantic run snapshots found in {snapshot_dir}")
    return payloads


def _mean_run_vector(records: list[SnapshotRecord]) -> list[float]:
    vectors = np.array([record["vector"] for record in records], dtype=float)
    return np.mean(vectors, axis=0).tolist()


def _compute_run_level_report(payloads: list[SnapshotPayload]) -> RunLevelReport:
    run_vectors = np.array(
        [_mean_run_vector(payload["records"]) for payload in payloads],
        dtype=float,
    )
    centroid = np.mean(run_vectors, axis=0)
    distances = np.linalg.norm(run_vectors - centroid, axis=1)

    return {
        "run_vector_dimensions": int(run_vectors.shape[1]),
        "run_centroid": centroid.tolist(),
        "run_distances_from_average": [
            {
                "snapshot_file": payloads[index]["snapshot_file"],
                "distance": float(distance),
            }
            for index, distance in enumerate(distances)
        ],
        "average_run_semantic_deviation": float(np.mean(distances)),
    }


def _compute_per_record_report(
    payloads: list[SnapshotPayload],
) -> list[PerRecordDeviation]:
    grouped: dict[str, list[tuple[str, list[float]]]] = {}
    for payload in payloads:
        snapshot_file = payload["snapshot_file"]
        for record in payload["records"]:
            grouped.setdefault(record["record_id"], []).append(
                (snapshot_file, record["vector"])
            )

    report: list[PerRecordDeviation] = []
    for record_id in sorted(grouped):
        entries = grouped[record_id]
        vectors = np.array([vector for _, vector in entries], dtype=float)
        centroid = np.mean(vectors, axis=0)
        distances = np.linalg.norm(vectors - centroid, axis=1)
        report.append(
            {
                "record_id": record_id,
                "run_count": len(entries),
                "average_deviation": float(np.mean(distances)),
                "distances_from_average": [
                    {
                        "snapshot_file": entries[index][0],
                        "distance": float(distance),
                    }
                    for index, distance in enumerate(distances)
                ],
            }
        )
    return report


def test_semantic_deviation() -> None:
    payloads = _load_semantic_snapshots(SEMANTIC_RUNS_DIR)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    report_dir = REPORTS_ROOT / timestamp
    report_dir.mkdir(parents=True, exist_ok=True)
    runs_output_path = report_dir / "semantic_run_vectors.json"
    deviation_output_path = report_dir / "semantic_deviation_report.json"

    runs_output_path.write_text(json.dumps(payloads, indent=2), encoding="utf-8")

    run_level_report = _compute_run_level_report(payloads)
    per_record_report = _compute_per_record_report(payloads)
    deviation_report: DeviationReport = {
        "snapshot_count": len(payloads),
        "run_vector_dimensions": run_level_report["run_vector_dimensions"],
        "run_centroid": run_level_report["run_centroid"],
        "run_distances_from_average": run_level_report["run_distances_from_average"],
        "average_run_semantic_deviation": run_level_report[
            "average_run_semantic_deviation"
        ],
        "per_record_deviation": per_record_report,
    }
    deviation_output_path.write_text(
        json.dumps(deviation_report, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote run snapshots to: {runs_output_path}")
    print(f"Wrote deviation report to: {deviation_output_path}")
    print(
        f"Run distances from average: {deviation_report['run_distances_from_average']}"
    )
    print(
        "Average Run Semantic Deviation: "
        f"{deviation_report['average_run_semantic_deviation']}"
    )

    assert deviation_report["snapshot_count"] > 0
    assert deviation_report["run_vector_dimensions"] > 0
