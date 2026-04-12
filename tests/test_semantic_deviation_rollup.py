import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict


REPORTS_ROOT = Path("reports/test_semantic_deviation")
ROLLUP_REPORTS_ROOT = Path("reports/test_semantic_deviation_rollup")


class DeviationHistoryEntry(TypedDict):
    timestamp: str
    report_path: str
    snapshot_count: int
    average_run_semantic_deviation: float
    change_from_previous: float | None


class RollupReport(TypedDict):
    report_count: int
    history: list[DeviationHistoryEntry]
    latest_average_run_semantic_deviation: float
    best_average_run_semantic_deviation: float
    worst_average_run_semantic_deviation: float
    net_change_over_time: float
    improving: bool
    most_unstable_records: list[dict[str, float | int | str]]


def _load_report_history(reports_root: Path) -> list[DeviationHistoryEntry]:
    if not reports_root.exists():
        raise FileNotFoundError(f"Report directory not found: {reports_root}")

    report_dirs = sorted(path for path in reports_root.iterdir() if path.is_dir())
    history: list[DeviationHistoryEntry] = []
    previous_value: float | None = None

    for report_dir in report_dirs:
        report_path = report_dir / "semantic_deviation_report.json"
        if not report_path.exists():
            continue

        payload = json.loads(report_path.read_text(encoding="utf-8"))
        average = float(payload["average_run_semantic_deviation"])
        entry: DeviationHistoryEntry = {
            "timestamp": report_dir.name,
            "report_path": str(report_path),
            "snapshot_count": int(payload["snapshot_count"]),
            "average_run_semantic_deviation": average,
            "change_from_previous": (
                None if previous_value is None else average - previous_value
            ),
        }
        history.append(entry)
        previous_value = average

    if not history:
        raise ValueError(f"No semantic deviation reports found under {reports_root}")
    return history


def test_semantic_deviation_rollup() -> None:
    history = _load_report_history(REPORTS_ROOT)

    averages = [entry["average_run_semantic_deviation"] for entry in history]
    latest = averages[-1]
    best = min(averages)
    worst = max(averages)
    net_change = latest - averages[0]
    per_record_history: dict[str, list[float]] = {}

    for entry in history:
        payload = json.loads(Path(entry["report_path"]).read_text(encoding="utf-8"))
        for record in payload.get("per_record_deviation") or []:
            record_id = str(record.get("record_id") or "")
            average_deviation = record.get("average_deviation")
            if not record_id or not isinstance(average_deviation, (int, float)):
                continue
            per_record_history.setdefault(record_id, []).append(
                float(average_deviation)
            )

    most_unstable_records = sorted(
        [
            {
                "record_id": record_id,
                "report_count": len(values),
                "mean_average_deviation": float(sum(values) / len(values)),
                "max_average_deviation": float(max(values)),
                "min_average_deviation": float(min(values)),
            }
            for record_id, values in per_record_history.items()
            if values
        ],
        key=lambda item: float(item["mean_average_deviation"]),
        reverse=True,
    )

    rollup_report: RollupReport = {
        "report_count": len(history),
        "history": history,
        "latest_average_run_semantic_deviation": latest,
        "best_average_run_semantic_deviation": best,
        "worst_average_run_semantic_deviation": worst,
        "net_change_over_time": net_change,
        "improving": net_change < 0,
        "most_unstable_records": most_unstable_records[:10],
    }

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    report_dir = ROLLUP_REPORTS_ROOT / timestamp
    report_dir.mkdir(parents=True, exist_ok=True)
    rollup_output_path = report_dir / "semantic_deviation_rollup.json"
    rollup_output_path.write_text(
        json.dumps(rollup_report, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote rollup report to: {rollup_output_path}")
    print(f"History points: {rollup_report['report_count']}")
    print(
        "Latest Average Run Semantic Deviation: "
        f"{rollup_report['latest_average_run_semantic_deviation']}"
    )
    print(f"Net change over time: {rollup_report['net_change_over_time']}")
    print(f"Improving: {rollup_report['improving']}")
    print(f"Most unstable records: {rollup_report['most_unstable_records']}")

    assert rollup_report["report_count"] > 0
