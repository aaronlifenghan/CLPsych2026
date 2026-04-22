"""
CLPsych 2026 — System 5 — Submission formatter for Task 3.1

Reads raw_task3.json (pipeline output) and writes:
  submission/task3_pred.json   — Codabench submission file
  submission/task3_pred.zip    — zipped for upload

Usage (standalone):
    python -m clpsych_assessment.system5.format_submission \\
        outputs/system5/llama3.1/zero_shot_direct/raw_task3.json \\
        --output-dir outputs/system5/llama3.1/zero_shot_direct/submission/

Or call write_submission() from Python.
"""

import argparse
import json
import logging
import os
import zipfile
from pathlib import Path
from typing import List, Dict

logger = logging.getLogger(__name__)


def load_raw(raw_path: str) -> List[Dict]:
    with open(raw_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {raw_path}, got {type(data)}")
    return data


def to_submission_format(raw: List[Dict]) -> List[Dict]:
    """
    Convert pipeline output to the exact Codabench submission schema:
        [{timeline_id, sequence_id, summary}, ...]

    Strips any extra fields (e.g. change_type, postids) that must not be
    submitted, and ensures no post text leaks through.
    """
    entries = []
    for item in raw:
        entry = {
            "timeline_id": item["timeline_id"],
            "sequence_id": item["sequence_id"],
            "summary": item.get("summary", ""),
        }
        entries.append(entry)
    return entries


def write_submission(
    raw_path: str,
    output_dir: str,
) -> str:
    """
    Write task3_pred.json and task3_pred.zip to output_dir.

    Returns path to the zip file.
    """
    os.makedirs(output_dir, exist_ok=True)

    raw = load_raw(raw_path)
    submission = to_submission_format(raw)

    # Write JSON
    json_path = os.path.join(output_dir, "task3_pred.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(submission, f, indent=2, ensure_ascii=False)
    logger.info(f"  task3_pred.json: {len(submission)} entries → {json_path}")

    # Zip (Codabench expects a zip containing task3_pred.json)
    zip_path = os.path.join(output_dir, "task3_pred.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, arcname="task3_pred.json")
    logger.info(f"  task3_pred.zip → {zip_path}")

    return zip_path


# ── CLI ──────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Format Task 3.1 pipeline output for Codabench submission"
    )
    parser.add_argument(
        "raw_output",
        help="Path to raw_task3.json from the pipeline",
    )
    parser.add_argument(
        "--output-dir", default="submission",
        help="Directory to write task3_pred.json and task3_pred.zip (default: submission/)",
    )
    args = parser.parse_args()

    zip_path = write_submission(args.raw_output, args.output_dir)
    print(f"\nSubmission ready: {zip_path}")


if __name__ == "__main__":
    main()
