import json
import os

import numpy as np


def load_predictions_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def extract_confidence_labels(rows):
    confidences = []
    correctness = []
    for row in rows:
        row_confidences = row.get("confidences", [])
        row_correctness = row.get("is_correct", [])
        if len(row_confidences) != len(row_correctness):
            raise ValueError(
                f"Mismatched confidences/is_correct lengths for example id={row.get('id')}: "
                f"{len(row_confidences)} vs {len(row_correctness)}"
            )
        confidences.extend(row_confidences)
        correctness.extend(row_correctness)
    if not confidences:
        raise ValueError("No confidences found in predictions file.")
    return np.asarray(confidences, dtype=float), np.asarray(correctness, dtype=float)


def attach_calibrated_confidences(rows, calibrated_confidences, field_name="calibrated_confidences"):
    calibrated_confidences = list(map(float, calibrated_confidences))
    offset = 0
    new_rows = []
    for row in rows:
        row_confidences = row.get("confidences", [])
        count = len(row_confidences)
        copied = dict(row)
        copied[field_name] = calibrated_confidences[offset : offset + count]
        offset += count
        new_rows.append(copied)
    if offset != len(calibrated_confidences):
        raise ValueError("Unused calibrated confidences remained after rebuilding row structure.")
    return new_rows


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
