"""
Constants and subelement schema for CLPsych 2026 System 4.
 
Mirrors the shared-task specification (§8) and the evaluation logic
in system3/evaluate.py so that predictions are always schema-valid.
"""
 
ELEMENTS = ["A", "B-O", "B-S", "C-O", "C-S", "D"]
VALENCES = ["adaptive-state", "maladaptive-state"]
VALENCE_SHORT = {"adaptive-state": "adaptive", "maladaptive-state": "maladaptive"}
 
# Valid subelement IDs per (valence_short, element).
# Class 0 = absent is always implicitly valid but excluded from scoring.
VALID_SUBELEMENTS: dict[tuple[str, str], list[int]] = {
    ("adaptive", "A"):   [1, 3, 5, 7, 9, 11, 13],
    ("adaptive", "B-O"): [1, 3],
    ("adaptive", "B-S"): [1],
    ("adaptive", "C-O"): [1, 3],
    ("adaptive", "C-S"): [1],
    ("adaptive", "D"):   [1, 3, 5],
    ("maladaptive", "A"):   [2, 4, 6, 8, 10, 12, 14],
    ("maladaptive", "B-O"): [2, 4],
    ("maladaptive", "B-S"): [2],
    ("maladaptive", "C-O"): [2, 4],
    ("maladaptive", "C-S"): [2],
    ("maladaptive", "D"):   [2, 4, 6],
}
 
# Full class set per slot: [0] + valid subelements
def classes_for_slot(valence_short: str, element: str) -> list[int]:
    """Return [0, sub1, sub2, ...] for this (valence, element) pair."""
    return [0] + VALID_SUBELEMENTS[(valence_short, element)]