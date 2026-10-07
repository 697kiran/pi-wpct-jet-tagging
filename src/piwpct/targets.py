"""Explicit target definitions for JetClass process or future flavour labels."""

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml


PROCESS_CLASSES: Tuple[Tuple[str, str], ...] = (
    ("QCD", "label_QCD"),
    ("Hbb", "label_Hbb"),
    ("Hcc", "label_Hcc"),
    ("Hgg", "label_Hgg"),
    ("H4q", "label_H4q"),
    ("Hqql", "label_Hqql"),
    ("Zqq", "label_Zqq"),
    ("Wqq", "label_Wqq"),
    ("Tbqq", "label_Tbqq"),
    ("Tbl", "label_Tbl"),
)
PROCESS_BRANCH_TO_NAME = {branch: name for name, branch in PROCESS_CLASSES}


class TargetDefinitionError(ValueError):
    """Raised when configured target semantics cannot be supported by ROOT."""


def default_target_definition(mode: str) -> Dict[str, Any]:
    if mode == "process_classification":
        classes = [{"name": name, "branch": branch} for name, branch in PROCESS_CLASSES]
        return {
            "mode": mode,
            "classes": classes,
            "semantics": "JetClass generated-process classification; not jet-flavour tagging.",
            "main_experiment": False,
        }
    if mode == "flavour_tagging":
        return {
            "mode": mode,
            "classes": [
                {"name": "light", "branch": "label_light"},
                {"name": "charm", "branch": "label_charm"},
                {"name": "b", "branch": "label_b"},
            ],
            "semantics": "Per-jet light/charm/b flavour tagging; requires a verified per-jet annotation.",
            "main_experiment": True,
        }
    raise TargetDefinitionError(
        f"Unknown target mode {mode!r}; choose process_classification or flavour_tagging."
    )


def load_target_definition(
    mode: str, config_path: Optional[str] = None
) -> Dict[str, Any]:
    """Load/validate a user-supplied class-to-branch mapping."""
    if config_path:
        path = Path(config_path)
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise TargetDefinitionError(f"Could not read target config {path}: {exc}") from exc
        if not isinstance(loaded, dict):
            raise TargetDefinitionError(f"Target config {path} must be a YAML mapping.")
        definition = dict(loaded.get("target", loaded))
        configured_mode = definition.get("mode")
        if configured_mode != mode:
            raise TargetDefinitionError(
                f"Target config mode {configured_mode!r} does not match requested {mode!r}."
            )
    else:
        definition = default_target_definition(mode)

    classes = definition.get("classes")
    if not isinstance(classes, list) or not classes:
        raise TargetDefinitionError("Target definition must contain a non-empty ordered 'classes' list.")
    names: List[str] = []
    branches: List[str] = []
    for item in classes:
        if not isinstance(item, dict) or not item.get("name") or not item.get("branch"):
            raise TargetDefinitionError("Each target class needs non-empty 'name' and 'branch' fields.")
        names.append(str(item["name"]))
        branches.append(str(item["branch"]))
    if len(names) < 2:
        raise TargetDefinitionError("At least two target classes are required for classification.")
    if len(set(names)) != len(names) or len(set(branches)) != len(branches):
        raise TargetDefinitionError("Target class names and ROOT branch names must be unique.")
    definition["mode"] = mode
    definition["classes"] = [
        {"id": index, "name": name, "branch": branch}
        for index, (name, branch) in enumerate(zip(names, branches))
    ]
    definition["class_names"] = names
    definition["label_branches"] = branches
    definition.setdefault("main_experiment", mode == "flavour_tagging")
    definition.setdefault("semantics", mode)
    if mode == "process_classification":
        definition["main_experiment"] = False
        if any(name in {"light", "charm", "b"} for name in names):
            raise TargetDefinitionError(
                "process_classification class names must retain their process semantics; "
                "light/charm/b are flavour labels."
            )
        mismatched_process_labels = [
            (name, branch, PROCESS_BRANCH_TO_NAME[branch])
            for name, branch in zip(names, branches)
            if branch in PROCESS_BRANCH_TO_NAME and name != PROCESS_BRANCH_TO_NAME[branch]
        ]
        if mismatched_process_labels:
            name, branch, process_name = mismatched_process_labels[0]
            raise TargetDefinitionError(
                f"Process label branch {branch!r} is {process_name}, not {name!r}; "
                "process labels cannot be renamed as flavour classes."
            )
    elif mode == "flavour_tagging":
        if names != ["light", "charm", "b"]:
            raise TargetDefinitionError(
                "flavour_tagging classes must be ordered exactly as light, charm, b."
            )
        process_branches = [branch for branch in branches if branch in PROCESS_BRANCH_TO_NAME]
        if process_branches:
            raise TargetDefinitionError(
                f"JetClass process label branches {process_branches} cannot define light/charm/b "
                "flavour targets. Provide verified per-jet flavour annotations instead."
            )
    return definition


def validate_label_branches(
    target_definition: Mapping[str, Any], available_branches: Iterable[str]
) -> None:
    available = set(available_branches)
    required = list(target_definition.get("label_branches", []))
    if not required:
        required = [str(item["branch"]) for item in target_definition.get("classes", [])]
    if not required:
        raise TargetDefinitionError("Target definition has no label branches to validate.")
    missing = [branch for branch in required if branch not in available]
    if not missing:
        return
    mode = target_definition["mode"]
    if mode == "flavour_tagging":
        present_process = sorted(branch for _, branch in PROCESS_CLASSES if branch in available)
        raise TargetDefinitionError(
            "Requested flavour_tagging requires verified per-jet flavour label branches "
            f"{required}, but the ROOT tree is missing {missing}. Available JetClass process "
            f"labels include {present_process or 'none'}. Process labels such as QCD, Hbb, and "
            "Hcc do not identify light/charm/b jet flavour; provide a valid flavour annotation "
            "and map it in a target config."
        )
    raise TargetDefinitionError(
        f"Requested process_classification label branches are unavailable: {missing}."
    )


def normalize_one_hot_labels(
    values: Sequence[np.ndarray],
    target_definition: Mapping[str, Any],
    *,
    tolerance: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize bool/int/float one-hot branches to class IDs and validity mask.

    Rows are valid only when all values are finite binary indicators and exactly
    one class is active. Invalid rows are returned in the mask for the caller's
    explicit malformed-sample policy; they are never silently relabelled.
    """
    arrays = [np.asarray(value).reshape(-1) for value in values]
    if not arrays:
        raise TargetDefinitionError("No label arrays were provided.")
    lengths = {len(value) for value in arrays}
    if len(lengths) != 1:
        raise TargetDefinitionError("Label branches have inconsistent entry counts.")
    matrix = np.column_stack(arrays).astype(np.float64, copy=False)
    finite = np.isfinite(matrix).all(axis=1)
    binary = np.isclose(matrix, 0.0, atol=tolerance) | np.isclose(matrix, 1.0, atol=tolerance)
    exactly_one = np.isclose(matrix.sum(axis=1), 1.0, atol=tolerance)
    valid = finite & binary.all(axis=1) & exactly_one
    labels = np.full(len(matrix), -1, dtype=np.int64)
    if np.any(valid):
        labels[valid] = np.argmax(matrix[valid], axis=1).astype(np.int64)
    return labels, valid


def target_manifest_metadata(definition: Mapping[str, Any]) -> Dict[str, Any]:
    """Return portable, explicit class semantics for a prepared manifest."""
    return {
        "mode": definition["mode"],
        "semantics": definition["semantics"],
        "main_experiment": bool(definition.get("main_experiment", False)),
        "classes": list(definition["classes"]),
        "label_branches": list(definition["label_branches"]),
        "class_names": list(definition["class_names"]),
        "warning": (
            "Process classes are not per-jet flavour labels."
            if definition["mode"] == "process_classification" else None
        ),
    }
