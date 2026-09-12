"""BFCL-style AST scoring, implemented from scratch (not BFCL's own checker code).

The real BFCL ``possible_answer`` ground truth for a call looks like::

    {"function_name": {"param": ["accepted value 1", "accepted value 2", ...]}}

Where an empty string ``""`` in the accepted-values list is BFCL's sentinel for
"this parameter is optional and may be omitted entirely". A prediction is scored
correct for a single call when:

  1. the function name matches exactly,
  2. every ground-truth parameter is present with a value equal to one of its
     accepted values (or is legitimately omitted, if optional), and
  3. the prediction has no extra/hallucinated parameters beyond what's expected.

For categories with several expected calls (``parallel``, ``parallel_multiple``)
the predicted calls are matched against the ground-truth calls allowing any
permutation (BFCL does not require the model to emit calls in a particular
order), and the call counts must match exactly.
"""

from __future__ import annotations

from itertools import permutations
from typing import Any


def _to_num(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return None
    return None


def _as_bool(v: Any) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.strip().lower() in ("true", "false"):
        return v.strip().lower() == "true"
    return None


def values_equal(pred: Any, accepted: Any) -> bool:
    """Lenient equality between a predicted argument value and one accepted value."""
    ab = _as_bool(accepted)
    if ab is not None:
        pb = _as_bool(pred)
        return pb is not None and pb == ab

    if isinstance(accepted, (list, tuple)):
        if not isinstance(pred, (list, tuple)) or len(pred) != len(accepted):
            return False
        return all(values_equal(p, a) for p, a in zip(pred, accepted, strict=True))

    pn, an = _to_num(pred), _to_num(accepted)

    if isinstance(accepted, (int, float)) and not isinstance(accepted, bool):
        return pn is not None and abs(pn - float(accepted)) < 1e-6

    if isinstance(pred, (int, float)) and not isinstance(pred, bool) and an is not None:
        return abs(pn - an) < 1e-6

    if isinstance(pred, str) and isinstance(accepted, str):
        if pn is not None and an is not None:
            return abs(pn - an) < 1e-6
        return pred.strip().lower() == accepted.strip().lower()

    return pred == accepted


def _param_ok(accepted_values: list, predicted_args: dict, key: str) -> bool:
    optional = "" in accepted_values
    non_empty = [v for v in accepted_values if v != ""]
    if key in predicted_args:
        pv = predicted_args[key]
        return any(values_equal(pv, av) for av in non_empty)
    return optional


def match_call(pred_name: str, pred_args: dict | None, gt_entry: dict) -> tuple[bool, str]:
    """Check one predicted call against one ground-truth {fn_name: {param: [...]}}} entry."""
    if len(gt_entry) != 1:
        return False, "malformed ground truth entry"
    (gt_name, gt_params), = gt_entry.items()
    if pred_name != gt_name:
        return False, f"function name mismatch: predicted '{pred_name}' expected '{gt_name}'"
    pred_args = pred_args or {}
    for key, accepted in gt_params.items():
        if not _param_ok(accepted, pred_args, key):
            got = pred_args.get(key, "<missing>")
            return False, f"arg '{key}': got {got!r}, expected one of {accepted!r}"
    extra = set(pred_args) - set(gt_params)
    if extra:
        return False, f"hallucinated extra args: {sorted(extra)}"
    return True, "match"


def score_case(
    predicted_calls: list[tuple[str, dict]], ground_truth: list[dict]
) -> dict[str, Any]:
    """Score a full test case: predicted_calls is [(name, args), ...] in emission order."""
    if len(predicted_calls) != len(ground_truth):
        return {
            "correct": False,
            "reason": (
                f"call count mismatch: predicted {len(predicted_calls)} "
                f"vs expected {len(ground_truth)}"
            ),
        }
    n = len(ground_truth)
    if n == 0:
        return {"correct": True, "reason": "no calls expected"}

    perms = permutations(range(n)) if n <= 6 else [tuple(range(n))]
    best_reason = "no valid assignment of predicted calls to ground truth"
    for perm in perms:
        ok = True
        for i in range(n):
            name, args = predicted_calls[i]
            matched, reason = match_call(name, args, ground_truth[perm[i]])
            if not matched:
                ok = False
                best_reason = reason
                break
        if ok:
            return {"correct": True, "reason": "match"}
    return {"correct": False, "reason": best_reason}
