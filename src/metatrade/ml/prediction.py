"""Prediction builder — assembles a rich MlPrediction from classifier output.

This module contains a single pure function ``build_ml_prediction`` that
bridges the raw classifier output (probabilities + class array) with the
``MlPrediction`` DTO.  It handles:

- XGBoost label remapping (raw class indices → domain labels)
- Optional calibrated probability injection
- ``confidence_margin`` computation (gap between top-1 and top-2 probas)
- Per-domain-class probability extraction (p_buy / p_hold / p_sell)
"""

from __future__ import annotations

import numpy as np

from metatrade.ml.contracts import MlPrediction


def build_ml_prediction(
    raw_direction: int,
    probas: np.ndarray,
    classes: list[int],
    inv_label_remap: dict[int, int] | None = None,
    calibrated_probas: np.ndarray | None = None,
) -> MlPrediction:
    """Assemble an ``MlPrediction`` from classifier output.

    Args:
        raw_direction:     Raw class value predicted by the model (argmax of
                           raw ``predict_proba``), before any label remapping.
        probas:            Raw class probabilities, shape ``(n_classes,)``,
                           ordered by ``classes``.
        classes:           Ordered list of class values matching ``probas``
                           columns (e.g. ``[0, 1, 2]`` for XGBoost or
                           ``[-1, 0, 1]`` for other backends).
        inv_label_remap:   Optional mapping from raw class value to domain
                           label (used by XGBoost to convert 0/1/2 → -1/0/1).
                           Pass ``None`` or ``{}`` for other backends.
        calibrated_probas: Calibrated probabilities in the same shape and
                           column order as ``probas``.  When provided,
                           ``direction`` and ``confidence`` are derived from
                           the calibrated values.

    Returns:
        ``MlPrediction`` with direction, confidence, raw_direction,
        per-class probabilities (p_buy / p_hold / p_sell) and
        confidence_margin.
    """
    remap = inv_label_remap or {}

    # Domain direction from raw argmax (overridden below when calibrated)
    direction = remap.get(raw_direction, raw_direction)

    # Select effective probas: calibrated if available, otherwise raw
    effective: np.ndarray = (
        calibrated_probas if calibrated_probas is not None else probas
    )

    # When calibrated, re-derive direction from calibrated argmax
    if calibrated_probas is not None:
        cal_max_idx = int(calibrated_probas.argmax())
        cal_raw_cls = classes[cal_max_idx]
        direction = remap.get(cal_raw_cls, cal_raw_cls)

    # Map each column to its domain label and extract p_buy / p_hold / p_sell
    p_by_domain: dict[int, float] = {
        remap.get(raw_cls, raw_cls): float(effective[i])
        for i, raw_cls in enumerate(classes)
    }
    p_buy = p_by_domain.get(1)
    p_hold = p_by_domain.get(0)
    p_sell = p_by_domain.get(-1)

    # Confidence: max probability from effective probas
    confidence = float(effective.max())

    # Confidence margin: gap between top-1 and top-2 probabilities
    n_classes = len(effective)
    if n_classes >= 2:
        top2 = np.partition(effective, -2)[-2:]
        confidence_margin: float | None = float(top2[-1] - top2[-2])
    else:
        confidence_margin = None

    return MlPrediction(
        direction=direction,
        confidence=confidence,
        raw_direction=raw_direction,
        p_buy=p_buy,
        p_hold=p_hold,
        p_sell=p_sell,
        confidence_margin=confidence_margin,
    )
