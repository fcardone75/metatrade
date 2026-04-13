"""Intermarket feature builder — converts DependencySnapshots to ML features.

Produces a flat dict of float features that can be appended to a FeatureVector
or passed directly to any sklearn-compatible model.

Naming convention (all lowercase):
    im_{src}_{tgt}_{metric}

Where:
    src / tgt  = symbol names stripped of "/" and lowercased
    metric     = one of: pcorr, scorr, lag, lagcorr, cov, stab, conf, impact

Example features for EURUSD / USDJPY pair:
    im_eurusd_usdjpy_pcorr     → Pearson correlation value (signed)
    im_eurusd_usdjpy_scorr     → Spearman correlation value (signed)
    im_eurusd_usdjpy_lag       → dominant lag in bars (signed)
    im_eurusd_usdjpy_lagcorr   → cross-correlation at dominant lag (signed)
    im_eurusd_usdjpy_cov       → rolling covariance (raw, not clipped)
    im_eurusd_usdjpy_stab      → stability score [0, 1]
    im_eurusd_usdjpy_conf      → confidence [0, 1]
    im_eurusd_usdjpy_impact    → current signal impact [-1, 1]
"""

from __future__ import annotations

from metatrade.intermarket.contracts import DependencySnapshot


def _sym(name: str) -> str:
    """Normalise symbol name to a compact lowercase key (e.g. 'EUR/USD' → 'eurusd')."""
    return name.lower().replace("/", "").replace("-", "").replace("_", "")


class IntermarketFeatureBuilder:
    """Converts a list of DependencySnapshots into a flat ML feature dict."""

    def build(self, dependencies: list[DependencySnapshot]) -> dict[str, float]:
        """Return a flat feature dict from all provided dependency snapshots.

        Dependencies of the same pair but different relationship types
        contribute different keys, so they do not overwrite each other.
        """
        features: dict[str, float] = {}

        for dep in dependencies:
            src = _sym(dep.source_instrument)
            tgt = _sym(dep.target_instrument)
            prefix = f"im_{src}_{tgt}"

            rt = dep.relationship_type
            signed_strength = dep.strength_score * dep.directionality

            if rt == "pearson":
                features[f"{prefix}_pcorr"] = signed_strength
            elif rt == "spearman":
                features[f"{prefix}_scorr"] = signed_strength
            elif rt == "cross_corr":
                features[f"{prefix}_lag"] = float(dep.lag_bars)
                features[f"{prefix}_lagcorr"] = dep.strength_score * dep.directionality
            elif rt == "covariance":
                features[f"{prefix}_cov"] = dep.current_signal_impact  # covariance is unbounded; store impact

            # Always include meta-features for every relationship type
            features[f"{prefix}_stab"] = dep.stability_score
            features[f"{prefix}_conf"] = dep.confidence
            features[f"{prefix}_impact"] = dep.current_signal_impact

        return features
