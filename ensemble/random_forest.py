"""Random-forest ensemble combining RatioWaveNet, TCFormer and raw EEG windows."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.ensemble import RandomForestClassifier

try:  # torch is optional at inference time but available during training
    import torch
except Exception:  # pragma: no cover - guard in case torch is missing at runtime
    torch = None

ArrayLike = Union[np.ndarray, "torch.Tensor", Sequence[float]]


def _to_numpy(array: ArrayLike) -> np.ndarray:
    """Convert tensors/lists to a float32 numpy array."""
    if torch is not None and isinstance(array, torch.Tensor):
        array = array.detach().cpu().numpy()
    array = np.asarray(array, dtype=np.float32)
    return array


def _ensure_batch_first(array: np.ndarray) -> np.ndarray:
    """Ensure the array has shape (batch, length, features)."""
    if array.ndim == 1:
        return array[None, :, None]
    if array.ndim == 2:
        return array[:, None, :]
    if array.ndim == 3:
        return array
    raise ValueError(
        f"Expected array with 1, 2 or 3 dimensions, received shape {array.shape}."
    )


def _resample_batch(batch: np.ndarray, target_length: int) -> np.ndarray:
    """Resample the second dimension of ``batch`` to ``target_length`` using linear interpolation."""
    if target_length <= 0:
        raise ValueError("target_length must be > 0")
    if batch.shape[1] == target_length:
        return batch
    if batch.shape[1] == 1:
        return np.repeat(batch, target_length, axis=1)

    n_samples, _, n_features = batch.shape
    old_positions = np.linspace(0.0, 1.0, batch.shape[1], dtype=np.float32)
    new_positions = np.linspace(0.0, 1.0, target_length, dtype=np.float32)
    resampled = np.empty((n_samples, target_length, n_features), dtype=batch.dtype)

    for sample_idx in range(n_samples):
        for feature_idx in range(n_features):
            resampled[sample_idx, :, feature_idx] = np.interp(
                new_positions,
                old_positions,
                batch[sample_idx, :, feature_idx],
            )
    return resampled


def prepare_ensemble_features(
    ratiowavenet_preds: ArrayLike,
    tcformer_preds: ArrayLike,
    signals: ArrayLike,
    *,
    target_length: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """Stack predictions and raw signals into a single feature matrix.

    Parameters
    ----------
    ratiowavenet_preds, tcformer_preds
        Per-sample predictions produced by the individual deep models. They can be
        2-D arrays of shape ``(n_samples, n_features)`` or 3-D arrays of shape
        ``(n_samples, length, n_features)``. When 2-D arrays are provided they are
        promoted to 3-D tensors with ``length=1``.
    signals
        Raw EEG windows with shape ``(n_samples, n_channels, n_timepoints)``.
    target_length
        Optional temporal resolution used to harmonise the three sources. When ``None``
        the number of time-points in ``signals`` is used and both prediction tensors
        are up-sampled accordingly. Provide a smaller value (e.g. 50 or 100) to
        down-sample the signal and keep the resulting feature space compact.

    Returns
    -------
    features : ndarray, shape (n_samples, n_total_features)
        Flattened representation concatenating the resampled sources.
    effective_length : int
        Time dimension adopted during the harmonisation step.
    """
    rw = _ensure_batch_first(_to_numpy(ratiowavenet_preds))
    tc = _ensure_batch_first(_to_numpy(tcformer_preds))
    sig = _to_numpy(signals)

    if sig.ndim != 3:
        raise ValueError(
            "signals must have shape (n_samples, n_channels, n_timepoints), "
            f"received {sig.shape}."
        )

    n_samples = sig.shape[0]
    sig = np.transpose(sig, (0, 2, 1))  # (batch, time, channels)

    if rw.shape[0] != n_samples or tc.shape[0] != n_samples:
        raise ValueError(
            "All inputs must contain the same number of samples: "
            f"signals={n_samples}, RatioWaveNet={rw.shape[0]}, TCFormer={tc.shape[0]}."
        )

    effective_length = (
        int(target_length)
        if target_length is not None
        else int(sig.shape[1])
    )
    if effective_length <= 0:
        raise ValueError("effective_length must be positive")

    rw_resampled = _resample_batch(rw, effective_length)
    tc_resampled = _resample_batch(tc, effective_length)
    sig_resampled = _resample_batch(sig, effective_length)

    features = np.concatenate(
        [
            rw_resampled.reshape(n_samples, -1),
            tc_resampled.reshape(n_samples, -1),
            sig_resampled.reshape(n_samples, -1),
        ],
        axis=1,
    )
    return features, effective_length


@dataclass
class RandomForestSignalEnsembler:
    """Random-forest ensemble built on top of RatioWaveNet, TCFormer and raw EEG."""

    n_estimators: int = 400
    max_depth: Optional[int] = None
    min_samples_leaf: int = 1
    target_length: Optional[int] = None
    random_state: Optional[int] = 42
    n_jobs: Optional[int] = -1
    class_weight: Optional[Union[str, dict]] = "balanced"

    _model: RandomForestClassifier = None
    _effective_length: Optional[int] = None

    def _get_model(self) -> RandomForestClassifier:
        if self._model is None:
            self._model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                class_weight=self.class_weight,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
        return self._model

    def fit(
        self,
        ratiowavenet_preds: ArrayLike,
        tcformer_preds: ArrayLike,
        signals: ArrayLike,
        labels: ArrayLike,
    ) -> "RandomForestSignalEnsembler":
        """Fit the ensemble using aligned features."""
        labels_np = _to_numpy(labels).astype(np.int64)
        if labels_np.ndim != 1:
            labels_np = labels_np.reshape(-1)

        features, eff_len = prepare_ensemble_features(
            ratiowavenet_preds,
            tcformer_preds,
            signals,
            target_length=self.target_length,
        )
        if features.shape[0] != labels_np.shape[0]:
            raise ValueError(
                "labels must contain one entry per sample: "
                f"features={features.shape[0]}, labels={labels_np.shape[0]}"
            )

        self._effective_length = eff_len
        self._get_model().fit(features, labels_np)
        return self

    def predict(
        self,
        ratiowavenet_preds: ArrayLike,
        tcformer_preds: ArrayLike,
        signals: ArrayLike,
    ) -> np.ndarray:
        """Predict classes for new samples."""
        features, _ = prepare_ensemble_features(
            ratiowavenet_preds,
            tcformer_preds,
            signals,
            target_length=self._effective_length
            if self._effective_length is not None
            else self.target_length,
        )
        return self._get_model().predict(features)

    def predict_proba(
        self,
        ratiowavenet_preds: ArrayLike,
        tcformer_preds: ArrayLike,
        signals: ArrayLike,
    ) -> np.ndarray:
        """Predict class probabilities for new samples."""
        features, _ = prepare_ensemble_features(
            ratiowavenet_preds,
            tcformer_preds,
            signals,
            target_length=self._effective_length
            if self._effective_length is not None
            else self.target_length,
        )
        return self._get_model().predict_proba(features)

    @property
    def effective_length(self) -> Optional[int]:
        """Return the temporal resolution adopted during fitting."""
        return self._effective_length