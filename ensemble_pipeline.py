#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Pipeline soggetto-wise per combinare RatioWaveNet, TCFormer e segnale grezzo.

Questo script:

* Carica i checkpoint RatioWaveNet/TCFormer per ciascun soggetto.
* Estrae logits (train/test) ed eventualmente il segnale grezzo.
* Allinea temporalmente le tre sorgenti e addestra un meta-modello
  (Random Forest, piccola CNN oppure TCN compatta).
* Stampa Accuracy/Kappa per soggetto, la media finale e salva un CSV con le
  metriche.

Esempio utilizzo::

    python ensemble_pipeline.py \
        --dataset bcic2b --loso --subject-ids all --gpu-id 0 \
        --rwn-pattern "results/ensemble_checkpoints/bci2b/ratiowavenet_subject_{sid}.ckpt" \
        --tcf-pattern "results/ensemble_checkpoints/bci2b/tcformer_subject_{sid}.ckpt" \
        --use-signal --target-length 64 --pca-sig 128 --meta rf

    python ensemble_pipeline.py \
        --dataset bcic2b --subject-ids all --gpu-id 0 \
        --rwn-pattern "results/ensemble_checkpoints/bci2b/ratiowavenet_subject_{sid}.ckpt" \
        --tcf-pattern "results/ensemble_checkpoints/bci2b/tcformer_subject_{sid}.ckpt" \
        --use-signal --target-length 64 --meta cnn --cnn-epochs 20 --cnn-emb 8

    python ensemble_pipeline.py \
        --dataset bcic2b --subject-ids all --gpu-id 0 \
        --rwn-pattern "results/ensemble_checkpoints/bci2b/ratiowavenet_subject_{sid}.ckpt" \
        --tcf-pattern "results/ensemble_checkpoints/bci2b/tcformer_subject_{sid}.ckpt" \
        --use-signal --target-length 128 --meta tcn --tcn-levels 5 --tcn-hidden 96
"""

from __future__ import annotations

import copy
import csv
import os
import shutil
from argparse import ArgumentParser
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
import yaml

from utils.get_datamodule_cls import get_datamodule_cls
from utils.get_model_cls import get_model_cls
from utils.seed import seed_everything

from train_pipeline import CONFIG_DIR as TP_CONFIG_DIR, train_and_test


# ----------------------------------------------------------------------------
# Utility helper per i checkpoint
# ----------------------------------------------------------------------------

def _dataset_folder_name(dataset: str) -> str:
    """Converte il nome CLI nel nome cartella usato nei checkpoint."""

    mapping = {"bcic2a": "bci2a", "bcic2b": "bci2b"}
    base = dataset.replace("_loso", "")
    return mapping.get(base, base)


def _format_pattern(pattern: str, sid: int, dataset: str) -> Path:
    """Applica placeholder supportati al pattern di checkpoint."""

    dataset_folder = _dataset_folder_name(dataset)
    return Path(
        pattern.format(
            sid=sid,
            subject_id=sid,
            dataset=dataset,
            dataset_folder=dataset_folder,
        )
    )


def _find_ckpt_any(
    patterns: Iterable[str | None], sid: int, dataset: str, model_name: str
) -> Path:
    """Restituisce il primo checkpoint esistente tra i pattern forniti."""

    tried: list[str] = []
    for pattern in patterns:
        if not pattern:
            continue
        candidate = _format_pattern(pattern, sid, dataset)
        tried.append(str(candidate))
        if candidate.exists():
            return candidate

    msg = [f"Checkpoint {model_name} non trovato per subject {sid}. Tentativi:"]
    msg.extend(f"  - {path}" for path in tried)
    raise FileNotFoundError("\n".join(msg))


def _seed_folder_name(seed: int) -> str:
    return f"sid{seed}"


def _target_checkpoint_dir(dataset: str, seed: int) -> Path:
    dataset_folder = _dataset_folder_name(dataset)
    return Path("results/ensemble_checkpoints") / dataset_folder / _seed_folder_name(seed)


def _maybe_copy_from_legacy(
    prefix: str, dataset: str, subject_id: int, target_path: Path
) -> bool:
    """Se esiste un checkpoint in layout legacy lo copia nella nuova posizione."""

    dataset_folder = _dataset_folder_name(dataset)
    legacy_patterns = [
        f"results/ensemble_checkpoints/{dataset_folder}/{prefix}_subject_{{sid}}.ckpt",
        f"results/ensemble_checkpoints/{dataset}/{prefix}_subject_{{sid}}.ckpt",
        f"results/ensemble_checkpoints/{prefix}_subject_{{sid}}.ckpt",
    ]

    for pattern in legacy_patterns:
        candidate = _format_pattern(pattern, subject_id, dataset)
        if candidate == target_path:
            continue
        if candidate.exists():
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, target_path)
            return True
    return False


def _build_training_config(
    model_name: str,
    args,
    datamodule_cls,
    subject_ids: list[int],
    seed_value: int,
):
    cfg_path = TP_CONFIG_DIR / f"{model_name.lower()}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config per il modello '{model_name}' non trovata: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config = copy.deepcopy(config)
    model_key = model_name.lower()
    config["model"] = model_key

    dataset_key = args.dataset
    dataset_name = dataset_key + "_loso" if args.loso else dataset_key
    config["dataset_name"] = dataset_name

    if args.loso:
        if dataset_key == "hgd":
            config["max_epochs"] = config.get("max_epochs_loso_hgd", config["max_epochs"])
        else:
            config["max_epochs"] = config.get("max_epochs_loso", config["max_epochs"])
        if "model_kwargs" in config and "warmup_epochs_loso" in config["model_kwargs"]:
            config["model_kwargs"]["warmup_epochs"] = config["model_kwargs"]["warmup_epochs_loso"]
    else:
        if dataset_key == "bcic2b" and "max_epochs_2b" in config:
            config["max_epochs"] = config["max_epochs_2b"]

    preprocess_cfg = copy.deepcopy(config["preprocessing"][dataset_key])
    preprocess_cfg["z_scale"] = config.get("z_scale", True)
    if args.interaug:
        preprocess_cfg["interaug"] = True
    elif args.no_interaug:
        preprocess_cfg["interaug"] = False
    else:
        if "interaug" in config:
            preprocess_cfg["interaug"] = config["interaug"]
        else:
            preprocess_cfg.setdefault("interaug", True)
    config["preprocessing"] = preprocess_cfg
    config.pop("interaug", None)

    config["gpu_id"] = args.gpu_id
    config["seed"] = seed_value

    requested_subjects = list(subject_ids)
    all_subjects = list(getattr(datamodule_cls, "all_subject_ids", []))
    if all_subjects and sorted(requested_subjects) == sorted(all_subjects):
        config["subject_ids"] = "all"
    elif len(requested_subjects) == 1:
        config["subject_ids"] = requested_subjects[0]
    else:
        config["subject_ids"] = requested_subjects

    config["save_checkpoint"] = True
    return config


def _locate_training_run_dir(config) -> Path:
    results_root = Path(__file__).resolve().parent / "results"
    dataset_name = config["dataset_name"]
    model_name = config["model"]
    seed_value = config["seed"]
    interaug_flag = config["preprocessing"].get("interaug", False)
    gpu_id = config.get("gpu_id", 0)
    pattern = f"{model_name}_{dataset_name}_seed-{seed_value}_aug-{interaug_flag}_GPU{gpu_id}_*"
    candidates = sorted(results_root.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(
            f"Impossibile trovare la cartella dei risultati per il run: pattern '{pattern}'"
        )
    return candidates[-1]


def _copy_model_checkpoints(
    run_dir: Path, prefix: str, target_dir: Path, subject_ids: list[int]
) -> None:
    ckpt_dir = run_dir / "checkpoints"
    for sid in subject_ids:
        src = ckpt_dir / f"subject_{sid}_model.ckpt"
        if not src.exists():
            raise FileNotFoundError(
                f"Checkpoint mancante dopo l'addestramento automatico: {src}"
            )
        dst = target_dir / f"{prefix}_subject_{sid}.ckpt"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _train_and_collect_checkpoints(
    model_name: str,
    prefix: str,
    args,
    datamodule_cls,
    subject_ids: list[int],
    seed_value: int,
    target_dir: Path,
) -> None:
    train_config = _build_training_config(model_name, args, datamodule_cls, subject_ids, seed_value)

    print(
        f"[AutoTrain] Avvio training {model_name} per dataset={args.dataset}, seed={seed_value} "
        f"(subjects={subject_ids})."
    )
    train_and_test(train_config)

    run_dir = _locate_training_run_dir(train_config)
    print(f"[AutoTrain] Checkpoint salvati in {run_dir}")
    _copy_model_checkpoints(run_dir, prefix, target_dir, subject_ids)


def _ensure_model_checkpoints(
    model_name: str,
    prefix: str,
    args,
    datamodule_cls,
    subject_ids: list[int],
    seed_value: int,
) -> None:
    target_dir = _target_checkpoint_dir(args.dataset, seed_value)
    target_dir.mkdir(parents=True, exist_ok=True)

    missing: list[int] = []
    for sid in subject_ids:
        dst = target_dir / f"{prefix}_subject_{sid}.ckpt"
        if dst.exists():
            continue
        if _maybe_copy_from_legacy(prefix, args.dataset, sid, dst):
            continue
        missing.append(sid)

    if not missing:
        return

    _train_and_collect_checkpoints(model_name, prefix, args, datamodule_cls, missing, seed_value, target_dir)


def _ensure_required_checkpoints(
    args,
    datamodule_cls,
    subject_ids: list[int],
    seed_value: int,
) -> None:
    _ensure_model_checkpoints("ratiowavenet", "ratiowavenet", args, datamodule_cls, subject_ids, seed_value)
    _ensure_model_checkpoints("tcformer", "tcformer", args, datamodule_cls, subject_ids, seed_value)


# ----------------------------------------------------------------------------
# Utility numeriche
# ----------------------------------------------------------------------------

def _softmax_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=-1, keepdims=True) + 1e-12)


def _ensure_batch_first(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        return arr[None, :, None]
    if arr.ndim == 2:
        return arr[:, None, :]
    if arr.ndim == 3:
        return arr
    raise ValueError(f"Forma non supportata: ndim={arr.ndim}")


def _linresample_time(batch: np.ndarray, target_len: int | None) -> np.ndarray:
    if target_len is None or batch.shape[1] == target_len:
        return batch
    if batch.shape[1] == 1:
        return np.repeat(batch, target_len, axis=1)

    n, L, F = batch.shape
    old = np.linspace(0, 1, L, dtype=np.float32)
    new = np.linspace(0, 1, target_len, dtype=np.float32)
    out = np.empty((n, target_len, F), dtype=batch.dtype)
    for i in range(n):
        for j in range(F):
            out[i, :, j] = np.interp(new, old, batch[i, :, j])
    return out


# ----------------------------------------------------------------------------
# Raccolta sorgenti dai modelli base
# ----------------------------------------------------------------------------


@torch.no_grad()
def _collect_sources(
    rw_model: torch.nn.Module,
    tc_model: torch.nn.Module,
    dataset,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    return_signal: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(0, num_workers),
        pin_memory=(device.type == "cuda"),
    )

    rw_list: list[np.ndarray] = []
    tc_list: list[np.ndarray] = []
    sig_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []

    rw_model.eval()
    tc_model.eval()

    for x, y in loader:
        y_list.append(y.numpy())
        if return_signal:
            sig_list.append(x.numpy())  # (B, C, T)

        x = x.to(device, non_blocking=True)
        rw_logits = rw_model(x)
        tc_logits = tc_model(x)

        rw_np = rw_logits.detach().cpu().numpy()
        tc_np = tc_logits.detach().cpu().numpy()

        if rw_np.ndim == 2:
            rw_np = rw_np[:, None, :]
        if tc_np.ndim == 2:
            tc_np = tc_np[:, None, :]

        rw_list.append(_softmax_np(rw_np))
        tc_list.append(_softmax_np(tc_np))

    rw = np.concatenate(rw_list, axis=0) if rw_list else np.empty((0,))
    tc = np.concatenate(tc_list, axis=0) if tc_list else np.empty((0,))
    sig = (
        np.concatenate(sig_list, axis=0)
        if (return_signal and sig_list)
        else np.empty((0,))
    )
    y = np.concatenate(y_list, axis=0) if y_list else np.empty((0,))
    return rw, tc, sig, y


# ----------------------------------------------------------------------------
# Feature engineering per il Random Forest
# ----------------------------------------------------------------------------


def _make_rf_features(
    rwn: np.ndarray,
    tcf: np.ndarray,
    sig: np.ndarray | None,
    *,
    target_len: int | None,
    use_signal: bool,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    rwn = _ensure_batch_first(rwn)
    tcf = _ensure_batch_first(tcf)
    rwn = _linresample_time(rwn, target_len)
    tcf = _linresample_time(tcf, target_len)

    feats = [rwn.reshape(rwn.shape[0], -1), tcf.reshape(tcf.shape[0], -1)]
    d_rwn = feats[0].shape[1]
    d_tcf = feats[1].shape[1]
    d_sig = 0

    if use_signal and sig is not None and sig.size:
        sig_bt = np.transpose(sig, (0, 2, 1))
        sig_bt = _linresample_time(sig_bt, target_len)
        sig_f = sig_bt.reshape(sig_bt.shape[0], -1)
        feats.append(sig_f)
        d_sig = sig_f.shape[1]

    X = np.concatenate(feats, axis=1)
    return X, (d_rwn, d_tcf, d_sig)


class RFBlockScalerPCA:
    """Scaling + PCA per blocco per evitare dominanza del segnale."""

    def __init__(
        self,
        dims: tuple[int, int, int],
        pca_rwn: int = 32,
        pca_tcf: int = 32,
        pca_sig: int = 128,
        whiten: bool = False,
        random_state: int = 42,
    ) -> None:
        self.slices: dict[str, slice] = {}
        self.scalers: dict[str, StandardScaler] = {}
        self.pcas: dict[str, PCA] = {}
        self.pca_cfg = {"rwn": pca_rwn, "tcf": pca_tcf, "sig": pca_sig}
        self.whiten = whiten
        self.random_state = random_state

        d_rwn, d_tcf, d_sig = dims
        cur = 0
        self.slices["rwn"] = slice(cur, cur + d_rwn)
        cur += d_rwn
        self.slices["tcf"] = slice(cur, cur + d_tcf)
        cur += d_tcf
        if d_sig > 0:
            self.slices["sig"] = slice(cur, cur + d_sig)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        outs = []
        for key, sl in self.slices.items():
            Xi = X[:, sl]
            scaler = StandardScaler().fit(Xi)
            Zi = scaler.transform(Xi)
            self.scalers[key] = scaler

            n_comp = self.pca_cfg.get(key)
            if n_comp and 0 < n_comp < Zi.shape[1]:
                pca = PCA(
                    n_components=n_comp,
                    whiten=self.whiten,
                    random_state=self.random_state,
                ).fit(Zi)
                Zi = pca.transform(Zi)
                self.pcas[key] = pca

            outs.append(Zi)
        return np.concatenate(outs, axis=1)

    def transform(self, X: np.ndarray) -> np.ndarray:
        outs = []
        for key, sl in self.slices.items():
            Xi = X[:, sl]
            Zi = self.scalers[key].transform(Xi)
            pca = self.pcas.get(key)
            if pca is not None:
                Zi = pca.transform(Zi)
            outs.append(Zi)
        return np.concatenate(outs, axis=1)


# ----------------------------------------------------------------------------
# CNN di fusione
# ----------------------------------------------------------------------------


def _prepare_cnn_tensors(
    rwn: np.ndarray,
    tcf: np.ndarray,
    sig: np.ndarray | None,
    *,
    target_len: int,
    use_signal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Prepara tensori (B, C, T) per modelli sequenziali (CNN/TCN)."""

    rwn = _ensure_batch_first(rwn)
    tcf = _ensure_batch_first(tcf)
    rwn = _linresample_time(rwn, target_len)
    tcf = _linresample_time(tcf, target_len)

    rwn_t = torch.from_numpy(np.transpose(rwn, (0, 2, 1)))
    tcf_t = torch.from_numpy(np.transpose(tcf, (0, 2, 1)))

    sig_t: torch.Tensor | None = None
    if use_signal and sig is not None and sig.size:
        sig_bt = np.transpose(sig, (0, 2, 1))
        sig_bt = _linresample_time(sig_bt, target_len)
        sig_t = torch.from_numpy(np.transpose(sig_bt, (0, 2, 1)))

    return rwn_t, tcf_t, sig_t


class SimpleCNNMeta(nn.Module):
    """CNN piccola per fusione: 3 rami 1x1, concat, 2 conv, GAP, FC."""

    def __init__(
        self,
        c_rwn: int,
        c_tcf: int,
        c_sig: int | None,
        n_classes: int,
        emb: int = 8,
        mid: int = 32,
        p_drop: float = 0.1,
    ) -> None:
        super().__init__()
        self.has_sig = (c_sig is not None) and (c_sig > 0)
        self.rwn_in = nn.Conv1d(c_rwn, emb, kernel_size=1)
        self.tcf_in = nn.Conv1d(c_tcf, emb, kernel_size=1)
        if self.has_sig:
            assert c_sig is not None
            self.sig_in = nn.Conv1d(c_sig, emb, kernel_size=1)

        c_tot = emb * (3 if self.has_sig else 2)
        self.backbone = nn.Sequential(
            nn.Conv1d(c_tot, mid, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Conv1d(mid, mid, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
        )
        self.head = nn.Linear(mid, n_classes)

    def forward(
        self,
        rwn: torch.Tensor,
        tcf: torch.Tensor,
        sig: torch.Tensor | None,
    ) -> torch.Tensor:
        z = [self.rwn_in(rwn), self.tcf_in(tcf)]
        if self.has_sig and sig is not None:
            z.append(self.sig_in(sig))

        x = torch.cat(z, dim=1)
        x = self.backbone(x)
        x = x.mean(dim=-1)
        return self.head(x)


def _compute_class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(y.astype(np.int64), minlength=n_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (n_classes * counts)
    return torch.from_numpy(weights)


class TemporalConvBlock(nn.Module):
    """Blocco TCN con due conv dilatate e residual connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        out = F.relu(self.bn1(out), inplace=True)
        out = self.dropout(out)
        out = self.conv2(out)
        out = F.relu(self.bn2(out), inplace=True)
        out = self.dropout(out)
        return out + residual


class SimpleTCNMeta(nn.Module):
    """TCN compatta per fondere logits e segnale."""

    def __init__(
        self,
        c_rwn: int,
        c_tcf: int,
        c_sig: int | None,
        n_classes: int,
        *,
        emb: int = 8,
        hidden: int = 64,
        levels: int = 4,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.has_sig = (c_sig is not None) and (c_sig > 0)
        self.rwn_in = nn.Conv1d(c_rwn, emb, kernel_size=1)
        self.tcf_in = nn.Conv1d(c_tcf, emb, kernel_size=1)
        if self.has_sig:
            assert c_sig is not None
            self.sig_in = nn.Conv1d(c_sig, emb, kernel_size=1)

        c_tot = emb * (3 if self.has_sig else 2)
        blocks: list[nn.Module] = []
        in_ch = c_tot
        for level in range(levels):
            blocks.append(
                TemporalConvBlock(
                    in_ch,
                    hidden,
                    kernel_size=kernel_size,
                    dilation=2**level,
                    dropout=dropout,
                )
            )
            in_ch = hidden
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden, n_classes)

    def forward(
        self,
        rwn: torch.Tensor,
        tcf: torch.Tensor,
        sig: torch.Tensor | None,
    ) -> torch.Tensor:
        z = [self.rwn_in(rwn), self.tcf_in(tcf)]
        if self.has_sig and sig is not None:
            z.append(self.sig_in(sig))

        x = torch.cat(z, dim=1)
        x = self.tcn(x)
        x = x.mean(dim=-1)
        return self.head(x)


class _MetaDataset(Dataset):
    """Dataset ausiliario che restituisce triple (rwn, tcf, sig) + label."""

    def __init__(
        self,
        r: torch.Tensor,
        t: torch.Tensor,
        s: torch.Tensor | None,
        y: np.ndarray,
    ) -> None:
        self.r = r
        self.t = t
        self.s = s
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, idx: int):
        sig = None if self.s is None else self.s[idx]
        return self.r[idx], self.t[idx], sig, self.y[idx]


def _collate_meta_batch(batch):
    r = torch.stack([b[0] for b in batch], 0)
    t = torch.stack([b[1] for b in batch], 0)
    sigs = [b[2] for b in batch]
    s = None if sigs[0] is None else torch.stack(sigs, 0)
    y = torch.stack([b[3] for b in batch], 0)
    return r, t, s, y


def _make_meta_loaders(
    rwn_tr: torch.Tensor,
    tcf_tr: torch.Tensor,
    sig_tr: torch.Tensor | None,
    y_tr: np.ndarray,
    rwn_te: torch.Tensor,
    tcf_te: torch.Tensor,
    sig_te: torch.Tensor | None,
    y_te: np.ndarray,
    *,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    train_ds = _MetaDataset(rwn_tr, tcf_tr, sig_tr, y_tr)
    test_ds = _MetaDataset(rwn_te, tcf_te, sig_te, y_te)
    tr_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=_collate_meta_batch,
    )
    te_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate_meta_batch,
    )
    return tr_loader, te_loader


def _train_meta_network(
    model: nn.Module,
    train_loader: DataLoader,
    *,
    epochs: int,
    class_weights: torch.Tensor,
    device: torch.device,
    lr: float,
    weight_decay: float,
    sid: int,
    meta_name: str,
) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    for epoch in range(epochs):
        tot_loss = 0.0
        for r, t, s, y in train_loader:
            r = r.to(device)
            t = t.to(device)
            y = y.to(device)
            s = s.to(device) if s is not None else None

            optimizer.zero_grad()
            logits = model(r, t, s)
            loss = F.cross_entropy(logits, y, weight=class_weights)
            loss.backward()
            optimizer.step()
            tot_loss += float(loss.detach().cpu())

        if (epoch + 1) % max(1, epochs // 5) == 0:
            avg_loss = tot_loss / max(1, len(train_loader))
            print(
                f"[S{sid}] {meta_name} epoch {epoch + 1}/{epochs}"
                f" - loss {avg_loss:.4f}"
            )


@torch.no_grad()
def _predict_meta_network(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    preds: list[np.ndarray] = []
    for r, t, s, _ in loader:
        r = r.to(device)
        t = t.to(device)
        s = s.to(device) if s is not None else None
        logits = model(r, t, s)
        preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds, axis=0)


# ----------------------------------------------------------------------------
# Config e soggetti
# ----------------------------------------------------------------------------


def _load_config_and_subjects(args):
    cfg_dir = Path(__file__).resolve().parent / "configs"
    cfg_path = cfg_dir / "ratiowavenet.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config non trovato: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.loso:
        config["dataset_name"] = args.dataset + "_loso"
        config["max_epochs"] = (
            config["max_epochs_loso_hgd"]
            if args.dataset == "hgd"
            else config["max_epochs_loso"]
        )
        config["model_kwargs"]["warmup_epochs"] = config["model_kwargs"][
            "warmup_epochs_loso"
        ]
    else:
        config["dataset_name"] = args.dataset
        config["max_epochs"] = (
            config["max_epochs_2b"]
            if args.dataset == "bcic2b"
            else config["max_epochs"]
        )

    config["preprocessing"] = config["preprocessing"][args.dataset]
    config["preprocessing"]["z_scale"] = config["z_scale"]

    if args.interaug:
        config["preprocessing"]["interaug"] = True
    elif args.no_interaug:
        config["preprocessing"]["interaug"] = False
    else:
        config["preprocessing"]["interaug"] = config["interaug"]
    config.pop("interaug", None)

    if args.seed is not None:
        config["seed"] = args.seed

    datamodule_cls = get_datamodule_cls(config["dataset_name"])
    subj_cfg = args.subject_ids or "all"
    subject_ids = (
        datamodule_cls.all_subject_ids
        if subj_cfg == "all"
        else [int(subj_cfg)]
        if subj_cfg.isdigit()
        else [int(x) for x in subj_cfg.split(",")]
    )

    return config, datamodule_cls, subject_ids


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="bcic2a")
    parser.add_argument("--loso", action="store_true", default=False)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--interaug", action="store_true")
    parser.add_argument("--no_interaug", action="store_true")
    parser.add_argument("--subject-ids", type=str, default="all")

    parser.add_argument(
        "--rwn-pattern",
        type=str,
        default=None,
        help=(
            "Pattern ckpt RatioWaveNet (usa {sid}/{subject_id}/{dataset}/{dataset_folder})."
            " Se omesso, autodetect."
        ),
    )
    parser.add_argument(
        "--tcf-pattern",
        type=str,
        default=None,
        help=(
            "Pattern ckpt TCFormer (usa {sid}/{subject_id}/{dataset}/{dataset_folder})."
            " Se omesso, autodetect."
        ),
    )

    parser.add_argument("--meta", type=str, default="rf", choices=["rf", "cnn", "tcn"])
    parser.add_argument("--use-signal", action="store_true")
    parser.add_argument("--target-length", type=int, default=64)
    parser.add_argument("--pca-rwn", type=int, default=32)
    parser.add_argument("--pca-tcf", type=int, default=32)
    parser.add_argument("--pca-sig", type=int, default=128)
    parser.add_argument("--whiten-pca", action="store_true")
    parser.add_argument("--rf-n-estimators", type=int, default=400)
    parser.add_argument("--rf-max-depth", type=int, default=None)
    parser.add_argument("--rf-min-samples-leaf", type=int, default=1)

    parser.add_argument("--cnn-epochs", type=int, default=20)
    parser.add_argument("--cnn-batch", type=int, default=128)
    parser.add_argument("--cnn-emb", type=int, default=8)
    parser.add_argument("--cnn-mid", type=int, default=32)
    parser.add_argument("--cnn-drop", type=float, default=0.1)
    parser.add_argument("--tcn-epochs", type=int, default=30)
    parser.add_argument("--tcn-batch", type=int, default=128)
    parser.add_argument("--tcn-emb", type=int, default=8)
    parser.add_argument("--tcn-hidden", type=int, default=64)
    parser.add_argument("--tcn-levels", type=int, default=4)
    parser.add_argument("--tcn-kernel", type=int, default=5)
    parser.add_argument("--tcn-drop", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)

    parser.add_argument("--results-dir", type=str, default="results/ensemble_subjectwise")
    args = parser.parse_args()

    device = torch.device(
        f"cuda:{args.gpu_id}" if args.gpu_id >= 0 and torch.cuda.is_available() else "cpu"
    )
    seed_everything(args.seed if args.seed is not None else 0)

    config, datamodule_cls, subject_ids = _load_config_and_subjects(args)

    seed_value = args.seed if args.seed is not None else config.get("seed", 0)
    _ensure_required_checkpoints(args, datamodule_cls, subject_ids, seed_value)
    seed_dir_component = _seed_folder_name(seed_value)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "per_subject_metrics.csv"

    n_workers = config["preprocessing"].get(
        "num_workers", max(1, (os.cpu_count() or 4) // 2)
    )
    batch_size = config["preprocessing"]["batch_size"]

    ModelRWN = get_model_cls("RatioWaveNet")
    ModelTCF = get_model_cls("TCFormer")

    all_rows: list[dict[str, object]] = []

    for sid in subject_ids:
        print(f"\n>>> Ensemble per soggetto {sid}")

        rwn_candidates = [
            args.rwn_pattern,
            f"results/ensemble_checkpoints/{{dataset_folder}}/{seed_dir_component}/ratiowavenet_subject_{{sid}}.ckpt",
            f"results/ensemble_checkpoints/{{dataset}}/{seed_dir_component}/ratiowavenet_subject_{{sid}}.ckpt",
            "results/ensemble_checkpoints/{dataset_folder}/ratiowavenet_subject_{sid}.ckpt",
            "results/ensemble_checkpoints/{dataset}/ratiowavenet_subject_{sid}.ckpt",
            "results/ensemble_checkpoints/ratiowavenet_subject_{sid}.ckpt",
        ]
        tcf_candidates = [
            args.tcf_pattern,
            f"results/ensemble_checkpoints/{{dataset_folder}}/{seed_dir_component}/tcformer_subject_{{sid}}.ckpt",
            f"results/ensemble_checkpoints/{{dataset}}/{seed_dir_component}/tcformer_subject_{{sid}}.ckpt",
            "results/ensemble_checkpoints/{dataset_folder}/tcformer_subject_{sid}.ckpt",
            "results/ensemble_checkpoints/{dataset}/tcformer_subject_{sid}.ckpt",
            "results/ensemble_checkpoints/tcformer_subject_{sid}.ckpt",
        ]

        rwn_ckpt = _find_ckpt_any(rwn_candidates, sid, args.dataset, "RatioWaveNet")
        tcf_ckpt = _find_ckpt_any(tcf_candidates, sid, args.dataset, "TCFormer")
        print(f"[S{sid}] RWN ckpt: {rwn_ckpt}")
        print(f"[S{sid}] TCF ckpt: {tcf_ckpt}")

        rwn = ModelRWN.load_from_checkpoint(str(rwn_ckpt), map_location=device).to(device)
        tcf = ModelTCF.load_from_checkpoint(str(tcf_ckpt), map_location=device).to(device)
        rwn.eval()
        tcf.eval()

        dm = datamodule_cls(config["preprocessing"], subject_id=sid)
        dm.prepare_data()
        dm.setup()

        rw_tr, tc_tr, sig_tr, y_tr = _collect_sources(
            rwn,
            tcf,
            dm.train_dataset,
            batch_size=batch_size,
            num_workers=n_workers,
            device=device,
            return_signal=args.use_signal,
        )
        rw_te, tc_te, sig_te, y_te = _collect_sources(
            rwn,
            tcf,
            dm.test_dataset,
            batch_size=batch_size,
            num_workers=n_workers,
            device=device,
            return_signal=args.use_signal,
        )

        n_classes = int(max(y_tr.max(initial=0), y_te.max(initial=0))) + 1

        if args.meta == "rf":
            X_tr, dims = _make_rf_features(
                rw_tr,
                tc_tr,
                sig_tr,
                target_len=args.target_length,
                use_signal=args.use_signal,
            )
            X_te, _ = _make_rf_features(
                rw_te,
                tc_te,
                sig_te,
                target_len=args.target_length,
                use_signal=args.use_signal,
            )

            scaler_pca = RFBlockScalerPCA(
                dims,
                args.pca_rwn,
                args.pca_tcf,
                args.pca_sig,
                args.whiten_pca,
                random_state=args.seed if args.seed is not None else 42,
            )
            Z_tr = scaler_pca.fit_transform(X_tr)
            Z_te = scaler_pca.transform(X_te)

            clf = RandomForestClassifier(
                n_estimators=args.rf_n_estimators,
                max_depth=args.rf_max_depth,
                min_samples_leaf=args.rf_min_samples_leaf,
                class_weight="balanced",
                random_state=args.seed if args.seed is not None else 42,
                n_jobs=-1,
            )
            clf.fit(Z_tr, y_tr.astype(np.int64))
            y_pred = clf.predict(Z_te)

        elif args.meta in {"cnn", "tcn"}:
            if args.target_length is None:
                raise ValueError(
                    "Per i meta-modelli sequenziali specificare --target-length."
                )

            rwn_tr_t, tcf_tr_t, sig_tr_t = _prepare_cnn_tensors(
                rw_tr,
                tc_tr,
                sig_tr,
                target_len=args.target_length,
                use_signal=args.use_signal,
            )
            rwn_te_t, tcf_te_t, sig_te_t = _prepare_cnn_tensors(
                rw_te,
                tc_te,
                sig_te,
                target_len=args.target_length,
                use_signal=args.use_signal,
            )

            c_rwn, c_tcf = rwn_tr_t.shape[1], tcf_tr_t.shape[1]
            c_sig = sig_tr_t.shape[1] if (args.use_signal and sig_tr_t is not None) else None

            if args.meta == "cnn":
                model = SimpleCNNMeta(
                    c_rwn,
                    c_tcf,
                    c_sig,
                    n_classes,
                    emb=args.cnn_emb,
                    mid=args.cnn_mid,
                    p_drop=args.cnn_drop,
                ).to(device)
                epochs = args.cnn_epochs
                batch_sz = args.cnn_batch
                meta_name = "CNN"
            else:
                model = SimpleTCNMeta(
                    c_rwn,
                    c_tcf,
                    c_sig,
                    n_classes,
                    emb=args.tcn_emb,
                    hidden=args.tcn_hidden,
                    levels=args.tcn_levels,
                    kernel_size=args.tcn_kernel,
                    dropout=args.tcn_drop,
                ).to(device)
                epochs = args.tcn_epochs
                batch_sz = args.tcn_batch
                meta_name = "TCN"

            tr_loader, te_loader = _make_meta_loaders(
                rwn_tr_t,
                tcf_tr_t,
                sig_tr_t,
                y_tr,
                rwn_te_t,
                tcf_te_t,
                sig_te_t,
                y_te,
                batch_size=batch_sz,
            )

            class_w = _compute_class_weights(y_tr, n_classes).to(device)
            _train_meta_network(
                model,
                tr_loader,
                epochs=epochs,
                class_weights=class_w,
                device=device,
                lr=args.lr,
                weight_decay=args.weight_decay,
                sid=sid,
                meta_name=meta_name,
            )
            y_pred = _predict_meta_network(model, te_loader, device=device)

        else:
            raise ValueError(f"Meta-modello non supportato: {args.meta}")

        y_true = y_te.reshape(-1).astype(np.int64)
        acc = accuracy_score(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)
        print(
            f"Subject {sid} | Test Accuracy: {acc:.4f} | "
            f"Kappa: {kappa:.4f} | meta={args.meta}"
        )

        all_rows.append(
            {
                "subject_id": int(sid),
                "accuracy": float(acc),
                "kappa": float(kappa),
                "meta": args.meta,
                "use_signal": bool(args.use_signal),
                "target_length": args.target_length,
            }
        )

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["subject_id", "accuracy", "kappa", "meta", "use_signal", "target_length"]
        )
        writer.writeheader()
        writer.writerows(all_rows)

    accs = [row["accuracy"] for row in all_rows]
    kappas = [row["kappa"] for row in all_rows]
    if accs:
        print(
            f"\n> Mean Accuracy: {np.mean(accs):.4f}"
            f" | Mean Kappa: {np.mean(kappas):.4f}"
            f" | N={len(accs)}"
        )
        print(f"> Saved: {csv_path}")


if __name__ == "__main__":
    main()