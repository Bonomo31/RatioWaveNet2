"""Gestione automatica dei checkpoint per l'ensembler.

Il modulo fornisce utility per garantire che i checkpoint dei modelli
di base siano disponibili quando si lancia ``ensemble_pipeline.py``:

* crea la gerarchia ``results/ensemble_checkpoints/<dataset>/sid<seed>/``;
* ricicla eventuali file salvati con layout legacy, copiandoli nella
  nuova struttura;
* avvia ``train_pipeline.train_and_test`` per addestrare RatioWaveNet e
  TCFormer sui soggetti mancanti e copia i checkpoint generati.
"""

from __future__ import annotations

import copy
import shutil
from pathlib import Path
from typing import Iterable

import yaml

from train_pipeline import CONFIG_DIR as TP_CONFIG_DIR, train_and_test


__all__ = [
    "dataset_folder_name",
    "seed_folder_name",
    "ensure_checkpoints_ready",
]


def dataset_folder_name(dataset: str) -> str:
    """Normalizza il nome dataset per l'albero dei checkpoint."""

    mapping = {"bcic2a": "bci2a", "bcic2b": "bci2b"}
    base = dataset.replace("_loso", "")
    return mapping.get(base, base)


def seed_folder_name(seed: int) -> str:
    return f"sid{seed}"


def _format_pattern(pattern: str, sid: int, dataset: str) -> Path:
    dataset_key = dataset_folder_name(dataset)
    return Path(
        pattern.format(
            sid=sid,
            subject_id=sid,
            dataset=dataset,
            dataset_folder=dataset_key,
        )
    )


def _target_checkpoint_dir(dataset: str, seed: int) -> Path:
    return (
        Path("results/ensemble_checkpoints")
        / dataset_folder_name(dataset)
        / seed_folder_name(seed)
    )


def _maybe_copy_from_legacy(
    prefix: str, dataset: str, subject_id: int, target_path: Path
) -> bool:
    """Riusa checkpoint esistenti con layout precedente, se presenti."""

    dataset_key = dataset_folder_name(dataset)
    legacy_patterns: Iterable[str] = (
        f"results/ensemble_checkpoints/{dataset_key}/{prefix}_subject_{{sid}}.ckpt",
        f"results/ensemble_checkpoints/{dataset}/{prefix}_subject_{{sid}}.ckpt",
        f"results/ensemble_checkpoints/{prefix}_subject_{{sid}}.ckpt",
    )

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
        raise FileNotFoundError(
            f"Config per il modello '{model_name}' non trovata: {cfg_path}"
        )

    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config = copy.deepcopy(config)
    config["model"] = model_name

    dataset_key = args.dataset
    dataset_name = dataset_key + "_loso" if args.loso else dataset_key
    config["dataset_name"] = dataset_name

    if args.loso:
        if dataset_key == "hgd":
            config["max_epochs"] = config.get("max_epochs_loso_hgd", config["max_epochs"])
        else:
            config["max_epochs"] = config.get("max_epochs_loso", config["max_epochs"])
        model_kwargs = config.get("model_kwargs", {})
        if "warmup_epochs_loso" in model_kwargs:
            model_kwargs["warmup_epochs"] = model_kwargs["warmup_epochs_loso"]
        config["model_kwargs"] = model_kwargs
    else:
        if dataset_key == "bcic2b" and "max_epochs_2b" in config:
            config["max_epochs"] = config["max_epochs_2b"]

    preprocess_cfg = copy.deepcopy(config["preprocessing"][dataset_key])
    preprocess_cfg["z_scale"] = config.get("z_scale", True)
    if getattr(args, "interaug", False):
        preprocess_cfg["interaug"] = True
    elif getattr(args, "no_interaug", False):
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

    config.setdefault("model_kwargs", {})
    config["save_checkpoint"] = True
    return config


def _locate_training_run_dir(config) -> Path:
    results_root = Path(__file__).resolve().parent.parent / "results"
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
    train_config = _build_training_config(
        model_name, args, datamodule_cls, subject_ids, seed_value
    )

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

    _train_and_collect_checkpoints(
        model_name, prefix, args, datamodule_cls, missing, seed_value, target_dir
    )


def ensure_checkpoints_ready(
    args,
    datamodule_cls,
    subject_ids: list[int],
    seed_value: int,
) -> None:
    """Garantisce che i checkpoint necessari siano disponibili."""

    _ensure_model_checkpoints(
        "ratiowavenet", "ratiowavenet", args, datamodule_cls, subject_ids, seed_value
    )
    _ensure_model_checkpoints(
        "tcformer", "tcformer", args, datamodule_cls, subject_ids, seed_value
    )
