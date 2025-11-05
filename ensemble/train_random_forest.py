"""Train a RandomForest ensemble using either cached predictions or live models."""
from __future__ import annotations

import argparse
import os
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import numpy as np
import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer

from .random_forest import RandomForestSignalEnsembler
from utils.get_datamodule_cls import get_datamodule_cls
from utils.get_model_cls import get_model_cls
from utils.seed import seed_everything


def _load_npz(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    required = {"ratiowavenet", "tcformer", "signals", "labels"}
    missing = required.difference(data.files)
    if missing:
        raise KeyError(
            f"File {path} is missing the following arrays: {', '.join(sorted(missing))}."
        )
    return (
        data["ratiowavenet"],
        data["tcformer"],
        data["signals"],
        data["labels"],
    )


def _parse_subject_ids(
    config_value: Sequence[int] | int | str,
    override: str | None,
    *,
    datamodule_subjects: Sequence[int],
) -> List[int]:
    if override is not None:
        value = override.strip().lower()
        if value == "all":
            return list(datamodule_subjects)
        ids = [int(chunk) for chunk in override.split(",") if chunk.strip()]
        if not ids:
            raise ValueError("subject_ids override must specify at least one subject")
        return ids

    if isinstance(config_value, str):
        if config_value.lower() == "all":
            return list(datamodule_subjects)
        return [int(config_value)]
    if isinstance(config_value, int):
        return [config_value]
    return list(config_value)


def _load_data_configuration(args: argparse.Namespace) -> Tuple[
    str,
    str,
    dict,
    List[int],
    int,
    int,
]:
    if args.config is None:
        raise ValueError("--config is required when running the ensemble from models.")

    with open(args.config) as handle:
        raw_config = yaml.safe_load(handle)

    base_dataset = args.dataset_name or raw_config.get("dataset_name")
    if not base_dataset:
        raise ValueError(
            "Dataset name missing. Provide --dataset-name or set dataset_name in the config."
        )

    if args.loso:
        dataset_name = f"{base_dataset}_loso"
    else:
        dataset_name = base_dataset

    if base_dataset not in raw_config.get("preprocessing", {}):
        raise KeyError(
            f"Config {args.config} does not provide preprocessing settings for '{base_dataset}'."
        )

    preprocessing = deepcopy(raw_config["preprocessing"][base_dataset])
    preprocessing["z_scale"] = raw_config.get("z_scale", False)

    if args.interaug:
        preprocessing["interaug"] = True
    elif args.no_interaug:
        preprocessing["interaug"] = False
    else:
        preprocessing["interaug"] = raw_config.get("interaug", False)

    seed = args.seed if args.seed is not None else raw_config.get("seed", 0)
    preprocessing["seed"] = seed

    datamodule_cls = get_datamodule_cls(dataset_name)
    subject_ids = _parse_subject_ids(
        raw_config.get("subject_ids", "all"),
        args.subject_ids,
        datamodule_subjects=getattr(datamodule_cls, "all_subject_ids", []),
    )
    if not subject_ids:
        raise ValueError(
            "Unable to determine the list of subjects. Check the config file or provide --subject-ids."
        )

    batch_size = preprocessing.get("batch_size")
    if batch_size is None:
        raise KeyError(
            "The preprocessing configuration must define a batch_size for the datamodule."
        )

    cpu_count = os.cpu_count() or 0
    default_workers = cpu_count // 2 if cpu_count else 0
    num_workers = preprocessing.get("num_workers", default_workers)

    return dataset_name, base_dataset, preprocessing, subject_ids, seed, int(num_workers)


def _load_model_training_config(
    config_path: Path,
    *,
    base_dataset: str,
    loso: bool,
) -> Dict:
    with open(config_path) as handle:
        raw_config = yaml.safe_load(handle)

    config = deepcopy(raw_config)

    if loso:
        if base_dataset == "hgd":
            config["max_epochs"] = config.get("max_epochs_loso_hgd", config["max_epochs"])
        else:
            config["max_epochs"] = config.get("max_epochs_loso", config["max_epochs"])
        model_kwargs = config.get("model_kwargs", {})
        if "warmup_epochs_loso" in model_kwargs:
            model_kwargs["warmup_epochs"] = model_kwargs["warmup_epochs_loso"]
    else:
        if base_dataset == "bcic2b":
            config["max_epochs"] = config.get("max_epochs_2b", config["max_epochs"])

    return config


def _build_loader(dataset, batch_size: int, num_workers: int, device: torch.device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(0, num_workers),
        pin_memory=device.type == "cuda",
    )


def _train_lightning_model(
    model_name: str,
    model_config: Dict,
    datamodule_cls,
    preprocessing: dict,
    *,
    subject_id: int,
    device: torch.device,
    checkpoint_dir: Path,
    seed: int,
) -> Tuple[torch.nn.Module, Path]:
    print(f"[Training] Avvio addestramento {model_name} per il soggetto {subject_id}...")

    datamodule = datamodule_cls(deepcopy(preprocessing), subject_id=subject_id)
    datamodule.prepare_data()
    datamodule.setup()

    model_cls = get_model_cls(model_name)
    model_kwargs = deepcopy(model_config.get("model_kwargs", {}))
    model_kwargs["n_channels"] = datamodule_cls.channels
    model_kwargs["n_classes"] = datamodule_cls.classes
    max_epochs = int(model_config.get("max_epochs", 1))

    model = model_cls(**model_kwargs, max_epochs=max_epochs)

    seed_everything(seed)
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
    )

    trainer.fit(model, datamodule=datamodule)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / f"{model_name.lower()}_subject_{subject_id}.ckpt"
    trainer.save_checkpoint(ckpt_path)
    print(f"[Training] Checkpoint {model_name} salvato in {ckpt_path}")

    model.to(device).eval()
    return model, ckpt_path


def _collect_sources(
    rw_model: torch.nn.Module,
    tc_model: torch.nn.Module,
    dataset,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    expected_channels: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    loader = _build_loader(dataset, batch_size, num_workers, device)

    rw_batches: List[np.ndarray] = []
    tc_batches: List[np.ndarray] = []
    signals: List[np.ndarray] = []
    labels: List[np.ndarray] = []

    for inputs, targets in loader:
        if inputs.ndim == 4 and inputs.shape[1] == 1:
            inputs = inputs[:, 0]
        elif inputs.ndim != 3:
            raise ValueError(
                "Il dataset deve restituire tensori con forma (batch, canali, tempo). "
                f"Ricevuto {tuple(inputs.shape)}."
            )

        if inputs.shape[1] != expected_channels and inputs.shape[2] == expected_channels:
            inputs = inputs.transpose(1, 2)

        if inputs.shape[1] != expected_channels:
            raise ValueError(
                "Numero di canali non compatibile con il modello: "
                f"attesi {expected_channels}, trovati {inputs.shape[1]}"
            )

        signals.append(inputs.numpy())
        labels.append(targets.numpy())
        with torch.no_grad():
            tensor_inputs = inputs.to(device)
            rw_logits = rw_model(tensor_inputs)
            tc_logits = tc_model(tensor_inputs)
        rw_batches.append(rw_logits.detach().cpu().numpy())
        tc_batches.append(tc_logits.detach().cpu().numpy())

    return (
        np.concatenate(rw_batches, axis=0) if rw_batches else np.empty((0,)),
        np.concatenate(tc_batches, axis=0) if tc_batches else np.empty((0,)),
        np.concatenate(signals, axis=0) if signals else np.empty((0,)),
        np.concatenate(labels, axis=0) if labels else np.empty((0,)),
    )


def _run_from_models(args: argparse.Namespace) -> None:
    dataset_name, base_dataset, preprocessing, subject_ids, seed, num_workers = _load_data_configuration(args)
    datamodule_cls = get_datamodule_cls(dataset_name)

    device = torch.device(args.model_device)
    seed_everything(seed)

    rw_config = None
    tc_config = None
    checkpoint_dir = args.checkpoint_dir or Path("results/ensemble_checkpoints")

    saved_checkpoints = {"RatioWaveNet": [], "TCFormer": []}

    train_rw, train_tc, train_signals, train_labels = [], [], [], []
    test_entries: List[Dict[str, np.ndarray]] = []

    for subject_id in subject_ids:
        if args.ratiowavenet_ckpt is not None:
            rw_model = get_model_cls("RatioWaveNet").load_from_checkpoint(
                args.ratiowavenet_ckpt, map_location=device
            )
            rw_model.to(device).eval()
        else:
            if rw_config is None:
                if args.ratiowavenet_train_config is None:
                    raise ValueError(
                        "Specifica --ratiowavenet-train-config quando manca --ratiowavenet-ckpt."
                    )
                rw_config = _load_model_training_config(
                    args.ratiowavenet_train_config,
                    base_dataset=base_dataset,
                    loso=args.loso,
                )
            rw_model, rw_ckpt_path = _train_lightning_model(
                "RatioWaveNet",
                rw_config,
                datamodule_cls,
                preprocessing,
                subject_id=subject_id,
                device=device,
                checkpoint_dir=checkpoint_dir,
                seed=seed,
            )
            saved_checkpoints["RatioWaveNet"].append(rw_ckpt_path)

        if args.tcformer_ckpt is not None:
            tc_model = get_model_cls("TCFormer").load_from_checkpoint(
                args.tcformer_ckpt, map_location=device
            )
            tc_model.to(device).eval()
        else:
            if tc_config is None:
                if args.tcformer_train_config is None:
                    raise ValueError(
                        "Specifica --tcformer-train-config quando manca --tcformer-ckpt."
                    )
                tc_config = _load_model_training_config(
                    args.tcformer_train_config,
                    base_dataset=base_dataset,
                    loso=args.loso,
                )
            tc_model, tc_ckpt_path = _train_lightning_model(
                "TCFormer",
                tc_config,
                datamodule_cls,
                preprocessing,
                subject_id=subject_id,
                device=device,
                checkpoint_dir=checkpoint_dir,
                seed=seed,
            )
            saved_checkpoints["TCFormer"].append(tc_ckpt_path)

        datamodule = datamodule_cls(deepcopy(preprocessing), subject_id=subject_id)
        datamodule.prepare_data()
        datamodule.setup()

        if datamodule.train_dataset is not None:
            rw, tc, signals, labels = _collect_sources(
                rw_model,
                tc_model,
                datamodule.train_dataset,
                batch_size=preprocessing["batch_size"],
                num_workers=num_workers,
                device=device,
                expected_channels=datamodule_cls.channels,
            )
            if signals.size:
                train_rw.append(rw)
                train_tc.append(tc)
                train_signals.append(signals)
                train_labels.append(labels)

        if datamodule.test_dataset is not None:
            rw, tc, signals, labels = _collect_sources(
                rw_model,
                tc_model,
                datamodule.test_dataset,
                batch_size=preprocessing["batch_size"],
                num_workers=num_workers,
                device=device,
                expected_channels=datamodule_cls.channels,
            )
            if signals.size:
                test_entries.append(
                    {
                        "subject_id": subject_id,
                        "ratiowavenet": rw,
                        "tcformer": tc,
                        "signals": signals,
                        "labels": labels,
                    }
                )

    if not train_signals or not train_labels:
        raise RuntimeError("No training samples collected from the provided datamodule configuration.")
    if not test_entries:
        raise RuntimeError("No test samples collected from the provided datamodule configuration.")

    rw_train = np.concatenate(train_rw, axis=0)
    tc_train = np.concatenate(train_tc, axis=0)
    signals_train = np.concatenate(train_signals, axis=0)
    labels_train = np.concatenate(train_labels, axis=0)

    ensemble = RandomForestSignalEnsembler(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        target_length=args.target_length,
        random_state=args.random_state,
    )
    ensemble.fit(rw_train, tc_train, signals_train, labels_train)

    print("Accuratezza per soggetto:")
    subject_metrics: List[Tuple[int, float]] = []
    all_predictions: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    for entry in test_entries:
        subject_preds = ensemble.predict(
            entry["ratiowavenet"], entry["tcformer"], entry["signals"]
        )
        labels = entry["labels"]
        accuracy = float((subject_preds == labels).mean())
        subject_metrics.append((entry["subject_id"], accuracy))
        all_predictions.append(subject_preds)
        all_labels.append(labels)
        print(f"  - Soggetto {entry['subject_id']}: {accuracy:.4f}")

    predictions = np.concatenate(all_predictions, axis=0)
    labels_test = np.concatenate(all_labels, axis=0)

    print("\nClassification report:\n", classification_report(labels_test, predictions))
    print("Confusion matrix:\n", confusion_matrix(labels_test, predictions))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": ensemble,
                "metadata": {
                    "dataset_name": dataset_name,
                    "target_length": ensemble.effective_length,
                    "subjects": subject_ids,
                    "seed": seed,
                    "saved_checkpoints": {
                        key: [str(path) for path in paths]
                        for key, paths in saved_checkpoints.items()
                        if paths
                    },
                    "per_subject_accuracy": {
                        subject_id: accuracy
                        for subject_id, accuracy in subject_metrics
                    },
                },
            },
            args.output,
        )
        print(f"Ensemble saved to {args.output}")

    if saved_checkpoints["RatioWaveNet"] or saved_checkpoints["TCFormer"]:
        print("Checkpoint generati durante l'esecuzione:")
        for model_name, paths in saved_checkpoints.items():
            for path in paths:
                print(f"  - {model_name}: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Optional path to an .npz file containing ratiowavenet, tcformer, signals and labels arrays.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML configuration used to instantiate the datamodule when computing predictions on-the-fly.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Dataset key to pick from the YAML configuration (e.g. bcic2a).",
    )
    parser.add_argument("--loso", action="store_true", help="Use leave-one-subject-out datamodule variants.")
    parser.add_argument(
        "--subject-ids",
        type=str,
        default=None,
        help="Override the list of subjects to use. Accepts 'all', '3' or '1,2,5'.",
    )
    parser.add_argument("--interaug", action="store_true", help="Enable inter-trial augmentation in the datamodule.")
    parser.add_argument("--no-interaug", action="store_true", help="Disable inter-trial augmentation in the datamodule.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed override.")
    parser.add_argument(
        "--ratiowavenet-ckpt",
        type=Path,
        default=None,
        help="Checkpoint Lightning già addestrato per RatioWaveNet.",
    )
    parser.add_argument(
        "--tcformer-ckpt",
        type=Path,
        default=None,
        help="Checkpoint Lightning già addestrato per TCFormer.",
    )
    parser.add_argument(
        "--ratiowavenet-train-config",
        type=Path,
        default=Path("configs/ratiowavenet.yaml"),
        help="Config YAML usata per addestrare RatioWaveNet se manca il checkpoint.",
    )
    parser.add_argument(
        "--tcformer-train-config",
        type=Path,
        default=Path("configs/tcformer.yaml"),
        help="Config YAML usata per addestrare TCFormer se manca il checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Cartella dove salvare i checkpoint generati durante l'addestramento automatico.",
    )
    parser.add_argument(
        "--model-device",
        type=str,
        default="cpu",
        help="Device used to execute RatioWaveNet and TCFormer (e.g. 'cpu' or 'cuda:0').",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path where the trained ensemble will be stored (.joblib).",
    )
    parser.add_argument(
        "--target-length",
        type=int,
        default=None,
        help="Temporal dimension used to harmonise the inputs. Defaults to the signal length.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of samples reserved for evaluation when using --dataset (default: 0.2).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state used for the train/validation split and the random forest.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=400,
        help="Number of trees in the random forest (default: 400).",
    )
    parser.add_argument(
        "--max-depth", type=int, default=None, help="Optional maximum depth for the trees."
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=1,
        help="Minimum number of samples per leaf node (default: 1).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.dataset is not None:
        rw_preds, tc_preds, signals, labels = _load_npz(args.dataset)

        indices = np.arange(labels.shape[0])
        train_idx, test_idx = train_test_split(
            indices,
            test_size=args.test_size,
            stratify=labels,
            random_state=args.random_state,
        )

        ensemble = RandomForestSignalEnsembler(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            target_length=args.target_length,
            random_state=args.random_state,
        )

        ensemble.fit(
            rw_preds[train_idx],
            tc_preds[train_idx],
            signals[train_idx],
            labels[train_idx],
        )

        y_true = labels[test_idx]
        y_pred = ensemble.predict(
            rw_preds[test_idx],
            tc_preds[test_idx],
            signals[test_idx],
        )
        print("Classification report:\n", classification_report(y_true, y_pred))
        print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))

        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                {
                    "model": ensemble,
                    "metadata": {
                        "dataset": str(args.dataset),
                        "target_length": ensemble.effective_length,
                        "test_size": args.test_size,
                    },
                },
                args.output,
            )
            print(f"Ensemble saved to {args.output}")
        return

    if args.config is not None:
        _run_from_models(args)
        return

    raise SystemExit("Either --dataset or --config must be specified.")


if __name__ == "__main__":
    main()