#!/usr/bin/env python3
"""End-to-end training, validation, and evaluation entrypoint.

This script connects the existing NLB preprocessing outputs with the local
paper-aligned DMM-WcycleGAN implementation. It assumes the public NLB export
layout produced by `dataset_loader.py all`:

1. Load `gan_ready/train.npz`
2. Derive 3-class labels from trial behavior
3. Adapt spike sequences into the paper model's `(80, 8)` feature shape
4. Create deterministic train/val/test splits for supervised evaluation
5. Run meta-initialization and domain fine-tuning
6. Train the online classifier with validation-based checkpoint selection
7. Report train/val/test metrics and save predictions for unlabeled `eval.npz`

The label construction and feature adapter are engineering bridges for the
public NLB data. They are clearly separated from the paper's closed-source
signal extraction pipeline.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import logging
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


LOGGER = logging.getLogger("dmm_wcyclegan_train_eval")


@dataclass
class TrainEvalConfig:
    gan_ready_dir: Path
    output_dir: Path
    num_classes: int = 3
    feature_channels: int = 80
    feature_bins: int = 8
    label_topk_bins: int = 10
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    seed: int = 42
    meta_tasks: int = 3
    meta_epochs: int = 3
    meta_inner_steps: int = 2
    trans_epochs: int = 10
    cnn_epochs: int = 20
    batch_size: int = 16
    generator_channels: int = 32
    critic_channels: int = 32
    classifier_channels: int = 32
    device: str = "auto"


def normalize_train_eval_config(config: TrainEvalConfig | dict[str, Any]) -> TrainEvalConfig:
    if isinstance(config, TrainEvalConfig):
        return config

    payload = dict(config)
    payload["gan_ready_dir"] = Path(payload["gan_ready_dir"])
    payload["output_dir"] = Path(payload["output_dir"])
    return TrainEvalConfig(**payload)


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def load_module(module_name: str, module_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    if spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")
    spec.loader.exec_module(module)
    return module


def load_npz_dict(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as handle:
        return {key: handle[key] for key in handle.files}


def derive_direction_labels(
    behavior: np.ndarray,
    num_classes: int,
    topk_bins: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if behavior.ndim != 3 or behavior.shape[-1] < 2:
        raise ValueError(
            "Expected behavior tensor with shape (trials, time, xy), "
            f"got {behavior.shape}"
        )

    velocity = np.nan_to_num(behavior.astype(np.float32), copy=False)
    speed = np.linalg.norm(velocity, axis=-1)
    topk = min(topk_bins, velocity.shape[1])
    top_indices = np.argsort(speed, axis=1)[:, -topk:]
    representative_vectors = np.take_along_axis(
        velocity,
        top_indices[..., None],
        axis=1,
    ).mean(axis=1)
    angles = np.arctan2(representative_vectors[:, 1], representative_vectors[:, 0])
    normalized_angles = (angles + np.pi) / (2.0 * np.pi)
    labels = np.floor(normalized_angles * num_classes).astype(np.int64)
    labels = np.clip(labels, 0, num_classes - 1)

    counts = {
        str(class_index): int((labels == class_index).sum())
        for class_index in range(num_classes)
    }
    metadata = {
        "label_mode": "direction_sector",
        "num_classes": num_classes,
        "topk_bins": topk,
        "class_counts": counts,
    }
    return labels, metadata


def stratified_train_val_test_split(
    labels: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, np.ndarray]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be in (0, 1)")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0, 1)")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")

    rng = np.random.default_rng(seed)
    splits = {"train": [], "val": [], "test": []}

    for class_id in np.unique(labels):
        class_indices = np.flatnonzero(labels == class_id)
        rng.shuffle(class_indices)
        n_samples = len(class_indices)

        n_train = int(round(n_samples * train_ratio))
        n_val = int(round(n_samples * val_ratio))
        n_train = max(1, n_train)
        n_val = max(1, n_val) if n_samples >= 3 else max(0, n_val)

        if n_train + n_val >= n_samples:
            if n_train > 1:
                n_train -= 1
            elif n_val > 1:
                n_val -= 1

        n_test = n_samples - n_train - n_val
        if n_test <= 0:
            if n_train > 1:
                n_train -= 1
            else:
                n_val = max(0, n_val - 1)
            n_test = n_samples - n_train - n_val

        splits["train"].append(class_indices[:n_train])
        splits["val"].append(class_indices[n_train : n_train + n_val])
        splits["test"].append(class_indices[n_train + n_val :])

    finalized = {}
    for split_name, chunks in splits.items():
        merged = np.concatenate(chunks).astype(np.int64)
        rng.shuffle(merged)
        finalized[split_name] = merged
    return finalized


def stratified_source_target_split(
    indices: np.ndarray,
    labels: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    source_chunks = []
    target_chunks = []

    for class_id in np.unique(labels[indices]):
        class_indices = indices[labels[indices] == class_id]
        rng.shuffle(class_indices)
        midpoint = max(1, len(class_indices) // 2)
        if midpoint == len(class_indices):
            midpoint = len(class_indices) - 1
        source_chunks.append(class_indices[:midpoint])
        target_chunks.append(class_indices[midpoint:])

    source_indices = np.concatenate(source_chunks).astype(np.int64)
    target_indices = np.concatenate(target_chunks).astype(np.int64)
    rng.shuffle(source_indices)
    rng.shuffle(target_indices)
    return source_indices, target_indices


def fit_feature_adapter(
    encoder_input: np.ndarray,
    target_channels: int,
    target_bins: int,
) -> dict[str, Any]:
    activity = np.nan_to_num(encoder_input, copy=False).sum(axis=(0, 1))
    selected = np.argsort(activity)[::-1][: min(target_channels, encoder_input.shape[-1])]
    return {
        "selected_channels": selected.astype(np.int64).tolist(),
        "target_channels": target_channels,
        "target_bins": target_bins,
    }


def apply_feature_adapter(
    encoder_input: np.ndarray,
    adapter: dict[str, Any],
) -> np.ndarray:
    spikes = np.nan_to_num(encoder_input.astype(np.float32), copy=False)
    channel_indices = np.asarray(adapter["selected_channels"], dtype=np.int64)
    selected = spikes[:, :, channel_indices]

    n_trials, n_time, n_channels = selected.shape
    target_bins = int(adapter["target_bins"])
    target_channels = int(adapter["target_channels"])
    pooled = np.zeros((n_trials, target_bins, n_channels), dtype=np.float32)
    boundaries = np.linspace(0, n_time, target_bins + 1).astype(np.int64)

    for bin_index in range(target_bins):
        start = boundaries[bin_index]
        end = max(start + 1, boundaries[bin_index + 1])
        pooled[:, bin_index] = selected[:, start:end].mean(axis=1)

    feature_map = np.transpose(pooled, (0, 2, 1))
    if feature_map.shape[1] < target_channels:
        padded = np.zeros((n_trials, target_channels, target_bins), dtype=np.float32)
        padded[:, : feature_map.shape[1], :] = feature_map
        feature_map = padded

    sample_min = feature_map.min(axis=(1, 2), keepdims=True)
    sample_max = feature_map.max(axis=(1, 2), keepdims=True)
    denom = np.where(sample_max > sample_min, sample_max - sample_min, 1.0)
    return (feature_map - sample_min) / denom


def build_meta_tasks(
    dmm_module: Any,
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    num_tasks: int,
) -> list[Any]:
    if num_tasks <= 0:
        return []

    tasks = []
    paired_length = min(source_features.shape[0], target_features.shape[0])
    source_chunks = torch.chunk(source_features[:paired_length], num_tasks)
    target_chunks = torch.chunk(target_features[:paired_length], num_tasks)

    for source_chunk, target_chunk in zip(source_chunks, target_chunks):
        support_source, query_source = torch.chunk(source_chunk, 2)
        support_target, query_target = torch.chunk(target_chunk, 2)
        if min(
            support_source.shape[0],
            query_source.shape[0],
            support_target.shape[0],
            query_target.shape[0],
        ) == 0:
            continue
        tasks.append(
            dmm_module.MetaTask(
                support_source=support_source,
                support_target=support_target,
                query_source=query_source,
                query_target=query_target,
            )
        )
    return tasks


def predict_logits(
    model: Any,
    features: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    dataset = TensorDataset(torch.from_numpy(features.astype(np.float32)))
    dataloader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=False)
    logits_list = []

    model.eval()
    with torch.no_grad():
        for (batch_features,) in dataloader:
            logits = model.classify(batch_features.to(device))
            logits_list.append(logits.cpu())

    return torch.cat(logits_list, dim=0)


def confusion_matrix(labels: np.ndarray, predictions: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for label, prediction in zip(labels, predictions, strict=True):
        matrix[int(label), int(prediction)] += 1
    return matrix


def macro_f1_score(labels: np.ndarray, predictions: np.ndarray, num_classes: int) -> float:
    f1_scores = []
    for class_index in range(num_classes):
        tp = np.sum((labels == class_index) & (predictions == class_index))
        fp = np.sum((labels != class_index) & (predictions == class_index))
        fn = np.sum((labels == class_index) & (predictions != class_index))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall == 0.0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2.0 * precision * recall / (precision + recall))
    return float(np.mean(f1_scores))


def evaluate_split(
    model: Any,
    features: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    batch_size: int,
    num_classes: int,
) -> dict[str, Any]:
    logits = predict_logits(model, features, device, batch_size)
    probabilities = torch.softmax(logits, dim=1).numpy()
    predictions = probabilities.argmax(axis=1)

    accuracy = float((predictions == labels).mean())
    macro_f1 = macro_f1_score(labels, predictions, num_classes)
    matrix = confusion_matrix(labels, predictions, num_classes)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "confusion_matrix": matrix.tolist(),
        "predictions": predictions.tolist(),
        "probabilities": probabilities.tolist(),
    }


def train_online_classifier_with_validation(
    model: Any,
    trainer: Any,
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    source_labels: torch.Tensor,
    target_labels: torch.Tensor,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    device: torch.device,
    batch_size: int,
    epochs: int,
    num_classes: int,
) -> tuple[list[dict[str, float]], dict[str, Any], dict[str, Any]]:
    augmented_features, augmented_labels = model.build_augmented_dataset(
        source_features.to(device),
        target_features.to(device),
        source_labels.to(device),
        target_labels.to(device),
    )

    dataloader = DataLoader(
        TensorDataset(augmented_features.detach().cpu(), augmented_labels.detach().cpu()),
        batch_size=min(batch_size, len(augmented_labels)),
        shuffle=True,
    )

    history = []
    best_epoch = 0
    best_metrics: dict[str, Any] | None = None
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        for batch_features, batch_labels in dataloader:
            loss, _ = model.classifier_loss(
                batch_features.to(device),
                batch_labels.to(device),
            )
            trainer.classifier_optimizer.zero_grad()
            loss.backward()
            trainer.classifier_optimizer.step()
            epoch_losses.append(float(loss.item()))

        val_metrics = evaluate_split(
            model,
            val_features,
            val_labels,
            device=device,
            batch_size=batch_size,
            num_classes=num_classes,
        )
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": float(np.mean(epoch_losses)),
                "val_accuracy": float(val_metrics["accuracy"]),
                "val_macro_f1": float(val_metrics["macro_f1"]),
            }
        )

        if best_metrics is None or val_metrics["accuracy"] >= best_metrics["accuracy"]:
            best_epoch = epoch + 1
            best_metrics = val_metrics
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    assert best_metrics is not None
    selection = {"best_epoch": best_epoch, "best_val_accuracy": best_metrics["accuracy"]}
    return history, best_metrics, selection


def save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def maybe_import_plotting(base_dir: Path) -> tuple[Any, Any] | tuple[None, None]:
    mpl_config_dir = ensure_dir(base_dir / ".mplconfig")
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception as exc:
        LOGGER.warning("Plotting libraries unavailable, skipping figure export: %s", exc)
        return None, None
    return plt, sns


def save_visualization_artifacts(
    output_dir: Path,
    label_metadata: dict[str, Any],
    histories: dict[str, list[dict[str, float]]],
    metrics: dict[str, dict[str, Any]],
    eval_artifact: dict[str, Any] | None,
) -> list[str]:
    plt, sns = maybe_import_plotting(output_dir)
    if plt is None or sns is None:
        return []

    plot_dir = ensure_dir(output_dir / "plots")
    saved_plots: list[str] = []

    counts = label_metadata.get("class_counts", {})
    if counts:
        fig, ax = plt.subplots(figsize=(6, 4))
        classes = list(counts.keys())
        values = [counts[key] for key in classes]
        sns.barplot(x=classes, y=values, ax=ax, hue=classes, palette="crest", legend=False)
        ax.set_title("Training Label Distribution")
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        fig.tight_layout()
        path = plot_dir / "label_distribution.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        saved_plots.append(str(path))

    classifier_history = histories.get("classifier", [])
    fine_tune_history = histories.get("fine_tune", [])

    if classifier_history:
        fig, ax1 = plt.subplots(figsize=(7, 4))
        epochs = [entry["epoch"] for entry in classifier_history]
        train_loss = [entry["train_loss"] for entry in classifier_history]
        val_accuracy = [entry["val_accuracy"] for entry in classifier_history]
        val_macro_f1 = [entry["val_macro_f1"] for entry in classifier_history]
        ax1.plot(epochs, train_loss, label="train_loss", color="#d55e00")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Train Loss", color="#d55e00")
        ax1.tick_params(axis="y", labelcolor="#d55e00")
        ax2 = ax1.twinx()
        ax2.plot(epochs, val_accuracy, label="val_accuracy", color="#0072b2")
        ax2.plot(epochs, val_macro_f1, label="val_macro_f1", color="#009e73")
        ax2.set_ylabel("Validation Metrics")
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="center right")
        ax1.set_title("Classifier Training Curve")
        fig.tight_layout()
        path = plot_dir / "classifier_history.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        saved_plots.append(str(path))

    if fine_tune_history:
        fig, ax = plt.subplots(figsize=(7, 4))
        epochs = [entry["epoch"] for entry in fine_tune_history]
        generator_loss = [entry["generator_loss"] for entry in fine_tune_history]
        critic_loss = [entry["critic_loss"] for entry in fine_tune_history]
        ax.plot(epochs, generator_loss, label="generator_loss", color="#cc79a7")
        ax.plot(epochs, critic_loss, label="critic_loss", color="#56b4e9")
        ax.set_title("Domain Adaptation Fine-Tuning Curve")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        fig.tight_layout()
        path = plot_dir / "fine_tune_history.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        saved_plots.append(str(path))

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for axis, split_name in zip(axes, ["train", "val", "test"], strict=True):
        matrix = np.asarray(metrics[split_name]["confusion_matrix"], dtype=np.int64)
        sns.heatmap(matrix, annot=True, fmt="d", cmap="YlGnBu", cbar=False, ax=axis)
        axis.set_title(f"{split_name.title()} Confusion")
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
    fig.tight_layout()
    path = plot_dir / "confusion_matrices.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved_plots.append(str(path))

    if eval_artifact is not None:
        predicted_counts = eval_artifact.get("predicted_class_counts", {})
        if predicted_counts:
            fig, ax = plt.subplots(figsize=(6, 4))
            classes = list(predicted_counts.keys())
            values = [predicted_counts[key] for key in classes]
            sns.barplot(x=classes, y=values, ax=ax, hue=classes, palette="mako", legend=False)
            ax.set_title("Eval Prediction Distribution")
            ax.set_xlabel("Predicted Class")
            ax.set_ylabel("Count")
            fig.tight_layout()
            path = plot_dir / "eval_prediction_distribution.png"
            fig.savefig(path, dpi=180)
            plt.close(fig)
            saved_plots.append(str(path))

    return saved_plots


def run_training(
    config: TrainEvalConfig | dict[str, Any],
    *,
    verbose: bool | None = None,
) -> dict[str, Any]:
    config = normalize_train_eval_config(config)
    if verbose is not None:
        configure_logging(verbose)

    set_seed(config.seed)
    output_dir = ensure_dir(config.output_dir)

    script_dir = Path(__file__).resolve().parent
    dmm_module = load_module("dmm_wcyclegan_runtime", script_dir / "DMM-WcycleGAN.py")
    device = resolve_device(config.device)

    train_npz = load_npz_dict(config.gan_ready_dir / "train.npz")
    eval_npz_path = config.gan_ready_dir / "eval.npz"
    eval_npz = load_npz_dict(eval_npz_path) if eval_npz_path.exists() else None

    if "extra__behavior" not in train_npz:
        raise KeyError(
            "Training labels require `extra__behavior` in train.npz. "
            "Re-export the dataset with behavior enabled."
        )

    labels, label_metadata = derive_direction_labels(
        train_npz["extra__behavior"],
        num_classes=config.num_classes,
        topk_bins=config.label_topk_bins,
    )

    adapter = fit_feature_adapter(
        train_npz["encoder_input"],
        target_channels=config.feature_channels,
        target_bins=config.feature_bins,
    )
    features = apply_feature_adapter(train_npz["encoder_input"], adapter)
    split_indices = stratified_train_val_test_split(
        labels,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        seed=config.seed,
    )
    source_indices, target_indices = stratified_source_target_split(
        split_indices["train"],
        labels,
        seed=config.seed + 1,
    )

    model_config = dmm_module.DMMWcycleGANConfig(
        feature_shape=(config.feature_channels, config.feature_bins),
        num_classes=config.num_classes,
        generator_channels=config.generator_channels,
        critic_channels=config.critic_channels,
        classifier_channels=config.classifier_channels,
        max_meta_epochs=config.meta_epochs,
        max_inner_steps=config.meta_inner_steps,
        max_trans_epochs=config.trans_epochs,
        max_cnn_epochs=config.cnn_epochs,
        batch_size=config.batch_size,
    )

    model = dmm_module.DMMWcycleGAN(model_config).to(device)
    trainer = dmm_module.DMMWcycleGANTrainer(model, model_config)

    source_features = torch.from_numpy(features[source_indices]).float()
    target_features = torch.from_numpy(features[target_indices]).float()
    source_labels = torch.from_numpy(labels[source_indices]).long()
    target_labels = torch.from_numpy(labels[target_indices]).long()

    paired_length = min(source_features.shape[0], target_features.shape[0])
    source_features = source_features[:paired_length]
    target_features = target_features[:paired_length]
    source_labels = source_labels[:paired_length]
    target_labels = target_labels[:paired_length]

    meta_tasks = build_meta_tasks(
        dmm_module,
        source_features,
        target_features,
        config.meta_tasks,
    )
    meta_history = trainer.meta_initialize(meta_tasks)

    # Paper Section II-C-2 / On-line Transfer Learning Part:
    # insert timestamps around the online fine-tuning stage only.
    fine_tune_started_at_utc = utc_timestamp()
    fine_tune_started_at_perf = time.perf_counter()
    fine_tune_history = trainer.fine_tune(source_features, target_features)
    fine_tune_finished_at_perf = time.perf_counter()
    fine_tune_finished_at_utc = utc_timestamp()

    fine_tune_elapsed_seconds = fine_tune_finished_at_perf - fine_tune_started_at_perf
    fine_tune_timing = {
        "paper_stage": "Model Fine-Tuning",
        "pipeline_stage": "online_fine_tuning",
        "start_time_utc": fine_tune_started_at_utc,
        "end_time_utc": fine_tune_finished_at_utc,
        "elapsed_seconds": float(fine_tune_elapsed_seconds),
        "elapsed_minutes": float(fine_tune_elapsed_seconds / 60.0),
        "epochs": int(config.trans_epochs),
        "seconds_per_epoch": float(
            fine_tune_elapsed_seconds / max(1, len(fine_tune_history))
        ),
    }
    LOGGER.info(
        "Online fine-tuning took %.4f seconds across %d epochs",
        fine_tune_timing["elapsed_seconds"],
        fine_tune_timing["epochs"],
    )

    classifier_history, best_val_metrics, checkpoint_selection = train_online_classifier_with_validation(
        model=model,
        trainer=trainer,
        source_features=source_features,
        target_features=target_features,
        source_labels=source_labels,
        target_labels=target_labels,
        val_features=features[split_indices["val"]],
        val_labels=labels[split_indices["val"]],
        device=device,
        batch_size=config.batch_size,
        epochs=config.cnn_epochs,
        num_classes=config.num_classes,
    )

    metrics = {
        "train": evaluate_split(
            model,
            features[split_indices["train"]],
            labels[split_indices["train"]],
            device=device,
            batch_size=config.batch_size,
            num_classes=config.num_classes,
        ),
        "val": best_val_metrics,
        "test": evaluate_split(
            model,
            features[split_indices["test"]],
            labels[split_indices["test"]],
            device=device,
            batch_size=config.batch_size,
            num_classes=config.num_classes,
        ),
    }

    eval_artifact = None
    if eval_npz is not None and "encoder_input" in eval_npz:
        eval_features = apply_feature_adapter(eval_npz["encoder_input"], adapter)
        eval_logits = predict_logits(model, eval_features, device=device, batch_size=config.batch_size)
        eval_probabilities = torch.softmax(eval_logits, dim=1).numpy()
        eval_predictions = eval_probabilities.argmax(axis=1)
        eval_artifact = {
            "num_samples": int(eval_predictions.shape[0]),
            "predicted_class_counts": {
                str(class_index): int((eval_predictions == class_index).sum())
                for class_index in range(config.num_classes)
            },
        }
        np.savez_compressed(
            output_dir / "eval_predictions.npz",
            predictions=eval_predictions,
            probabilities=eval_probabilities,
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model_config),
            "adapter": adapter,
            "label_metadata": label_metadata,
        },
        output_dir / "checkpoint.pt",
    )

    np.savez_compressed(
        output_dir / "split_indices.npz",
        train=split_indices["train"],
        val=split_indices["val"],
        test=split_indices["test"],
        source=source_indices,
        target=target_indices,
        labels=labels,
    )

    np.savez_compressed(
        output_dir / "test_predictions.npz",
        labels=labels[split_indices["test"]],
        predictions=np.asarray(metrics["test"]["predictions"], dtype=np.int64),
        probabilities=np.asarray(metrics["test"]["probabilities"], dtype=np.float32),
    )

    histories = {
        "meta": meta_history,
        "fine_tune": fine_tune_history,
        "classifier": classifier_history,
    }
    history_summary = {
        "meta_epochs_ran": len(meta_history),
        "fine_tune_epochs_ran": len(fine_tune_history),
        "classifier_epochs_ran": len(classifier_history),
    }

    plot_files = save_visualization_artifacts(
        output_dir=output_dir,
        label_metadata=label_metadata,
        histories=histories,
        metrics=metrics,
        eval_artifact=eval_artifact,
    )

    summary = {
        "config": {
            **asdict(config),
            "gan_ready_dir": str(config.gan_ready_dir),
            "output_dir": str(config.output_dir),
            "device_resolved": str(device),
        },
        "label_metadata": label_metadata,
        "adapter": adapter,
        "history_summary": history_summary,
        "timing": {
            "online_fine_tuning": fine_tune_timing,
        },
        "checkpoint_selection": checkpoint_selection,
        "metrics": {
            split_name: {
                key: value
                for key, value in split_metrics.items()
                if key not in {"predictions", "probabilities"}
            }
            for split_name, split_metrics in metrics.items()
        },
        "eval_artifact": eval_artifact,
        "plot_files": plot_files,
        "notes": [
            "Labels are derived from NLB behavior using coarse direction sectors.",
            "Feature maps are adapted from spike sequences into the model's (80, 8) shape.",
            "Official eval.npz is inference-only because public labels are unavailable in the export.",
        ],
    }

    save_json(output_dir / "histories.json", histories)
    save_json(
        output_dir / "timing.json",
        {
            "online_fine_tuning": fine_tune_timing,
        },
    )
    save_json(output_dir / "summary.json", summary)

    LOGGER.info("Train accuracy: %.4f", metrics["train"]["accuracy"])
    LOGGER.info("Val accuracy: %.4f", metrics["val"]["accuracy"])
    LOGGER.info("Test accuracy: %.4f", metrics["test"]["accuracy"])
    LOGGER.info("Artifacts saved to %s", output_dir)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train, validate, and evaluate the paper-aligned DMM-WcycleGAN pipeline.",
    )
    parser.add_argument("--gan-ready-dir", type=Path, required=True, help="Directory containing train.npz and eval.npz.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for checkpoints and metrics.")
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--feature-channels", type=int, default=80)
    parser.add_argument("--feature-bins", type=int, default=8)
    parser.add_argument("--label-topk-bins", type=int, default=10)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--meta-tasks", type=int, default=3)
    parser.add_argument("--meta-epochs", type=int, default=3)
    parser.add_argument("--meta-inner-steps", type=int, default=2)
    parser.add_argument("--trans-epochs", type=int, default=10)
    parser.add_argument("--cnn-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--generator-channels", type=int, default=32)
    parser.add_argument("--critic-channels", type=int, default=32)
    parser.add_argument("--classifier-channels", type=int, default=32)
    parser.add_argument("--device", default="auto", help="`auto`, `cpu`, or `cuda`.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = TrainEvalConfig(
        gan_ready_dir=args.gan_ready_dir,
        output_dir=args.output_dir,
        num_classes=args.num_classes,
        feature_channels=args.feature_channels,
        feature_bins=args.feature_bins,
        label_topk_bins=args.label_topk_bins,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        meta_tasks=args.meta_tasks,
        meta_epochs=args.meta_epochs,
        meta_inner_steps=args.meta_inner_steps,
        trans_epochs=args.trans_epochs,
        cnn_epochs=args.cnn_epochs,
        batch_size=args.batch_size,
        generator_channels=args.generator_channels,
        critic_channels=args.critic_channels,
        classifier_channels=args.classifier_channels,
        device=args.device,
    )
    run_training(config, verbose=args.verbose)


if __name__ == "__main__":
    main()
