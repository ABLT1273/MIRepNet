#!/usr/bin/env python3
"""NLB data import and preprocessing pipeline for GAN experiments.

This script focuses on the public Neural Latents Benchmark (NLB) pipeline and
supports two stages:

1. Raw NWB -> official NLB H5 tensors (via `nlb_tools`)
2. Official NLB H5 -> GAN-friendly numpy archives

The exported archives keep the sequence structure required by spike-sequence
models, while also exposing held-in / held-out / forward-prediction targets.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


LOGGER = logging.getLogger("nlb_gan_pipeline")

NLB_DATASET_ALIASES = {
    "maze": "mc_maze",
    "mcmaze": "mc_maze",
    "mc_maze": "mc_maze",
    "mc_maze_small": "mc_maze_small",
    "mc_maze_medium": "mc_maze_medium",
    "mc_maze_large": "mc_maze_large",
    "rtt": "mc_rtt",
    "mcrtt": "mc_rtt",
    "mc_rtt": "mc_rtt",
    "area2": "area2_bump",
    "area2_bump": "area2_bump",
    "dmfc": "dmfc_rsg",
    "dmfc_rsg": "dmfc_rsg",
}

KNOWN_SPLIT_PREFIXES = ("train", "eval", "test")


@dataclass
class PreprocessConfig:
    dataset_name: str
    nwb_path: Path
    output_dir: Path
    bin_width_ms: int = 5
    include_behavior: bool = True
    include_forward_pred: bool = True
    eval_splits: tuple[str, ...] = ("val",)
    build_full_h5: bool = True


@dataclass
class ExportConfig:
    input_h5: Path
    output_dir: Path
    flatten_time: bool = False


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def normalize_dataset_name(dataset_name: str) -> str:
    key = dataset_name.strip().lower()
    return NLB_DATASET_ALIASES.get(key, key)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_nwb_source(path_like: Path) -> Path:
    path = path_like.expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() != ".nwb":
            raise ValueError(f"`{path}` is a file but not an `.nwb` file.")
        return path

    if not path.exists():
        raise FileNotFoundError(f"NWB path does not exist: {path}")

    candidates = sorted(path.rglob("*.nwb"))
    if not candidates:
        raise FileNotFoundError(
            f"No `.nwb` file found under `{path}`. Please point to a dataset "
            "directory or a specific NWB file."
        )
    if len(candidates) > 1:
        preferred_candidates = [
            candidate
            for candidate in candidates
            if "desc-train" in candidate.name or "train_behavior+ecephys" in candidate.name
        ]
        if len(preferred_candidates) == 1:
            LOGGER.info(
                "Multiple NWB files detected under `%s`; defaulting to training file `%s`.",
                path,
                preferred_candidates[0].name,
            )
            return preferred_candidates[0]

        candidate_text = "\n".join(str(candidate) for candidate in candidates[:10])
        raise ValueError(
            "Multiple NWB files were found. Please pass a single NWB file path or "
            "use a directory that contains a unique training NWB.\n"
            f"{candidate_text}"
        )
    return candidates[0]


def import_nlb_tools() -> tuple[Any, Any, Any, Any]:
    cache_root = ensure_dir(Path.cwd() / ".cache")
    os.environ["HOME"] = str(Path.cwd())
    os.environ["XDG_CACHE_HOME"] = str(cache_root)

    try:
        from nlb_tools.make_tensors import (
            combine_h5,
            make_eval_input_tensors,
            make_train_input_tensors,
        )
        from nlb_tools.nwb_interface import NWBDataset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing official NLB dependency. Install `nlb-tools` and `pynwb` "
            "in the project environment before running raw NWB preprocessing."
        ) from exc
    return NWBDataset, make_train_input_tensors, make_eval_input_tensors, combine_h5


def resample_dataset_compat(dataset: Any, target_bin_ms: int) -> None:
    """Compatibility resampler for newer pandas versions.

    `nlb_tools` sets a fixed `DatetimeIndex.freq` after resampling. With the
    versions in this environment, the inferred frequency may be `None` even
    when the resampled index values are correct, which raises a ValueError.
    This mirrors the official implementation but keeps the resampled index
    values without requiring a strict `.freq`.
    """
    import pandas as pd
    from scipy import signal

    if target_bin_ms == dataset.bin_width:
        LOGGER.warning(
            "Dataset already at %s ms resolution, skipping resampling...",
            target_bin_ms,
        )
        return

    if target_bin_ms % dataset.bin_width != 0:
        raise ValueError("target_bin must be an integer multiple of bin_width.")

    resample_factor = int(round(target_bin_ms / dataset.bin_width))
    cols = dataset.data.columns
    data_list = []

    for signal_type in cols.get_level_values(0).unique():
        if "spikes" in signal_type:
            arr = dataset.data[signal_type].to_numpy()
            dtype = dataset.data[signal_type].dtypes.iloc[0]
            nan_mask = np.isnan(arr[::resample_factor])
            if arr.shape[0] % resample_factor != 0:
                extra = arr[-(arr.shape[0] % resample_factor) :]
                arr = arr[: -(arr.shape[0] % resample_factor)]
            else:
                extra = None

            arr = (
                np.nan_to_num(arr, copy=False)
                .reshape((arr.shape[0] // resample_factor, resample_factor, -1))
                .sum(axis=1)
            )
            if extra is not None:
                arr = np.vstack([arr, np.nan_to_num(extra, copy=False).sum(axis=0)])

            arr[nan_mask] = np.nan
            resampled = pd.DataFrame(
                arr,
                index=dataset.data.index[::resample_factor],
                dtype=dtype,
            )
        elif signal_type == "target_pos":
            resampled = dataset.data[signal_type].iloc[::resample_factor]
        else:
            dtype = dataset.data[signal_type].dtypes.iloc[0]
            nan_mask = dataset.data[signal_type].iloc[::resample_factor].isna()
            if np.any(dataset.data[signal_type].isna()):
                dataset.data[signal_type] = dataset.data[signal_type].apply(
                    lambda x: x.interpolate(limit_direction="both")
                )
            decimated = signal.decimate(
                dataset.data[signal_type],
                resample_factor,
                axis=0,
                n=500,
                ftype="fir",
            )
            decimated[nan_mask] = np.nan
            resampled = pd.DataFrame(
                decimated,
                index=dataset.data.index[::resample_factor],
                dtype=dtype,
            )

        resampled.columns = pd.MultiIndex.from_product(
            [[signal_type], dataset.data[signal_type].columns],
            names=("signal_type", "channel"),
        )
        data_list.append(resampled)

    dataset.data = pd.concat(data_list, axis=1)
    try:
        dataset.data.index.freq = f"{target_bin_ms}ms"
    except ValueError:
        if isinstance(dataset.data.index, pd.TimedeltaIndex):
            dataset.data.index = pd.TimedeltaIndex(dataset.data.index.values)
        else:
            dataset.data.index = pd.DatetimeIndex(dataset.data.index.values)
    dataset.bin_width = target_bin_ms


def preprocess_nlb_from_nwb(config: PreprocessConfig) -> list[Path]:
    dataset_name = normalize_dataset_name(config.dataset_name)
    nwb_source = resolve_nwb_source(config.nwb_path)
    output_dir = ensure_dir(config.output_dir)

    LOGGER.info("Using NLB dataset `%s` from `%s`", dataset_name, nwb_source)

    (
        NWBDataset,
        make_train_input_tensors,
        make_eval_input_tensors,
        combine_h5,
    ) = import_nlb_tools()

    dataset = NWBDataset(str(nwb_source))
    if not dataset.data.index.is_monotonic_increasing:
        LOGGER.warning(
            "NWB time index is not monotonic; sorting rows by timestamp for compatibility."
        )
        dataset.data = dataset.data.sort_index()
    resample_dataset_compat(dataset, config.bin_width_ms)
    if not dataset.data.index.is_monotonic_increasing:
        dataset.data = dataset.data.sort_index()

    generated_files: list[Path] = []

    train_h5 = output_dir / f"{dataset_name}_train.h5"
    LOGGER.info("Writing official train tensors -> %s", train_h5)
    make_train_input_tensors(
        dataset,
        dataset_name=dataset_name,
        trial_split="train",
        include_behavior=config.include_behavior,
        include_forward_pred=config.include_forward_pred,
        save_file=True,
        save_path=str(train_h5),
    )
    generated_files.append(train_h5)

    for split in config.eval_splits:
        eval_h5 = output_dir / f"{dataset_name}_{split}.h5"
        LOGGER.info("Writing official %s tensors -> %s", split, eval_h5)
        make_eval_input_tensors(
            dataset,
            dataset_name=dataset_name,
            trial_split=split,
            save_file=True,
            save_path=str(eval_h5),
        )
        generated_files.append(eval_h5)

    if config.build_full_h5 and len(generated_files) > 1:
        full_h5 = output_dir / f"{dataset_name}_full.h5"
        LOGGER.info("Combining split tensors -> %s", full_h5)
        combine_h5(
            [str(path) for path in generated_files],
            save_path=str(full_h5),
        )
        generated_files.append(full_h5)

    return generated_files


def load_h5_dict(h5_path: Path) -> dict[str, np.ndarray]:
    with h5py.File(h5_path, "r") as handle:
        return {key: handle[key][()] for key in handle.keys()}


def infer_available_prefixes(h5_dict: dict[str, np.ndarray]) -> list[str]:
    prefixes = []
    for prefix in KNOWN_SPLIT_PREFIXES:
        probe = f"{prefix}_spikes_heldin"
        if probe in h5_dict:
            prefixes.append(prefix)
    if not prefixes:
        keys = ", ".join(sorted(h5_dict.keys()))
        raise ValueError(
            "Could not find any official NLB split prefix in the H5 file. "
            f"Available keys: {keys}"
        )
    return prefixes


def zeros_like_split(
    n_trials: int,
    n_time: int,
    n_units: int,
    dtype: np.dtype[np.generic],
) -> np.ndarray:
    return np.zeros((n_trials, n_time, n_units), dtype=dtype)


def extract_split_arrays(
    h5_dict: dict[str, np.ndarray],
    prefix: str,
) -> dict[str, np.ndarray]:
    heldin_key = f"{prefix}_spikes_heldin"
    heldin = h5_dict[heldin_key].astype(np.float32)

    n_trials, n_time, heldin_units = heldin.shape
    dtype = heldin.dtype

    heldout_key = f"{prefix}_spikes_heldout"
    heldout = h5_dict.get(
        heldout_key,
        zeros_like_split(n_trials, n_time, 0, dtype),
    ).astype(np.float32)

    heldin_forward_key = f"{prefix}_spikes_heldin_forward"
    if heldin_forward_key in h5_dict:
        heldin_forward = h5_dict[heldin_forward_key].astype(np.float32)
        forward_time = heldin_forward.shape[1]
    else:
        heldin_forward = zeros_like_split(n_trials, 0, heldin_units, dtype).astype(
            np.float32
        )
        forward_time = 0

    heldout_units = heldout.shape[-1]
    heldout_forward_key = f"{prefix}_spikes_heldout_forward"
    if heldout_forward_key in h5_dict:
        heldout_forward = h5_dict[heldout_forward_key].astype(np.float32)
    else:
        heldout_forward = zeros_like_split(
            n_trials,
            forward_time,
            heldout_units,
            dtype,
        ).astype(np.float32)

    full_target = np.concatenate([heldin, heldout], axis=-1)
    if heldin_forward.shape[1] > 0 or heldout_forward.shape[1] > 0:
        forward_target = np.concatenate([heldin_forward, heldout_forward], axis=-1)
    else:
        forward_target = zeros_like_split(n_trials, 0, full_target.shape[-1], dtype)

    extras = {
        key[len(prefix) + 1 :]: value
        for key, value in h5_dict.items()
        if key.startswith(f"{prefix}_") and not key.startswith(f"{prefix}_spikes_")
    }

    export = {
        "encoder_input": heldin,
        "heldout_target": heldout,
        "full_target": full_target,
        "forward_target": forward_target.astype(np.float32),
        "heldout_mask": np.ones_like(heldout, dtype=np.float32),
        "forward_mask": np.ones_like(forward_target, dtype=np.float32),
    }

    if export["heldout_target"].shape[-1] == 0:
        export["heldout_mask"] = np.zeros_like(export["heldout_target"], dtype=np.float32)
    if export["forward_target"].shape[1] == 0:
        export["forward_mask"] = np.zeros_like(export["forward_target"], dtype=np.float32)

    for extra_name, extra_value in extras.items():
        export[f"extra__{extra_name}"] = extra_value

    return export


def save_split_npz(
    split_name: str,
    split_data: dict[str, np.ndarray],
    output_dir: Path,
    flatten_time: bool,
) -> dict[str, Any]:
    serializable = dict(split_data)
    if flatten_time:
        serializable["encoder_input_flat"] = serializable["encoder_input"].reshape(
            serializable["encoder_input"].shape[0],
            -1,
        )
        serializable["full_target_flat"] = serializable["full_target"].reshape(
            serializable["full_target"].shape[0],
            -1,
        )

    save_path = output_dir / f"{split_name}.npz"
    np.savez_compressed(save_path, **serializable)

    return {
        "file": str(save_path),
        "arrays": {
            key: list(value.shape) for key, value in serializable.items() if hasattr(value, "shape")
        },
    }


def export_h5_to_gan_ready(config: ExportConfig) -> Path:
    input_h5 = config.input_h5.expanduser().resolve()
    if not input_h5.exists():
        raise FileNotFoundError(f"H5 file does not exist: {input_h5}")

    output_dir = ensure_dir(config.output_dir)
    h5_dict = load_h5_dict(input_h5)
    prefixes = infer_available_prefixes(h5_dict)

    LOGGER.info("Exporting GAN-ready tensors from `%s`", input_h5)
    LOGGER.info("Detected split prefixes: %s", ", ".join(prefixes))

    summary: dict[str, Any] = {
        "input_h5": str(input_h5),
        "flatten_time": config.flatten_time,
        "splits": {},
        "original_keys": sorted(h5_dict.keys()),
    }

    for prefix in prefixes:
        split_data = extract_split_arrays(h5_dict, prefix)
        summary["splits"][prefix] = save_split_npz(
            prefix,
            split_data,
            output_dir,
            config.flatten_time,
        )

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    LOGGER.info("GAN-ready export complete -> %s", output_dir)
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NLB import + preprocessing pipeline for GAN experiments.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess_parser = subparsers.add_parser(
        "preprocess",
        help="Convert raw NLB NWB data into official NLB H5 tensors.",
    )
    preprocess_parser.add_argument("--dataset-name", required=True, help="NLB dataset name.")
    preprocess_parser.add_argument(
        "--nwb-path",
        required=True,
        type=Path,
        help="Path to a specific `.nwb` file or a directory containing one.",
    )
    preprocess_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("GAN/processed/nlb"),
        help="Directory for generated official H5 files.",
    )
    preprocess_parser.add_argument(
        "--bin-width-ms",
        type=int,
        default=5,
        help="Resample bin width in milliseconds. Official NLB examples often use 5 ms.",
    )
    preprocess_parser.add_argument(
        "--no-behavior",
        action="store_true",
        help="Do not include behavior tensors in the exported train split.",
    )
    preprocess_parser.add_argument(
        "--no-forward-pred",
        action="store_true",
        help="Do not include forward-prediction tensors in the exported train split.",
    )
    preprocess_parser.add_argument(
        "--eval-splits",
        nargs="+",
        default=["val"],
        help="Evaluation trial splits to export, e.g. `val` or `val test`.",
    )
    preprocess_parser.add_argument(
        "--skip-full-h5",
        action="store_true",
        help="Skip creation of the combined `*_full.h5` file.",
    )

    export_parser = subparsers.add_parser(
        "export",
        help="Convert official NLB H5 tensors into GAN-ready numpy archives.",
    )
    export_parser.add_argument(
        "--input-h5",
        required=True,
        type=Path,
        help="Path to an official NLB H5 tensor file.",
    )
    export_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("GAN/processed/nlb_gan_ready"),
        help="Directory for exported GAN-ready `.npz` files.",
    )
    export_parser.add_argument(
        "--flatten-time",
        action="store_true",
        help="Also save flattened per-trial views for MLP-style GAN variants.",
    )

    all_parser = subparsers.add_parser(
        "all",
        help="Run `preprocess` and then `export` in one command.",
    )
    all_parser.add_argument("--dataset-name", required=True, help="NLB dataset name.")
    all_parser.add_argument(
        "--nwb-path",
        required=True,
        type=Path,
        help="Path to a specific `.nwb` file or a directory containing one.",
    )
    all_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("GAN/processed/nlb"),
        help="Base directory for official H5 files and GAN-ready exports.",
    )
    all_parser.add_argument(
        "--bin-width-ms",
        type=int,
        default=5,
        help="Resample bin width in milliseconds.",
    )
    all_parser.add_argument(
        "--no-behavior",
        action="store_true",
        help="Do not include behavior tensors in the exported train split.",
    )
    all_parser.add_argument(
        "--no-forward-pred",
        action="store_true",
        help="Do not include forward-prediction tensors in the exported train split.",
    )
    all_parser.add_argument(
        "--eval-splits",
        nargs="+",
        default=["val"],
        help="Evaluation trial splits to export, e.g. `val` or `val test`.",
    )
    all_parser.add_argument(
        "--flatten-time",
        action="store_true",
        help="Also save flattened per-trial views for MLP-style GAN variants.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    if args.command == "preprocess":
        config = PreprocessConfig(
            dataset_name=args.dataset_name,
            nwb_path=args.nwb_path,
            output_dir=args.output_dir,
            bin_width_ms=args.bin_width_ms,
            include_behavior=not args.no_behavior,
            include_forward_pred=not args.no_forward_pred,
            eval_splits=tuple(args.eval_splits),
            build_full_h5=not args.skip_full_h5,
        )
        files = preprocess_nlb_from_nwb(config)
        LOGGER.info("Generated files:\n%s", "\n".join(str(path) for path in files))
        return

    if args.command == "export":
        summary_path = export_h5_to_gan_ready(
            ExportConfig(
                input_h5=args.input_h5,
                output_dir=args.output_dir,
                flatten_time=args.flatten_time,
            )
        )
        LOGGER.info("Summary saved to `%s`", summary_path)
        return

    if args.command == "all":
        base_dir = ensure_dir(args.output_dir)
        official_dir = ensure_dir(base_dir / "official_h5")
        gan_dir = ensure_dir(base_dir / "gan_ready")

        preprocess_config = PreprocessConfig(
            dataset_name=args.dataset_name,
            nwb_path=args.nwb_path,
            output_dir=official_dir,
            bin_width_ms=args.bin_width_ms,
            include_behavior=not args.no_behavior,
            include_forward_pred=not args.no_forward_pred,
            eval_splits=tuple(args.eval_splits),
            build_full_h5=True,
        )
        generated_files = preprocess_nlb_from_nwb(preprocess_config)

        dataset_name = normalize_dataset_name(args.dataset_name)
        full_h5 = official_dir / f"{dataset_name}_full.h5"
        export_input = full_h5 if full_h5 in generated_files else generated_files[-1]
        summary_path = export_h5_to_gan_ready(
            ExportConfig(
                input_h5=export_input,
                output_dir=gan_dir,
                flatten_time=args.flatten_time,
            )
        )
        LOGGER.info("Summary saved to `%s`", summary_path)
        return

    raise RuntimeError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
