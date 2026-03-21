import argparse
import copy
from datetime import datetime
from pathlib import Path

import pandas as pd

from utils.utils import DATASET_CONFIGS, SUPPORTED_MODELS, run_experiment


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASETS = ["BNCI2014001", "BNCI2015001", "BNCI2014004", "BNCI2014001-4"]
DEFAULT_MODELS = [
    "MIRepNet",
    "ShallowConv",
    "DeepConv",
    "EEGNet",
    "IFNet",
    "ADFCNN",
    "Conformer",
    "FBCNet",
    "EDPNet",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Batch benchmark for MIRepNet and specialist baselines")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS, help="Datasets to evaluate")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Models to evaluate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay")
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam", help="Optimizer to use")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD")
    parser.add_argument("--scheduler", choices=["cosine", "step", "none"], default="cosine", help="LR scheduler")
    parser.add_argument("--step_size", type=int, default=30, help="Step size for StepLR")
    parser.add_argument("--gamma", type=float, default=0.1, help="Gamma for StepLR")
    parser.add_argument("--emb_size", type=int, default=256, help="Embedding size for MIRepNet")
    parser.add_argument("--depth", type=int, default=6, help="Transformer depth for MIRepNet")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--val_split", type=float, default=0.7, help="Validation split ratio")
    parser.add_argument("--num_exp", type=int, default=1, help="Number of repeated seeds")
    parser.add_argument(
        "--pretrain_path",
        default=str(PROJECT_ROOT / "weight" / "MIRepNet.pth"),
        help="Path to pretrained MIRepNet weights",
    )
    return parser.parse_args()


def _validate_selection(values, supported, label):
    unsupported = sorted(set(values) - set(supported))
    if unsupported:
        raise ValueError(f"Unsupported {label}: {', '.join(unsupported)}")


def main():
    args = parse_args()
    _validate_selection(args.datasets, DATASET_CONFIGS.keys(), "datasets")
    _validate_selection(args.models, SUPPORTED_MODELS, "models")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    benchmark_dir = PROJECT_ROOT / "result" / "benchmark" / timestamp
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    status_rows = []

    for dataset_name in args.datasets:
        dataset_dir = PROJECT_ROOT / "data" / dataset_name
        if not dataset_dir.exists():
            status_rows.append(
                {
                    "dataset": dataset_name,
                    "model": "",
                    "status": "missing_dataset",
                    "error": f"Missing dataset directory: {dataset_dir}",
                }
            )
            continue

        for model_name in args.models:
            run_args = copy.deepcopy(args)
            run_args.dataset_name = dataset_name
            run_args.model_name = model_name

            log_dir = benchmark_dir / f"{dataset_name}_{model_name}"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / "driver.log"

            with open(log_path, "w") as log_file:
                try:
                    summary = run_experiment(run_args, log_file)
                    summary_rows.append(summary)
                    status_rows.append(
                        {
                            "dataset": dataset_name,
                            "model": model_name,
                            "status": "success",
                            "error": "",
                            "subject_metrics_csv": summary["subject_metrics_csv"],
                            "seed_summary_csv": summary["seed_summary_csv"],
                            "legacy_acc_csv": summary["legacy_acc_csv"],
                        }
                    )
                except Exception as exc:
                    status_rows.append(
                        {
                            "dataset": dataset_name,
                            "model": model_name,
                            "status": "failed",
                            "error": str(exc),
                        }
                    )

    status_df = pd.DataFrame(status_rows)
    status_df.to_csv(benchmark_dir / "benchmark_status.csv", index=False)

    if not summary_rows:
        raise RuntimeError("No benchmark run finished successfully.")

    summary_df = pd.DataFrame(summary_rows)
    summary_df["acc_display"] = summary_df.apply(
        lambda row: f"{row['mean_val_acc']:.2f}±{row['std_val_acc']:.2f}",
        axis=1,
    )
    summary_df.to_csv(benchmark_dir / "benchmark_summary.csv", index=False)

    accuracy_table = (
        summary_df.pivot(index="model", columns="dataset", values="acc_display")
        .reindex(DEFAULT_MODELS)
    )
    accuracy_table.to_csv(benchmark_dir / "final_accuracy_table.csv")

    metric_table = summary_df[
        [
            "model",
            "dataset",
            "mean_val_acc",
            "std_val_acc",
            "mean_val_balanced_acc",
            "mean_val_macro_f1",
            "mean_val_kappa",
            "subject_metrics_csv",
            "seed_summary_csv",
        ]
    ].sort_values(["model", "dataset"])
    metric_table.to_csv(benchmark_dir / "final_metric_table.csv", index=False)

    print(f"Benchmark finished. Results saved under {benchmark_dir}")


if __name__ == "__main__":
    main()
