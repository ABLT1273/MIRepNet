import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASETS = ["BNCI2014001", "BNCI2015001", "BNCI2014004", "AlexMI", "BNCI2014001-4"]
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
    parser = argparse.ArgumentParser(description="Aggregate latest benchmark CSV outputs into final tables")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS, help="Datasets to include")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Models to include")
    return parser.parse_args()


def _latest_file(directory, pattern):
    matches = sorted(directory.glob(pattern))
    return matches[-1] if matches else None


def main():
    args = parse_args()
    metrics_dir = PROJECT_ROOT / "result" / "metrics"
    acc_dir = PROJECT_ROOT / "result" / "acc"

    rows = []
    status_rows = []
    for dataset in args.datasets:
        for model in args.models:
            subject_metrics_csv = _latest_file(metrics_dir, f"{dataset}_{model}_*_subject_metrics.csv")
            seed_summary_csv = _latest_file(metrics_dir, f"{dataset}_{model}_*_seed_summary.csv")
            legacy_acc_csv = _latest_file(acc_dir, f"{dataset}_{model}_*_results.csv")

            if subject_metrics_csv is None or seed_summary_csv is None or legacy_acc_csv is None:
                status_rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "status": "missing",
                        "subject_metrics_csv": subject_metrics_csv,
                        "seed_summary_csv": seed_summary_csv,
                        "legacy_acc_csv": legacy_acc_csv,
                    }
                )
                continue

            subject_df = pd.read_csv(subject_metrics_csv)
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "subject_count": len(subject_df),
                    "mean_val_acc": subject_df["val_acc"].mean(),
                    "std_val_acc": subject_df["val_acc"].std(ddof=0),
                    "mean_val_balanced_acc": subject_df["val_balanced_acc"].mean(),
                    "mean_val_macro_f1": subject_df["val_macro_f1"].mean(),
                    "mean_val_kappa": subject_df["val_kappa"].mean(),
                    "subject_metrics_csv": str(subject_metrics_csv),
                    "seed_summary_csv": str(seed_summary_csv),
                    "legacy_acc_csv": str(legacy_acc_csv),
                }
            )
            status_rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "status": "ok",
                    "subject_metrics_csv": str(subject_metrics_csv),
                    "seed_summary_csv": str(seed_summary_csv),
                    "legacy_acc_csv": str(legacy_acc_csv),
                }
            )

    output_dir = PROJECT_ROOT / "result" / "final_tables" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    status_df = pd.DataFrame(status_rows).sort_values(["dataset", "model"])
    status_df.to_csv(output_dir / "collection_status.csv", index=False)

    if not rows:
        raise RuntimeError("No completed results found to aggregate.")

    summary_df = pd.DataFrame(rows).sort_values(["model", "dataset"])
    summary_df["acc_display"] = summary_df.apply(
        lambda row: f"{row['mean_val_acc']:.2f}±{row['std_val_acc']:.2f}",
        axis=1,
    )
    summary_df.to_csv(output_dir / "benchmark_long_summary.csv", index=False)

    accuracy_table = summary_df.pivot(index="model", columns="dataset", values="acc_display")
    accuracy_table = accuracy_table.reindex(args.models)
    accuracy_table = accuracy_table.reindex(columns=args.datasets)
    accuracy_table.to_csv(output_dir / "final_accuracy_table.csv")

    metric_table = summary_df[
        [
            "model",
            "dataset",
            "subject_count",
            "mean_val_acc",
            "std_val_acc",
            "mean_val_balanced_acc",
            "mean_val_macro_f1",
            "mean_val_kappa",
            "subject_metrics_csv",
            "seed_summary_csv",
            "legacy_acc_csv",
        ]
    ]
    metric_table.to_csv(output_dir / "final_metric_table.csv", index=False)

    print(f"Aggregated results saved to {output_dir}")


if __name__ == "__main__":
    main()
