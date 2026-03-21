import argparse
from pathlib import Path

import numpy as np
from moabb.datasets import BNCI2014_001, BNCI2015_001
from moabb.paradigms import MotorImagery


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"


DATASET_CONFIGS = {
    "BNCI2014001": {
        "dataset_cls": BNCI2014_001,
        "events": ["left_hand", "right_hand"],
        "n_classes": 2,
        "fmin": 8,
        "fmax": 30,
        "resample": 250,
        "expected_trials_per_subject": 144,
        "output_dir": "BNCI2014001",
    },
    "BNCI2014001-4": {
        "dataset_cls": BNCI2014_001,
        "events": ["left_hand", "right_hand", "feet", "tongue"],
        "n_classes": 4,
        "fmin": 8,
        "fmax": 30,
        "resample": 250,
        "expected_trials_per_subject": 288,
        "output_dir": "BNCI2014001-4",
    },
    "BNCI2015001": {
        "dataset_cls": BNCI2015_001,
        "events": ["right_hand", "feet"],
        "n_classes": 2,
        "fmin": 8,
        "fmax": 30,
        "resample": 250,
        "expected_trials_per_subject": 200,
        "output_dir": "BNCI2015001",
    },
}


def load_subject_trials(config, subject_id):
    dataset = config["dataset_cls"]()
    paradigm = MotorImagery(
        events=config["events"],
        n_classes=config["n_classes"],
        fmin=config["fmin"],
        fmax=config["fmax"],
        resample=config["resample"],
    )
    X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[subject_id])
    return X.astype(np.float32), y.astype(str), metadata.reset_index(drop=True)


def select_trials(config, X, y, metadata):
    expected = config["expected_trials_per_subject"]
    if X.shape[0] == expected:
        return X, y, "all", metadata["session"].value_counts().sort_index().to_dict()

    session_counts = metadata["session"].value_counts().sort_index()
    matching_sessions = [session for session, count in session_counts.items() if count == expected]
    if not matching_sessions:
        raise ValueError(
            f"Unable to match expected trial count {expected}. "
            f"Available session counts: {session_counts.to_dict()}"
        )

    selected_session = matching_sessions[0]
    mask = metadata["session"] == selected_session
    return X[mask.to_numpy()], y[mask.to_numpy()], selected_session, session_counts.to_dict()


def build_dataset(dataset_name):
    config = DATASET_CONFIGS[dataset_name]
    subject_count = len(config["dataset_cls"]().subject_list)
    all_x = []
    all_y = []
    all_subject_ids = []
    selection_notes = []

    for subject_idx in range(subject_count):
        subject_id = subject_idx + 1
        X, y, metadata = load_subject_trials(config, subject_id)
        X_sel, y_sel, selection_name, session_counts = select_trials(config, X, y, metadata)
        all_x.append(X_sel)
        all_y.append(y_sel)
        all_subject_ids.append(np.full((X_sel.shape[0],), subject_idx, dtype=np.int64))
        selection_notes.append((subject_id, selection_name, X_sel.shape[0], session_counts))
        print(
            f"{dataset_name} subject {subject_id}: selection={selection_name}, "
            f"counts={session_counts}, X={X_sel.shape}, y={y_sel.shape}"
        )

    X_all = np.concatenate(all_x, axis=0).astype(np.float32)
    y_all = np.concatenate(all_y, axis=0)
    subject_ids = np.concatenate(all_subject_ids, axis=0).astype(np.int64)

    data_dir = DATA_ROOT / config["output_dir"]
    data_dir.mkdir(parents=True, exist_ok=True)
    np.save(data_dir / "X.npy", X_all)
    np.save(data_dir / "labels.npy", y_all)
    np.save(data_dir / "subject_ids.npy", subject_ids)
    with open(data_dir / "selected_sessions.txt", "w") as f:
        for subject_id, selection_name, trial_count, session_counts in selection_notes:
            f.write(
                f"subject_{subject_id}: {selection_name} ({trial_count} trials), "
                f"available={session_counts}\n"
            )

    print(f"saved X -> {data_dir / 'X.npy'} shape={X_all.shape}")
    print(f"saved y -> {data_dir / 'labels.npy'} shape={y_all.shape}")
    print(f"saved subject ids -> {data_dir / 'subject_ids.npy'} shape={subject_ids.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        required=True,
        choices=sorted(DATASET_CONFIGS.keys()),
        help="Downstream MI dataset to prepare",
    )
    args = parser.parse_args()
    build_dataset(args.dataset_name)
