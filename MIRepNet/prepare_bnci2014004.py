from pathlib import Path

import numpy as np
from moabb.datasets import BNCI2014_004
from moabb.paradigms import MotorImagery


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "BNCI2014004"

# The original repo expects these per-subject trial counts.
EXPECTED_TRIALS = {
    0: 160,
    1: 120,
    2: 160,
    3: 160,
    4: 160,
    5: 160,
    6: 160,
    7: 160,
    8: 160,
}

LABEL_MAP = {"left_hand": 0, "right_hand": 1}


def load_subject_trials(subject_id):
    dataset = BNCI2014_004()
    paradigm = MotorImagery(n_classes=2, fmin=8, fmax=30, resample=250)
    X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[subject_id])
    y = np.asarray([LABEL_MAP[label] for label in y], dtype=np.int64)
    return X.astype(np.float32), y, metadata.reset_index(drop=True)


def select_session(metadata, expected_trials):
    session_counts = metadata["session"].value_counts().sort_index()
    matching_sessions = [
        session for session, count in session_counts.items() if count == expected_trials
    ]
    if matching_sessions:
        return matching_sessions[0], session_counts.to_dict()

    fallback_session = session_counts.idxmax()
    return fallback_session, session_counts.to_dict()


def build_dataset():
    all_x = []
    all_y = []
    all_subject_ids = []
    selected_sessions = []

    for subject_idx in range(9):
        subject_id = subject_idx + 1
        X, y, metadata = load_subject_trials(subject_id)
        expected_trials = EXPECTED_TRIALS[subject_idx]
        selected_session, session_counts = select_session(metadata, expected_trials)
        session_mask = metadata["session"] == selected_session
        X_session = X[session_mask.to_numpy()]
        y_session = y[session_mask.to_numpy()]

        if X_session.shape[0] != expected_trials:
            raise ValueError(
                f"Subject {subject_id} selected session {selected_session} with "
                f"{X_session.shape[0]} trials, expected {expected_trials}. "
                f"Available sessions: {session_counts}"
            )

        all_x.append(X_session)
        all_y.append(y_session)
        all_subject_ids.append(np.full((expected_trials,), subject_idx, dtype=np.int64))
        selected_sessions.append((subject_id, selected_session, expected_trials))
        print(
            f"subject {subject_id}: selected {selected_session}, "
            f"counts={session_counts}, X={X_session.shape}, y={y_session.shape}"
        )

    X_all = np.concatenate(all_x, axis=0).astype(np.float32)
    y_all = np.concatenate(all_y, axis=0).astype(np.int64)
    subject_ids = np.concatenate(all_subject_ids, axis=0).astype(np.int64)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(DATA_DIR / "X.npy", X_all)
    np.save(DATA_DIR / "labels.npy", y_all)
    np.save(DATA_DIR / "subject_ids.npy", subject_ids)
    with open(DATA_DIR / "selected_sessions.txt", "w") as f:
        for subject_id, session_name, trial_count in selected_sessions:
            f.write(f"subject_{subject_id}: {session_name} ({trial_count} trials)\n")

    print(f"saved X -> {DATA_DIR / 'X.npy'} shape={X_all.shape}")
    print(f"saved y -> {DATA_DIR / 'labels.npy'} shape={y_all.shape}")
    print(f"saved subject ids -> {DATA_DIR / 'subject_ids.npy'} shape={subject_ids.shape}")


if __name__ == "__main__":
    build_dataset()
