import numpy as np
import torch
from pathlib import Path
from sklearn import preprocessing
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parent


def _dataset_dir(dataset_name):
    direct_dir = PROJECT_ROOT / 'data' / dataset_name
    if direct_dir.exists():
        return direct_dir
    if dataset_name == 'BNCI2014001-4':
        return PROJECT_ROOT / 'data' / 'BNCI2014001'
    return direct_dir

class EEGDataset(Dataset):
    def __init__(self, args=None):
        self.dataset_name = args.dataset_name
        self.args = args

        data_dir = _dataset_dir(self.dataset_name)
        x_path = data_dir / 'X.npy'
        y_path = data_dir / 'labels.npy'
        subject_ids_path = data_dir / 'subject_ids.npy'

        if not x_path.exists() or not y_path.exists():
            raise FileNotFoundError(
                f"Dataset files not found for '{self.dataset_name}'. "
                f"Expected:\n- {x_path}\n- {y_path}\n"
                "Please place the processed .npy files under MIRepNet/data/<dataset_name>/."
            )

        X = np.load(x_path)
        y = np.load(y_path)
        print("original data shape:", X.shape, "labels shape:", y.shape)

        if subject_ids_path.exists():
            subject_ids = np.load(subject_ids_path)
            if len(subject_ids) != len(y):
                raise ValueError(
                    f"subject_ids length mismatch for {self.dataset_name}: "
                    f"{len(subject_ids)} vs {len(y)} labels."
                )
            if hasattr(self.args, 'sub') and self.args.sub is not None:
                subject_mask = np.isin(subject_ids, np.asarray(self.args.sub))
                X = X[subject_mask]
                y = y[subject_mask]

        if self.dataset_name == 'BNCI2014004':
            self.paradigm = 'MI'
            self.num_subjects = len(self.args.sub)
            self.sample_rate = 250

            if not subject_ids_path.exists():
                # Backward-compatible path for the original sparse indexing layout.
                subject_indices = {
                    0: np.arange(160) + 400,
                    1: np.arange(120) + 1120,
                    2: np.arange(160) + 1800,
                    3: np.arange(160) + 2540,
                    4: np.arange(160) + 3280,
                    5: np.arange(160) + 4000,
                    6: np.arange(160) + 4720,
                    7: np.arange(160) + 5480,
                    8: np.arange(160) + 6200
                }

                indices = []
                for subject_id in self.args.sub:
                    indices.append(subject_indices[subject_id])

                indices = np.concatenate(indices, axis=0)
                X = X[indices]
                y = y[indices]

            X = X[:, :, :1000]
        elif self.dataset_name in {'BNCI2014001', 'BNCI2014001-4'}:
            self.paradigm = 'MI'
            self.num_subjects = len(self.args.sub)
            self.sample_rate = 250
            X = X[:, :, :1000]
        else:
            self.paradigm = None
            self.num_subjects = None
            self.sample_rate = None
            self.ch_num = None

        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y)
        print("preprocessed data shape:", X.shape, "preprocessed labels shape:", y.shape)

        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        sample = self.X[idx]
        label = self.y[idx]
        return torch.tensor(sample, dtype=torch.float32), torch.tensor(label, dtype=torch.long)  # Ensure label is of type long
