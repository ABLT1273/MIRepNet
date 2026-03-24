import copy
import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.linalg import fractional_matrix_power
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, TensorDataset

from model.ADFCNN import Net as ADFCNNNet
from model.Conformer import Conformer
from model.Deep_Shallow_Conv import DeepConvNet, ShallowConvNet
from model.EDPNet import EDPNet
from model.EEGNet import EEGNet
from model.FBCNet import FBCNet
from model.IFNet import IFNet
from model.mlm import mlm_mask
from utils.channel_list import *


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_CONFIGS = {
    "BNCI2014001": {"subjects": 9, "num_classes": 2, "sampling_rate": 250},
    "BNCI2015001": {"subjects": 12, "num_classes": 2, "sampling_rate": 512},
    "BNCI2014004": {"subjects": 9, "num_classes": 2, "sampling_rate": 250},
    "BNCI2014001-4": {"subjects": 9, "num_classes": 4, "sampling_rate": 250},
    "AlexMI": {"subjects": 8, "num_classes": 3, "sampling_rate": 512},
}

SUPPORTED_MODELS = {
    "MIRepNet",
    "ShallowConv",
    "DeepConv",
    "EEGNet",
    "IFNet",
    "ADFCNN",
    "Conformer",
    "FBCNet",
    "EDPNet",
}

MODELS_WITH_4D_INPUT = {"EEGNet", "DeepConv", "ShallowConv", "ADFCNN", "FBCNet"}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_seeds(args):
    if hasattr(args, "seeds") and args.seeds:
        return [int(seed) for seed in args.seeds]
    return [seed_offset + 666 for seed_offset in range(args.num_exp)]


def pad_missing_channels_diff(x, target_channels, actual_channels):
    bsz, channels, time_points = x.shape
    num_target = len(target_channels)

    existing_pos = np.array([channel_positions[ch] for ch in actual_channels])
    target_pos = np.array([channel_positions[ch] for ch in target_channels])

    weights = np.zeros((num_target, channels))
    for idx, (target_ch, pos) in enumerate(zip(target_channels, target_pos)):
        if target_ch in actual_channels:
            src_idx = actual_channels.index(target_ch)
            weights[idx, src_idx] = 1.0
            continue

        dist = cdist([pos], existing_pos)[0]
        channel_weights = 1 / (dist + 1e-6)
        channel_weights /= channel_weights.sum()
        weights[idx] = channel_weights

    padded = np.zeros((bsz, num_target, time_points))
    for batch_idx in range(bsz):
        padded[batch_idx] = weights @ x[batch_idx]

    return padded


def EA(x):
    cov = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
    for idx in range(x.shape[0]):
        cov[idx] = np.cov(x[idx])
    ref_ea = np.mean(cov, axis=0)
    sqrt_ref_ea = fractional_matrix_power(ref_ea, -0.5)
    x_ea = np.zeros_like(x)
    for idx in range(x.shape[0]):
        x_ea[idx] = np.dot(sqrt_ref_ea, x[idx])
    return x_ea


def _get_dataset_channel_names(dataset_name):
    if dataset_name == "BNCI2014001":
        return BNCI2014001_chn_names
    if dataset_name == "BNCI2014004":
        return BNCI2014004_chn_names
    if dataset_name == "BNCI2014001-4":
        return BNCI2014001_chn_names
    if dataset_name == "AlexMI":
        return AlexMI_chn_names
    if dataset_name == "BNCI2015001":
        return BNCI2015001_chn_names
    raise ValueError(f"Unsupported dataset for channel mapping: {dataset_name}")


def process_and_replace_loader(loader, ischangechn, dataset):
    all_data = []
    all_labels = []
    for idx in range(len(loader.dataset)):
        data, label = loader.dataset[idx]
        all_data.append(data.numpy())
        all_labels.append(label)

    data_np = np.stack(all_data, axis=0)
    labels_tensor = torch.stack(all_labels)

    processed_data = EA(data_np).astype(np.float32)

    if ischangechn:
        channels_names = _get_dataset_channel_names(dataset)
        processed_data = pad_missing_channels_diff(
            processed_data,
            use_channels_names,
            channels_names,
        )

    new_dataset = TensorDataset(
        torch.from_numpy(processed_data).float(),
        labels_tensor,
    )

    loader_args = {
        "batch_size": loader.batch_size,
        "num_workers": loader.num_workers,
        "pin_memory": loader.pin_memory,
        "drop_last": loader.drop_last,
        "shuffle": isinstance(loader.sampler, torch.utils.data.RandomSampler),
    }

    return DataLoader(new_dataset, **loader_args)


def _needs_channel_alignment(model_name):
    return model_name == "MIRepNet"


def _prepare_model_input(data, model_name):
    if model_name in MODELS_WITH_4D_INPUT:
        return data.unsqueeze(1)
    return data


def _compute_metrics(y_true, y_pred):
    metrics = {
        "val_acc": accuracy_score(y_true, y_pred) * 100.0,
        "val_balanced_acc": balanced_accuracy_score(y_true, y_pred) * 100.0,
        "val_macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0) * 100.0,
    }
    try:
        metrics["val_kappa"] = cohen_kappa_score(y_true, y_pred) * 100.0
    except ValueError:
        metrics["val_kappa"] = np.nan
    return metrics


def _patch_size_for_ifnet(input_samples):
    for patch_size in (125, 100, 50, 25):
        if input_samples % patch_size == 0:
            return patch_size
    return 1


def build_model(args, input_channels, input_samples):
    model_name = args.model_name
    num_classes = args.num_classes

    if model_name == "MIRepNet":
        return mlm_mask(
            emb_size=args.emb_size,
            depth=args.depth,
            n_classes=num_classes,
            pretrainmode=False,
            pretrain=args.pretrain_path,
        )
    if model_name == "ShallowConv":
        model = ShallowConvNet(n_classes=num_classes, Chans=input_channels, Samples=input_samples)
        model.classifier_block = nn.Sequential(nn.LazyLinear(num_classes))
        return model
    if model_name == "DeepConv":
        model = DeepConvNet(n_classes=num_classes, Chans=input_channels, Samples=input_samples)
        model.classifier_block = nn.Sequential(nn.LazyLinear(num_classes))
        return model
    if model_name == "EEGNet":
        return EEGNet(
            n_classes=num_classes,
            Chans=input_channels,
            Samples=input_samples,
            kernLenght=min(125, input_samples),
            F1=8,
            D=2,
            F2=16,
            dropoutRate=0.5,
            norm_rate=0.25,
        )
    if model_name == "IFNet":
        patch_size = _patch_size_for_ifnet(input_samples)
        return IFNet(
            in_planes=input_channels,
            out_planes=64,
            kernel_size=63,
            radix=1,
            patch_size=patch_size,
            time_points=input_samples,
            num_classes=num_classes,
        )
    if model_name == "ADFCNN":
        model = ADFCNNNet(
            num_classes=num_classes,
            num_channels=input_channels,
            sampling_rate=args.sampling_rate,
        )
        model.classifier = nn.Sequential(nn.Flatten(), nn.LazyLinear(num_classes))
        return model
    if model_name == "Conformer":
        return Conformer(n_classes=num_classes, num_cha=input_channels)
    if model_name == "FBCNet":
        return FBCNet(nChan=input_channels, nTime=input_samples, nClass=num_classes, nBands=1)
    if model_name == "EDPNet":
        return EDPNet(chans=input_channels, samples=input_samples, num_classes=num_classes)

    raise ValueError(f"Unsupported model: {model_name}")


def forward_model(model, data, model_name):
    prepared = _prepare_model_input(data, model_name)

    if model_name == "MIRepNet":
        _, logits = model(prepared)
        return logits
    if model_name == "FBCNet":
        features = model.scb(prepared)
        features = features.reshape(
            [*features.shape[:2], model.strideFactor, int(features.shape[-1] / model.strideFactor)]
        )
        features = model.temporalLayer(features)
        features = torch.flatten(features, start_dim=1)
        return model.lastLayer[0](features)

    return model(prepared)


def train(model, model_name, train_loader, criterion, optimizer, device, scheduler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for data, labels in train_loader:
        data = data.to(device)
        labels = labels.to(device)

        outputs = forward_model(model, data, model_name)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    if scheduler is not None:
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
    else:
        current_lr = optimizer.param_groups[0]["lr"]

    epoch_loss = running_loss / max(len(train_loader), 1)
    accuracy = correct / max(total, 1) * 100.0
    return epoch_loss, accuracy, current_lr


def validate(model, model_name, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for data, labels in val_loader:
            data = data.to(device)
            labels = labels.to(device)

            outputs = forward_model(model, data, model_name)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    epoch_loss = running_loss / max(len(val_loader), 1)
    metrics = _compute_metrics(np.asarray(all_labels), np.asarray(all_predictions))
    return epoch_loss, metrics


def run_experiment(args, log_file):
    if args.dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")
    if args.model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {args.model_name}")

    dataset_cfg = DATASET_CONFIGS[args.dataset_name]
    args.sub_num = dataset_cfg["subjects"]
    args.num_classes = dataset_cfg["num_classes"]
    args.sampling_rate = dataset_cfg["sampling_rate"]

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_root = os.path.join(PROJECT_ROOT, "result")
    log_filename = os.path.join(result_root, "log", f"{args.dataset_name}_{args.model_name}_{now}_log.txt")
    acc_filename = os.path.join(result_root, "acc", f"{args.dataset_name}_{args.model_name}_{now}_results.csv")
    metrics_dir = os.path.join(result_root, "metrics")
    subject_metrics_filename = os.path.join(
        metrics_dir, f"{args.dataset_name}_{args.model_name}_{now}_subject_metrics.csv"
    )
    seed_summary_filename = os.path.join(
        metrics_dir, f"{args.dataset_name}_{args.model_name}_{now}_seed_summary.csv"
    )

    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    file_handler = open(log_filename, "w")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    legacy_results = []
    subject_records = []
    seed_summaries = []

    seed_list = resolve_seeds(args)
    for seed in seed_list:
        set_seed(seed)
        subject_results = {}
        seed_records = []

        for subject in range(args.sub_num):
            log_message = f"Starting Dataset={args.dataset_name}, Model={args.model_name}, Subject={subject}, Seed={seed}\n"
            log_file.write(log_message)
            file_handler.write(log_message)

            metrics = train_subject(args, subject, seed, device, log_file)
            subject_results[subject] = metrics["val_acc"]
            seed_records.append(metrics)
            subject_records.append(metrics)

        legacy_results.append([seed] + [subject_results[idx] for idx in range(args.sub_num)])
        seed_summaries.append(_summarize_seed(seed_records, args.dataset_name, args.model_name, seed))

    save_results(legacy_results, args.sub_num, acc_filename)
    subject_df = pd.DataFrame(subject_records)
    seed_summary_df = pd.DataFrame(seed_summaries)
    subject_df.to_csv(subject_metrics_filename, index=False)
    seed_summary_df.to_csv(seed_summary_filename, index=False)

    final_summary = {
        "dataset": args.dataset_name,
        "model": args.model_name,
        "subject_count": args.sub_num,
        "seed_count": len(seed_list),
        "mean_val_acc": subject_df["val_acc"].mean(),
        "std_val_acc": subject_df["val_acc"].std(ddof=0),
        "mean_val_balanced_acc": subject_df["val_balanced_acc"].mean(),
        "mean_val_macro_f1": subject_df["val_macro_f1"].mean(),
        "mean_val_kappa": subject_df["val_kappa"].mean(),
        "seeds": ",".join(str(seed) for seed in seed_list),
        "legacy_acc_csv": acc_filename,
        "subject_metrics_csv": subject_metrics_filename,
        "seed_summary_csv": seed_summary_filename,
        "log_path": log_filename,
    }

    file_handler.close()
    return final_summary


def _summarize_seed(seed_records, dataset_name, model_name, seed):
    seed_df = pd.DataFrame(seed_records)
    return {
        "dataset": dataset_name,
        "model": model_name,
        "seed": seed,
        "subject_count": len(seed_df),
        "mean_train_loss": seed_df["train_loss"].mean(),
        "mean_train_acc": seed_df["train_acc"].mean(),
        "mean_val_loss": seed_df["val_loss"].mean(),
        "mean_val_acc": seed_df["val_acc"].mean(),
        "std_val_acc": seed_df["val_acc"].std(ddof=0),
        "mean_val_balanced_acc": seed_df["val_balanced_acc"].mean(),
        "mean_val_macro_f1": seed_df["val_macro_f1"].mean(),
        "mean_val_kappa": seed_df["val_kappa"].mean(),
        "mean_train_samples": seed_df["train_samples"].mean(),
        "mean_val_samples": seed_df["val_samples"].mean(),
    }


def train_subject(args, subject, seed, device, log_file):
    from dataset import EEGDataset

    args = copy.deepcopy(args)
    args.sub = [subject]
    dataset = EEGDataset(args=args)

    indices = np.arange(len(dataset))
    stratify_labels = dataset.y if len(np.unique(dataset.y)) > 1 else None
    train_idx, val_idx = train_test_split(
        indices,
        test_size=args.val_split,
        random_state=seed,
        stratify=stratify_labels,
    )

    train_data = Subset(dataset, train_idx)
    val_data = Subset(dataset, val_idx)
    effective_num_workers = 0 if device.type == "cpu" else args.num_workers

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=effective_num_workers,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=effective_num_workers,
    )

    train_loader = process_and_replace_loader(
        train_loader,
        ischangechn=_needs_channel_alignment(args.model_name),
        dataset=args.dataset_name,
    )
    val_loader = process_and_replace_loader(
        val_loader,
        ischangechn=_needs_channel_alignment(args.model_name),
        dataset=args.dataset_name,
    )

    sample_tensor = train_loader.dataset.tensors[0]
    input_channels = sample_tensor.shape[1]
    input_samples = sample_tensor.shape[2]

    model = build_model(args, input_channels, input_samples).to(device)
    criterion = nn.CrossEntropyLoss()

    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    if args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.scheduler == "none":
        scheduler = None
    else:
        raise ValueError(f"Unsupported scheduler: {args.scheduler}")

    final_metrics = None
    for epoch in range(args.epochs):
        train_loss, train_acc, curr_lr = train(
            model,
            args.model_name,
            train_loader,
            criterion,
            optimizer,
            device,
            scheduler,
        )
        val_loss, val_metrics = validate(model, args.model_name, val_loader, criterion, device)

        final_metrics = {
            "dataset": args.dataset_name,
            "model": args.model_name,
            "seed": seed,
            "subject": subject,
            "epoch": epoch + 1,
            "train_samples": len(train_idx),
            "val_samples": len(val_idx),
            "input_channels": input_channels,
            "input_samples": input_samples,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            **val_metrics,
            "lr": curr_lr,
        }

        log_file.write(
            f"Seed={seed}, Subject={subject}, Epoch={epoch + 1}, "
            f"TrainLoss={train_loss:.4f}, TrainAcc={train_acc:.2f}, "
            f"ValLoss={val_loss:.4f}, ValAcc={val_metrics['val_acc']:.2f}, "
            f"ValBalancedAcc={val_metrics['val_balanced_acc']:.2f}, "
            f"ValMacroF1={val_metrics['val_macro_f1']:.2f}, "
            f"ValKappa={val_metrics['val_kappa']:.2f}, LR={curr_lr:.6f}\n"
        )

    return final_metrics


def save_results(results, subject_count, filename):
    columns = ["Seed"] + [f"Subject_{idx}_Acc" for idx in range(subject_count)]
    results_df = pd.DataFrame(results, columns=columns)

    results_df["Seed_Avg_Acc"] = results_df.iloc[:, 1:].mean(axis=1)
    subject_avg = results_df.iloc[:, 1:-1].mean(axis=0)
    seed_avg = results_df["Seed_Avg_Acc"].mean()

    summary_row = ["Average"] + subject_avg.tolist() + [seed_avg]
    results_df.loc[len(results_df)] = summary_row

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    results_df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")
