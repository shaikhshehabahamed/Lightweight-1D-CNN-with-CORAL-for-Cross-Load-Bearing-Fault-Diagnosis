# main_coral.py
import argparse
import os
import random
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.nn import CrossEntropyLoss

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report

try:
    import pandas as pd
except Exception:
    pd = None

from helper import CWRUDataset
from nn_model import CNN_1D_3L, CNN_1D_2L
from train_helper import fit_coral


# --------------------
# Utilities
# --------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_model(n_in: int, which: str):
    if which == "2L":
        return CNN_1D_2L(n_in=n_in)
    return CNN_1D_3L(n_in=n_in)


@torch.no_grad()
def eval_collect(
    model: torch.nn.Module,
    dl: DataLoader,
    loss_func: torch.nn.Module,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
   
    model.eval()
    total_loss, total_samples = 0.0, 0
    all_true, all_pred = [], []
    for xb, yb in dl:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        logits = model(xb)
        loss = loss_func(logits, yb)

        bs = yb.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

        preds = logits.argmax(dim=1)
        all_true.append(yb.cpu().numpy())
        all_pred.append(preds.cpu().numpy())

    y_true = np.concatenate(all_true, axis=0) if all_true else np.array([])
    y_pred = np.concatenate(all_pred, axis=0) if all_pred else np.array([])
    acc = float((y_true == y_pred).mean()) if y_true.size else 0.0
    avg_loss = total_loss / max(1, total_samples)
    return avg_loss, acc, y_true, y_pred


# --------------------
# CLI
# --------------------
def parse_args():
    parser = argparse.ArgumentParser(description="CNN1D + CORAL on CWRU (12k_DE)")

    # Data / domain
    parser.add_argument(
        "--data",
        type=str,
        default=r"C:\Users\sheha\OneDrive\Documents\Lightweight-1D-CNN-with-CORAL-for-Cross-Load-Bearing-Fault-Diagnosis\Data",
        help="Root folder containing the .mat files (subfolders allowed).",
    )
    parser.add_argument("--source_load", type=int, default=0, help="Source domain load id (e.g., 0)")
    parser.add_argument("--target_load", type=int, default=1, help="Target domain load id (e.g., 1)")

    # Windowing / dataset options
    parser.add_argument("--segment_length", type=int, default=2048, help="Window length (samples)")
    parser.add_argument("--per_window_norm", action="store_true",
                        help="Apply z-score per window (in addition to dataset-level normalize=True)")
    parser.add_argument("--src_size", type=str, default=None,
                        help="Filter source by fault size: 007 | 014 | 021 | None")
    parser.add_argument("--tgt_size", type=str, default=None,
                        help="Filter target by fault size: 007 | 014 | 021 | None")

    # Training
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--lambda_coral", type=float, default=0.5, help="Weight for CORAL loss")
    parser.add_argument("--model", choices=["2L", "3L"], default="3L", help="Backbone depth")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (0 is safe on Windows)")
    parser.add_argument("--val_frac", type=float, default=0.15, help="Fraction of source data used for validation")
    parser.add_argument("--early_stop_patience", type=int, default=10, help="Early stopping patience (epochs)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Output
    parser.add_argument("--out_dir", type=str, default=".",
                        help="Directory to write checkpoints and reports")

    return parser.parse_args()


# --------------------
# Main
# --------------------
def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_mem = torch.cuda.is_available()

    print(f"[INFO] Using device: {device} | CUDA available: {torch.cuda.is_available()}")
    print(f"[INFO] Data root: {args.data}")
    print(f"[INFO] Source load: {args.source_load} -> Target load: {args.target_load}")
    if args.src_size or args.tgt_size:
        print(f"[INFO] Size filters | src: {args.src_size}  tgt: {args.tgt_size}")
    print(f"[INFO] per_window_norm: {args.per_window_norm}")

    # 50% overlap
    overlap = args.segment_length // 2

    # --------------------
    # Datasets
    # --------------------
    src_ds = CWRUDataset(
        args.data,
        segment_length=args.segment_length,
        normalize=True,
        load_id=args.source_load,
        overlap=overlap,
        size_filter=args.src_size,
        per_window_norm=args.per_window_norm,
    )
    tgt_ds = CWRUDataset(
        args.data,
        segment_length=args.segment_length,
        normalize=True,
        load_id=args.target_load,
        overlap=overlap,
        size_filter=args.tgt_size,
        per_window_norm=args.per_window_norm,
    )

    # --------------------
    # Grouped Source train/val split (by filename to avoid leakage)
    # --------------------
    val_frac = max(0.0, min(0.5, args.val_frac))  # clip to [0, 0.5]
    gss_s = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=args.seed)
    s_train_idx, s_val_idx = next(gss_s.split(src_ds.X, src_ds.y, groups=src_ds.filenames))
    src_train_ds = Subset(src_ds, s_train_idx)
    src_val_ds = Subset(src_ds, s_val_idx)

    print(
        "[INFO] Source samples: {} (train {} / val {}) | Target samples: {}".format(
            len(src_ds), len(src_train_ds), len(src_val_ds), len(tgt_ds)
        )
    )
    print("[INFO] Grouped split by filename applied to SOURCE (prevents window-level leakage).")

    # --------------------
    # Grouped Target adapt/test split (hold out target TEST)
    # --------------------
    gss_t = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=args.seed)
    t_adapt_idx, t_test_idx = next(gss_t.split(tgt_ds.X, tgt_ds.y, groups=tgt_ds.filenames))
    tgt_adapt_ds = Subset(tgt_ds, t_adapt_idx)   # unlabeled target for CORAL
    tgt_test_ds = Subset(tgt_ds, t_test_idx)     # held-out target test, only used after training

    print(
        "[INFO] Target split: adapt {} | test {} (grouped by filename)".format(
            len(tgt_adapt_ds), len(tgt_test_ds)
        )
    )

    # --------------------
    # Dataloaders
    # --------------------
    src_dl = DataLoader(
        src_train_ds, batch_size=args.batch_size, shuffle=True,
        drop_last=True, num_workers=args.num_workers, pin_memory=pin_mem
    )
    src_val_dl = DataLoader(
        src_val_ds, batch_size=args.batch_size, shuffle=False,
        drop_last=False, num_workers=args.num_workers, pin_memory=pin_mem
    )
    tgt_dl = DataLoader(
        tgt_adapt_ds, batch_size=args.batch_size, shuffle=True,
        drop_last=True, num_workers=args.num_workers, pin_memory=pin_mem
    )
    tgt_eval_dl = DataLoader(
        tgt_test_ds, batch_size=args.batch_size, shuffle=False,
        drop_last=False, num_workers=args.num_workers, pin_memory=pin_mem
    )

    # --------------------
    # Model / Optimizer / Loss
    # --------------------
    model = make_model(args.segment_length, args.model).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Class-weighted CE on SOURCE TRAIN distribution
    cls_counts = np.bincount(np.array(src_ds.y)[s_train_idx], minlength=4)
    cls_weights = (cls_counts.sum() / (np.maximum(1, cls_counts) * 4.0)).astype("float32")
    ce = CrossEntropyLoss(weight=torch.tensor(cls_weights, device=device))

    print(f"[INFO] Source class counts (train): {cls_counts.tolist()}")
    print(f"[INFO] CE class weights: {cls_weights.tolist()}")

    # --------------------
    # Train with CORAL (+ early stopping on source-val)
    # --------------------
    fit_coral(
        epochs=args.epochs,
        model=model,
        opt=opt,
        src_dl=src_dl,
        tgt_dl=tgt_dl,
        lambda_coral=args.lambda_coral,
        loss_func=ce,
        src_val_dl=src_val_dl,
        early_stop_patience=args.early_stop_patience,
    )

    # --------------------
    # Evaluate on held-out TARGET TEST + artifacts
    # --------------------
    print("\n[INFO] Evaluating on held-out TARGET TEST:")
    test_loss, test_acc, y_true, y_pred = eval_collect(model, tgt_eval_dl, ce, device)
    print(f"[RESULT] Target loss: {test_loss:.4f} | Target acc: {test_acc:.4f}")

    # Confusion matrix + classification report
    os.makedirs(args.out_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    labels_short = ["N", "B", "IR", "OR"]

    if pd is not None:
        cm_df = pd.DataFrame(cm, index=labels_short, columns=labels_short)
        cm_path = os.path.join(
            args.out_dir,
            f"cm_target_s{args.source_load}_t{args.target_load}_seg{args.segment_length}_{args.model}.csv",
        )
        cm_df.to_csv(cm_path, index=True)
        print(f"[INFO] Saved confusion matrix: {os.path.abspath(cm_path)}")
    else:
        print("[WARN] pandas not available; skipping CSV export of confusion matrix.")

    clf_rep = classification_report(y_true, y_pred, target_names=labels_short, digits=4)
    rep_path = os.path.join(
        args.out_dir,
        f"classification_report_s{args.source_load}_t{args.target_load}_seg{args.segment_length}_{args.model}.txt",
    )
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write(clf_rep + "\n")
    print("\n" + clf_rep)
    print(f"[INFO] Saved classification report: {os.path.abspath(rep_path)}")

    # --------------------
    # Save checkpoint
    # --------------------
    ckpt_name = f"model_coral_s{args.source_load}_t{args.target_load}_seg{args.segment_length}_{args.model}.pt"
    ckpt_path = os.path.join(args.out_dir, ckpt_name)
    torch.save(model.state_dict(), ckpt_path)
    print(f"[INFO] Saved checkpoint: {os.path.abspath(ckpt_path)}")


if __name__ == "__main__":

    main()
