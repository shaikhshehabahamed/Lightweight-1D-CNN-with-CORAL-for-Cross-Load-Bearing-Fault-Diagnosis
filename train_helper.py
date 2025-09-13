# train_helper.py
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss

from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

# AMP imports (enable larger batches / lower memory on CUDA)
from torch.cuda.amp import autocast, GradScaler


# ------------------------------------------------------------------------
# Supervised training helpers
# ------------------------------------------------------------------------
def get_dataloader(train_ds, valid_ds, bs):
  
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )


def loss_batch(model, loss_func, xb, yb, opt=None):
   
    out = model(xb)
    loss = loss_func(out, yb)
    pred = torch.argmax(out, dim=1).cpu().numpy()

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb), pred


def fit(epochs, model, loss_func, opt, train_dl, valid_dl, train_metric=False):
    
    print('EPOCH', '\t', 'Train Loss', '\t','Val Loss', '\t', 'Train Acc', '\t','Val Acc', '\t')

    metrics_dic = {'train_loss': [], 'train_accuracy': [], 'val_loss': [], 'val_accuracy': []}
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_accuracy = 0.0
        num_examples = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            loss, batch_size, pred = loss_batch(model, loss_func, xb, yb, opt)
            if not train_metric:
                train_loss += loss
                num_examples += batch_size

        # Validate
        model.eval()
        with torch.no_grad():
            val_loss, val_accuracy, _ = validate(model, valid_dl, loss_func)
            if train_metric:
                train_loss, train_accuracy, _ = validate(model, train_dl, loss_func)
            else:
                train_loss = train_loss / max(1, num_examples)

        metrics_dic['val_loss'].append(val_loss)
        metrics_dic['val_accuracy'].append(val_accuracy)
        metrics_dic['train_loss'].append(train_loss)
        metrics_dic['train_accuracy'].append(train_accuracy)

        print(f'{epoch} \t{train_loss:.05f}\t{val_loss:.05f}\t{train_accuracy:.05f}\t{val_accuracy:.05f}\t')

    metrics = pd.DataFrame.from_dict(metrics_dic)
    return model, metrics


def validate(model, dl, loss_func):
 
    total_loss, total_size = 0.0, 0
    predictions, y_true = [], []
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    for xb, yb in dl:
        xb, yb = xb.to(device), yb.to(device)
        loss, batch_size, pred = loss_batch(model, loss_func, xb, yb)
        total_loss += loss * batch_size
        total_size += batch_size
        predictions.append(pred)
        y_true.append(yb.cpu().numpy())

    mean_loss = total_loss / max(1, total_size)
    predictions = np.concatenate(predictions, axis=0) if predictions else np.array([])
    y_true = np.concatenate(y_true, axis=0) if y_true else np.array([])
    accuracy = float(np.mean((predictions == y_true))) if y_true.size else 0.0
    return mean_loss, accuracy, (y_true, predictions)


# ------------------------------------------------------------------------
# Domain Adaptation with CORAL
# ------------------------------------------------------------------------
def coral_loss(source, target):
  
    # source covariance
    xm = source - source.mean(0, keepdim=True)
    xc = xm.t() @ xm / (source.size(0) - 1 + 1e-5)

    # target covariance
    xmt = target - target.mean(0, keepdim=True)
    xct = xmt.t() @ xmt / (target.size(0) - 1 + 1e-5)

    return torch.mean((xc - xct) ** 2)


def fit_coral(epochs, model, opt, src_dl, tgt_dl, lambda_coral=1.0, loss_func=None,
              src_val_dl=None, early_stop_patience=10):
  
    if loss_func is None:
        loss_func = CrossEntropyLoss()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # AMP setup
    amp_enabled = torch.cuda.is_available()
    scaler = GradScaler(enabled=amp_enabled)

    # ---- λ warm-up settings ----
    warmup_frac = 0.20
    warmup_epochs = max(1, int(round(epochs * warmup_frac)))

    best_val = float("inf")
    best_state = None
    patience = max(1, int(early_stop_patience))
    since_improved = 0
    improve_eps = 1e-6  # minimal improvement threshold

    print("EPOCH\tSrc Loss\tCORAL\tTotal Loss\t(lambda_eff)\tValLoss\tValAcc")
    for epoch in range(epochs):
        # Linear warm-up from 0→1 over warmup_epochs, then stay at 1.0
        warm = 1.0 if (epoch + 1) > warmup_epochs else (epoch + 1) / warmup_epochs
        eff_lambda = lambda_coral * warm

        model.train()
        total_loss = total_src = total_coral = 0.0
        n_batches = min(len(src_dl), len(tgt_dl))

        for (xs, ys), (xt, _) in zip(src_dl, tgt_dl):
            xs, ys, xt = xs.to(device), ys.to(device), xt.to(device)

            opt.zero_grad(set_to_none=True)

            # Forward pass in mixed precision
            with autocast(enabled=amp_enabled):
                logits_s, feats_s = model(xs, return_feats=True)
                _, feats_t = model(xt, return_feats=True)

                loss_src = loss_func(logits_s, ys)
                loss_c = coral_loss(feats_s, feats_t)
                loss = loss_src + eff_lambda * loss_c

            # Backward + step with GradScaler
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            total_loss += loss.item()
            total_src  += loss_src.item()
            total_coral+= loss_c.item()

        # ---- validation on source split (if provided) ----
        val_loss = float("nan")
        val_acc  = float("nan")
        if src_val_dl is not None:
            model.eval()
            with torch.no_grad():
                val_loss, val_acc, _ = validate(model, src_val_dl, loss_func)

            # early stopping on val_loss
            if val_loss < best_val - improve_eps:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                since_improved = 0
            else:
                since_improved += 1

        print(f"{epoch}\t{total_src/n_batches:.6f}\t{total_coral/n_batches:.6e}\t"
              f"{total_loss/n_batches:.6f}\t({eff_lambda:.3g})\t"
              f"{val_loss:.6f}\t{val_acc:.6f}")

        # stop if patience exceeded
        if src_val_dl is not None and since_improved >= patience:
            print(f"[EARLY STOP] No val improvement for {patience} epochs. Restoring best weights.")
            if best_state is not None:
                model.load_state_dict(best_state)
            break

    # ensure best weights are loaded after loop ends
    if src_val_dl is not None and best_state is not None:
        model.load_state_dict(best_state)