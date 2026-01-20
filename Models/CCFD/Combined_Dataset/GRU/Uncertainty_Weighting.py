import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

# =========================
# GPU SETUP
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

# =========================
# LOAD CACHE
# =========================
from torch.utils.data import Dataset, DataLoader

CACHE_PATH = "TranVanCuong/seq_cache.pt"
cache = torch.load(CACHE_PATH, map_location="cpu")

feature_cols = cache["feature_cols"]
memory_size  = int(cache["memory_size"])

X_train_users = cache["X_train_users"]
y_train_users = cache["y_train_users"]
train_idx_user = cache["train_idx_user"].numpy()
train_idx_pos  = cache["train_idx_pos"].numpy()

X_test_users = cache["X_test_users"]
y_test_users = cache["y_test_users"]
test_idx_user = cache["test_idx_user"].numpy()
test_idx_pos  = cache["test_idx_pos"].numpy()

print("Loaded cache:", CACHE_PATH)
print("memory_size:", memory_size)
print("num_features:", X_train_users[0].shape[1])
print("train samples:", len(train_idx_user))
print("test  samples:", len(test_idx_user))


class WindowDataset(Dataset):
    def __init__(self, X_list, y_list, idx_user, idx_pos, memory_size, pad_mode="repeat_first"):
        self.X_list = X_list
        self.y_list = y_list
        self.idx_user = idx_user
        self.idx_pos = idx_pos
        self.M = memory_size
        assert pad_mode in ["repeat_first", "zeros"]
        self.pad_mode = pad_mode
        self.F = X_list[0].shape[1]

    def __len__(self):
        return len(self.idx_user)

    def __getitem__(self, i):
        u = int(self.idx_user[i])
        t = int(self.idx_pos[i])
        X_u = self.X_list[u]  # [n_i, F] (float16 hoặc float32)
        y_u = self.y_list[u]  # [n_i] float32

        start = t - self.M + 1
        if start >= 0:
            seq = X_u[start:t+1]  # [M, F]
        else:
            need = -start
            if self.pad_mode == "repeat_first":
                pad_row = X_u[0:1].expand(need, -1)
            else:
                pad_row = torch.zeros((need, self.F), dtype=X_u.dtype)
            seq = torch.cat([pad_row, X_u[0:t+1]], dim=0)

        label = y_u[t]
        return seq, label  # giữ dtype của seq, label float32

# =========================
# Shift-GCN (GIỮ NGUYÊN)
# =========================
import math
import numpy as np

class Shift_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A=None, num_nodes=25, coff_embedding=4, num_subset=3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_nodes = num_nodes

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.Linear_weight = nn.Parameter(torch.zeros(in_channels, out_channels))
        nn.init.normal_(self.Linear_weight, 0, math.sqrt(1.0 / out_channels))

        self.Linear_bias = nn.Parameter(torch.zeros(1, 1, out_channels))
        nn.init.constant_(self.Linear_bias, 0)

        self.Feature_Mask = nn.Parameter(torch.ones(1, num_nodes, in_channels))
        nn.init.constant_(self.Feature_Mask, 0)

        self.bn = nn.BatchNorm1d(num_nodes * out_channels)
        self.relu = nn.ReLU()

        index_array = np.empty(num_nodes * in_channels, dtype=np.int64)
        for i in range(num_nodes):
            for j in range(in_channels):
                index_array[i * in_channels + j] = (i * in_channels + j + j * in_channels) % (in_channels * num_nodes)
        self.register_buffer('shift_in', torch.from_numpy(index_array))

        index_array = np.empty(num_nodes * out_channels, dtype=np.int64)
        for i in range(num_nodes):
            for j in range(out_channels):
                index_array[i * out_channels + j] = (i * out_channels + j - j * out_channels) % (out_channels * num_nodes)
        self.register_buffer('shift_out', torch.from_numpy(index_array))

    def forward(self, x0, edge_index=None):
        x0_proc = x0.permute(0, 3, 1, 2).contiguous()
        n, c, t, v = x0_proc.size()

        x = x0_proc.permute(0, 2, 3, 1).contiguous()
        x = x.view(n * t, v * c)
        x = torch.index_select(x, 1, self.shift_in)
        x = x.view(n * t, v, c)

        x = x * (torch.tanh(self.Feature_Mask) + 1)
        x = torch.einsum('nwc,cd->nwd', x, self.Linear_weight).contiguous()
        x = x + self.Linear_bias

        x = x.view(n * t, -1)
        x = torch.index_select(x, 1, self.shift_out)
        x = self.bn(x)

        x = x.view(n, t, v, self.out_channels).permute(0, 3, 1, 2).contiguous()
        shortcut = self.down(x0_proc)
        x = x + shortcut
        x = self.relu(x)
        return x

# =========================
# MODEL (GIỮ SIGMOID NHƯ BASELINE)
# =========================
class FraudGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.shift_gcn = Shift_gcn(in_channels=input_size, out_channels=hidden_size, num_nodes=1)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.unsqueeze(2)  # (B, seq, 1, F)
        x = self.shift_gcn(x)
        x = x.squeeze(3).permute(0, 2, 1)  # (B, seq, hidden)
        out, _ = self.gru(x)
        logits = self.fc(out[:, -1, :]).squeeze(-1)  # (B,)
        return logits


def make_loaders(batch_size=256, num_workers=4):
    train_ds = WindowDataset(X_train_users, y_train_users, train_idx_user, train_idx_pos, memory_size)
    test_ds  = WindowDataset(X_test_users,  y_test_users,  test_idx_user,  test_idx_pos,  memory_size)

    persistent = (num_workers > 0)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=persistent
    )
    test_loader = DataLoader(
        test_ds, batch_size=512, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=persistent
    )
    return train_loader, test_loader

@torch.no_grad()
def evaluate_model_loader_metrics(model, loader):
    model.eval()
    preds, targets = [], []

    for Xb_cpu, yb_cpu in loader:
        Xb = Xb_cpu.to(device, non_blocking=True).float()
        yb = yb_cpu.to(device, non_blocking=True).float()
        logits = model(Xb)
        out = torch.sigmoid(logits).detach().cpu().numpy()
        preds.extend(out.tolist())
        targets.extend(yb_cpu.numpy().tolist())

    preds = np.array(preds)
    targets = np.array(targets)

    auc = roc_auc_score(targets, preds)
    thresholds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    best_f1, best_th = 0.0, 0.5
    for th in thresholds:
        f1 = f1_score(targets, (preds > th).astype(int))
        if f1 > best_f1:
            best_f1, best_th = f1, th

    cm = confusion_matrix(targets, (preds > best_th).astype(int))
    TP = cm[1,1] if cm.shape == (2,2) else 0
    FP = cm[0,1] if cm.shape == (2,2) else 0
    FN = cm[1,0] if cm.shape == (2,2) else 0
    TN = cm[0,0] if cm.shape[0] > 0 else 0
    acc = (TP+TN) / max(1, (TP+TN+FP+FN))
    prec = TP / max(1, (TP+FP))
    rec  = TP / max(1, (TP+FN))
    return best_th, best_f1, auc, (best_f1+auc)/2, acc, prec, rec


# =========================
# TRAIN LOOP
# =========================
def train_model_from_loader(model, train_loader, test_loader, criterion, optimizer, num_epochs=50):
    best_loss = float("inf")
    patience = 8
    bad_epochs = 0

    best_test_info = None
    best_score_track = -1.0
    best_epoch = -1

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        nb = 0

        for Xb_cpu, yb_cpu in train_loader:
            Xb = Xb_cpu.to(device, non_blocking=True).float()
            yb = yb_cpu.to(device, non_blocking=True).float()
            optimizer.zero_grad(set_to_none=True)
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += float(loss.detach().item())
            nb += 1

        avg_loss = total_loss / max(1, nb)
        print(f"\nEpoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.6f}")

        tr = evaluate_model_loader_metrics(model, train_loader)
        te = evaluate_model_loader_metrics(model, test_loader)
        print(f"Train - Th:{tr[0]:.2f} F1:{tr[1]:.4f} AUC:{tr[2]:.4f} Comb:{tr[3]:.4f} Acc:{tr[4]:.4f} Prec:{tr[5]:.4f} Rec:{tr[6]:.4f}")
        print(f"Test  - Th:{te[0]:.2f} F1:{te[1]:.4f} AUC:{te[2]:.4f} Comb:{te[3]:.4f} Acc:{te[4]:.4f} Prec:{te[5]:.4f} Rec:{te[6]:.4f}")

        if te[3] > best_score_track:
            best_score_track = te[3]
            best_test_info = te
            best_epoch = epoch + 1

        if avg_loss < best_loss:
            best_loss = avg_loss
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("Early stopping.")
                break

    return best_test_info, best_epoch

import json
from datetime import datetime

if __name__ == "__main__":
    input_size = X_train_users[0].shape[1]  # num_features
    hidden_size = 64
    num_layers = 2

    model = FraudGRU(input_size, hidden_size, num_layers).to(device)
    # pos_weight = (#neg / #pos) để cân bằng fraud (label=1)
    pos = 0
    neg = 0
    for yt in y_train_users:
        pos += int((yt == 1).sum().item())
        neg += int((yt == 0).sum().item())

    pos_weight = torch.tensor([neg / max(1, pos)], device=device, dtype=torch.float32)
    print("pos:", pos, "neg:", neg, "pos_weight:", float(pos_weight.item()))

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)


    train_loader, test_loader = make_loaders(batch_size=256, num_workers=4)

    start = time.perf_counter()
    best_results, best_epoch = train_model_from_loader(
        model, train_loader, test_loader, criterion, optimizer,
        num_epochs=50
    )

    elapsed_s = time.perf_counter() - start
    print(f"\nTraining finished in {elapsed_s:.2f}s")

    # Save model
    torch.save(model.state_dict(), "fraudgru_from_cache.pth")
    print("→ Saved model state_dict to fraudgru_from_cache.pth")

    # Save best result to file
    run_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result_path = "best_test_result.json"

    payload = {
        "finished_at": run_time_str,
        "elapsed_seconds": float(elapsed_s),
        "best_epoch": int(best_epoch),
        "best_test": None if best_results is None else {
            "threshold": float(best_results[0]),
            "f1": float(best_results[1]),
            "auc": float(best_results[2]),
            "comb": float(best_results[3]),
            "acc": float(best_results[4]),
            "precision": float(best_results[5]),
            "recall": float(best_results[6]),
        },
        "cache_path": CACHE_PATH,
        "model_path": "fraudgru_from_cache.pth",
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "input_size": int(input_size),
    }

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"→ Saved best results to {result_path}")

    if best_results is not None:
        print(
            f"\n>>> BEST TEST RESULT (epoch {best_epoch}): "
            f"Th:{best_results[0]:.2f} | F1:{best_results[1]:.4f} | "
            f"AUC:{best_results[2]:.4f} | Comb:{best_results[3]:.4f} | "
            f"Acc:{best_results[4]:.4f}"
        )
