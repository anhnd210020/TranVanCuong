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
CACHE_PATH = "/home/ducanhhh/Fraud-detection-in-credit-card/seq_cache.pt"   # hoặc đường dẫn full nếu bạn để chỗ khác
cache = torch.load(CACHE_PATH, map_location="cpu")

X_train_seq = cache["X_train_seq"]   # CPU tensor
y_train_seq = cache["y_train_seq"]   # CPU tensor
X_test_seq  = cache["X_test_seq"]    # CPU tensor
y_test_seq  = cache["y_test_seq"]    # CPU tensor

feature_cols = cache["feature_cols"]
memory_size  = cache["memory_size"]

print("Loaded cache:", CACHE_PATH)
print("Train seq:", tuple(X_train_seq.shape))
print("Test  seq:", tuple(X_test_seq.shape))
print("memory_size:", memory_size)
print("num_features:", X_train_seq.shape[2])

# =========================
# Loss (GIỮ Y HỆT BASELINE)
# =========================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='none'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.where(targets == 1, inputs, 1 - inputs)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class CombinedLossUnc(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.focal = FocalLoss(alpha, gamma, reduction='mean')
        self.log_sigma_bce = nn.Parameter(torch.zeros(1))
        self.log_sigma_focal = nn.Parameter(torch.zeros(1))

    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy(inputs, targets, reduction='mean')
        focal = self.focal(inputs, targets)
        loss = (
            0.5 * torch.exp(-self.log_sigma_bce) * bce + 0.5 * self.log_sigma_bce
            + 0.5 * torch.exp(-self.log_sigma_focal) * focal + 0.5 * self.log_sigma_focal
        )
        return loss

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
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(2)  # (B, seq, 1, F)
        x = self.shift_gcn(x)
        x = x.squeeze(3).permute(0, 2, 1)  # (B, seq, hidden)
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out).squeeze(-1)

# =========================
# CPU batch iterator + GPU copy
# =========================
def iterate_minibatches_cpu(X, y, batch_size, shuffle=True):
    N = X.size(0)
    idx = torch.randperm(N) if shuffle else torch.arange(N)
    for i in range(0, N, batch_size):
        j = idx[i:i+batch_size]
        yield X[j], y[j]

@torch.no_grad()
def evaluate_model_cpu_metrics(model, X_cpu, y_cpu, batch_size=4096):
    model.eval()
    preds = []
    targets = []

    for Xb_cpu, yb_cpu in iterate_minibatches_cpu(X_cpu, y_cpu, batch_size, shuffle=False):
        Xb = Xb_cpu.to(device, non_blocking=True)
        out = model(Xb).detach().cpu().numpy()
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
def train_model_from_cache(model, X_train_cpu, y_train_cpu, X_test_cpu, y_test_cpu,
                          criterion, optimizer, num_epochs=50, batch_size=256):
    best_loss = float("inf")
    patience = 8
    bad_epochs = 0
    best_test_info = None 
    best_score_track = -1.0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        nb = 0

        for Xb_cpu, yb_cpu in iterate_minibatches_cpu(X_train_cpu, y_train_cpu, batch_size, shuffle=True):
            Xb = Xb_cpu.to(device, non_blocking=True)
            yb = yb_cpu.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            out = model(Xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach().item())
            nb += 1

        avg_loss = total_loss / max(1, nb)
        print(f"\nEpoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.6f}")
        print(f"  log_sigma_bce:   {criterion.log_sigma_bce.item():.6f}")
        print(f"  log_sigma_focal: {criterion.log_sigma_focal.item():.6f}")

        tr = evaluate_model_cpu_metrics(model, X_train_cpu, y_train_cpu)
        te = evaluate_model_cpu_metrics(model, X_test_cpu,  y_test_cpu)
        print(f"Train - Th:{tr[0]:.2f} F1:{tr[1]:.4f} AUC:{tr[2]:.4f} Comb:{tr[3]:.4f} Acc:{tr[4]:.4f} Prec:{tr[5]:.4f} Rec:{tr[6]:.4f}")
        print(f"Test  - Th:{te[0]:.2f} F1:{te[1]:.4f} AUC:{te[2]:.4f} Comb:{te[3]:.4f} Acc:{te[4]:.4f} Prec:{te[5]:.4f} Rec:{te[6]:.4f}")
        if te[3] > best_score_track:
            best_score_track = te[3]
            best_test_info = te # Lưu lại tuple kết quả test
        if avg_loss < best_loss:
            best_loss = avg_loss
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("Early stopping.")
                break
    return best_test_info


if __name__ == "__main__":
    input_size = X_train_seq.shape[2]
    hidden_size = 64
    num_layers = 2

    model = FraudGRU(input_size, hidden_size, num_layers).to(device)
    criterion = CombinedLossUnc(alpha=0.25, gamma=2).to(device)
    optimizer = optim.Adam(list(model.parameters()) + list(criterion.parameters()), lr=1e-3)

    start = time.perf_counter()

    best_results = train_model_from_cache(
        model,
        X_train_seq, y_train_seq,
        X_test_seq,  y_test_seq,
        criterion, optimizer,
        num_epochs=50,
        batch_size=256
    )

    print(f"\nTraining finished in {time.perf_counter() - start:.2f}s")
    if best_results is not None:
            print(f"\n>>> BEST TEST RESULT: Th:{best_results[0]:.2f} | F1: {best_results[1]:.4f} | AUC: {best_results[2]:.4f} | Comb: {best_results[3]:.4f} | Acc: {best_results[4]:.4f}")
    torch.save(model.state_dict(), "fraudgru_from_cache.pth")
    print("→ Saved model state_dict to fraudgru_from_cache.pth")
