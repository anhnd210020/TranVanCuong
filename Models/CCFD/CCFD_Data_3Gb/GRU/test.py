import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from tqdm import tqdm

# =================== 1. Load & Tiền xử lý dữ liệu ===================
file_path = 'Financial Risk & Fraud Detection/Credit Card Fraud Detection/Datasets/CCFD/CCFD_Data_3Gb.csv'
df = pd.read_csv(file_path)

df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
# Sắp xếp dữ liệu theo 'cc_num' và 'trans_date_trans_time'
df = df.sort_values(by=['cc_num', 'trans_date_trans_time'])
# Tính toán delta T: hiệu số thời gian giữa các giao dịch của cùng một cc_num dựa trên cột 'unix_time'
df['delta_T'] = df.groupby('cc_num')['unix_time'].diff()
df['delta_T'] = df['delta_T'].fillna(0)  # Điền giá trị NaN thành 0

# =================== 2. Xác định các cột cần dùng ===================
# Cột nhóm và nhãn
group_col = 'cc_num'
label_col = 'is_fraud'
# Chỉ giữ 3 feature quan trọng
selected_features = ['amt', 'trans_time_day', 'delta_T']
feature_cols = selected_features  # Sử dụng 3 feature này

# =================== 3. Split dữ liệu thành Train/Test ===================
# Train-test split (stratify theo nhãn)
train_df, test_df = train_test_split(df, test_size=0.33, random_state=42, stratify=df[label_col])
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

# =================== 4. Scaling dữ liệu ===================
X_train = train_df[feature_cols].copy()
X_test  = test_df[feature_cols].copy()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols, index=train_df.index)
X_test_scaled  = pd.DataFrame(X_test_scaled, columns=feature_cols, index=test_df.index)

# Nối thêm cột nhóm và nhãn
train_scaled = X_train_scaled.copy()
train_scaled[group_col] = train_df[group_col].values
train_scaled[label_col] = train_df[label_col].values
if 'unix_time' in train_df.columns:
    train_scaled['unix_time'] = train_df['unix_time'].values
else:
    train_scaled['unix_time'] = pd.to_datetime(train_df['trans_date_trans_time']).apply(lambda x: x.timestamp()).values

test_scaled = X_test_scaled.copy()
test_scaled[group_col] = test_df[group_col].values
test_scaled[label_col] = test_df[label_col].values
if 'unix_time' in test_df.columns:
    test_scaled['unix_time'] = test_df['unix_time'].values
else:
    test_scaled['unix_time'] = pd.to_datetime(test_df['trans_date_trans_time']).apply(lambda x: x.timestamp()).values

# =================== 5. Tạo chuỗi theo Transactional Expansion ===================
def create_sequences_transactional_expansion(df, memory_size, order_col='unix_time', group_col='cc_num'):
    sequences, labels = [], []
    grouped = df.groupby(group_col)
    for user_id, group in grouped:
        # Sắp xếp theo cột order_col để đảm bảo thứ tự thời gian
        group = group.sort_values(by=order_col)
        # Lấy giá trị của các features (loại trừ nhãn và group)
        values = group.drop(columns=[label_col, group_col]).values
        targets = group[label_col].values
        n = len(group)
        for i in range(n):
            if i < memory_size:
                pad_needed = memory_size - (i + 1)
                pad = np.repeat(values[0:1, :], pad_needed, axis=0)
                seq = np.concatenate((pad, values[:i+1]), axis=0)
            else:
                seq = values[i - memory_size + 1:i + 1]
            sequences.append(seq)
            labels.append(targets[i])
    return np.array(sequences), np.array(labels)

memory_size = 100  # Bạn có thể điều chỉnh giá trị này
X_train_seq, y_train_seq = create_sequences_transactional_expansion(train_scaled, memory_size, order_col='unix_time', group_col=group_col)
X_test_seq,  y_test_seq  = create_sequences_transactional_expansion(test_scaled, memory_size, order_col='unix_time', group_col=group_col)
print("Sequence shape (train):", X_train_seq.shape)
print("Sequence shape (test):", X_test_seq.shape)

# =================== 6. Định nghĩa Dataset cho PyTorch ===================
class FraudDataset(Dataset):
    def __init__(self, X, y):
        self.X = X  # shape: (num_sequences, sequence_length, num_features)
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

batch_size = 64
train_dataset = FraudDataset(torch.tensor(X_train_seq, dtype=torch.float32),
                               torch.tensor(y_train_seq, dtype=torch.float32))
test_dataset = FraudDataset(torch.tensor(X_test_seq, dtype=torch.float32),
                              torch.tensor(y_test_seq, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# =================== 7. Định nghĩa mô hình GRU ===================
class FraudGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(FraudGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out, _ = self.gru(x)  # out: (batch, seq_length, hidden_size)
        out = self.fc(out[:, -1, :])  # Lấy output của bước thời gian cuối cùng
        return self.sigmoid(out)  # Áp dụng sigmoid để đảm bảo giá trị đầu ra nằm trong [0, 1]

# =================== 8. Hàm đánh giá mô hình ===================
def evaluate_model(loader, model, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch).squeeze()
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    auc = roc_auc_score(all_targets, all_preds)
    
    thresholds = [0.1 * i for i in range(1, 10)]
    best_f1 = 0
    best_threshold = 0.5
    for t in thresholds:
        binary_preds = (all_preds > t).astype(int)
        f1 = f1_score(all_targets, binary_preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
    combined_metric = (best_f1 + auc) / 2
    
    binary_preds = (all_preds > best_threshold).astype(int)
    cm = confusion_matrix(all_targets, binary_preds)
    TP = cm[1, 1] if cm.shape[0] > 1 and cm.shape[1] > 1 else 0
    FP = cm[0, 1] if cm.shape[1] > 1 else 0
    FN = cm[1, 0] if cm.shape[0] > 1 else 0
    TN = cm[0, 0]
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    
    return best_threshold, best_f1, auc, combined_metric, accuracy, precision, recall

# =================== 9. Hàm Training ===================
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device):
    best_loss = float('inf')
    best_combined_metric_test = -float('inf')
    epochs_without_improvement = 0

    best_epoch = None
    best_train_metrics = None
    best_test_metrics = None

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            outputs = outputs.squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        average_loss = total_loss / len(train_loader)
        print(f'\nEpoch {epoch+1}, Loss: {average_loss:.4f}')
        
        train_threshold, train_f1, train_auc, train_combined, train_acc, train_prec, train_rec = evaluate_model(train_loader, model, device)
        print(f"Train Metrics - Best Threshold: {train_threshold:.2f}, F1: {train_f1:.4f}, AUC: {train_auc:.4f}, Combined: {train_combined:.4f}, Accuracy: {train_acc:.4f}, Precision: {train_prec:.4f}, Recall: {train_rec:.4f}")
        
        test_threshold, test_f1, test_auc, test_combined, test_acc, test_prec, test_rec = evaluate_model(test_loader, model, device)
        print(f"Test Metrics  - Best Threshold: {test_threshold:.2f}, F1: {test_f1:.4f}, AUC: {test_auc:.4f}, Combined: {test_combined:.4f}, Accuracy: {test_acc:.4f}, Precision: {test_prec:.4f}, Recall: {test_rec:.4f}")
        
        if test_combined > best_combined_metric_test:
            best_combined_metric_test = test_combined
            best_epoch = epoch + 1
            best_train_metrics = (train_f1, train_auc, train_combined)
            best_test_metrics = (test_f1, test_auc, test_combined)
            print(f'*** Best metrics updated at epoch {epoch+1} ***')
        
        if average_loss < best_loss:
            best_loss = average_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= 8:
                print(f'Early stopping triggered at epoch {epoch+1}')
                break
    print("\n========== Final Best Results ==========")
    print(f"Best Epoch: {best_epoch}")
    print(f"Train Metrics - F1: {best_train_metrics[0]:.4f}, AUC: {best_train_metrics[1]:.4f}, Combined: {best_train_metrics[2]:.4f}")
    print(f"Test Metrics  - F1: {best_test_metrics[0]:.4f}, AUC: {best_test_metrics[1]:.4f}, Combined: {best_test_metrics[2]:.4f}")

# =================== 10. Model Initialization & Training ===================
input_size = X_train_seq.shape[2]  # Sẽ là 3 vì chúng ta chỉ sử dụng 3 features: amt, trans_time_day, delta_T
hidden_size = 64
num_layers = 2
model = FraudGRU(input_size, hidden_size, num_layers)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs for training.")
    model = nn.DataParallel(model)
model.to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 100

train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device)