import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from tqdm import tqdm

# Đọc dữ liệu
df = pd.read_csv(r'/home/ducanh/Financial Risk & Fraud Detection/Credit Card Fraud Detection/Datasets/CCFD/Combined_Data/combined_data.csv')

# Xử lý thời gian
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['trans_date_trans_time_numeric'] = df['trans_date_trans_time'].apply(lambda x: x.timestamp())
df['trans_hour'] = df['trans_date_trans_time'].dt.time.apply(lambda x: str(x)[:2])

df['dob'] = pd.to_datetime(df['dob'])
df['cust_age'] = df['dob'].dt.year.apply(lambda x: 2021 - x)
df['cust_age_groups'] = df['cust_age'].apply(lambda x: 'below 10' if x < 10 
                                              else ('10-20' if 10 <= x < 20 
                                              else ('20-30' if 20 <= x < 30 
                                              else ('30-40' if 30 <= x < 40 
                                              else ('40-50' if 40 <= x < 50 
                                              else ('50-60' if 50 <= x < 60 
                                              else ('60-70' if 60 <= x < 70 
                                              else ('70-80' if 70 <= x < 80 else 'Above 80'))))))))

# Mapping age groups by mean fraud amount
age_piv_2 = pd.pivot_table(data=df,
                           index='cust_age_groups',
                           columns='is_fraud',
                           values='amt',
                           aggfunc=np.mean)
age_piv_2.sort_values(by=1, ascending=True, inplace=True)
age_dic = {k: v for (k, v) in zip(age_piv_2.index.values, age_piv_2.reset_index().index.values)}
df['cust_age_groups'] = df['cust_age_groups'].map(age_dic)

# Mapping merchant categories
merch_cat = df[df['is_fraud'] == 1].groupby('category')['amt'].mean().sort_values(ascending=True)
merch_cat_dic = {k: v for (k, v) in zip(merch_cat.index.values, merch_cat.reset_index().index.values)}
df['category'] = df['category'].map(merch_cat_dic)

# Mapping jobs
job_txn_piv_2 = pd.pivot_table(data=df,
                               index='job',
                               columns='is_fraud',
                               values='amt',
                               aggfunc=np.mean)
job_cat_dic = {k: v for (k, v) in zip(job_txn_piv_2.index.values, job_txn_piv_2.reset_index().index.values)}
df['job'] = df['job'].map(job_cat_dic)

# Factorize categorical fields
for col in ['merchant','last','street','city','zip','state']:
    df[f'{col}_num'] = pd.factorize(df[col])[0]

# One-hot gender
df = pd.get_dummies(data=df, columns=['gender'], drop_first=True, dtype='int')

# Drop unused columns
drop_cols = ['Unnamed: 0', 'trans_date_trans_time', 'merchant', 'first', 'last', 'street', 'city', 'state', 'lat', 'long', 'dob',
             'unix_time', 'merch_lat', 'merch_long', 'city_pop']
df.drop(columns=drop_cols, errors='ignore', inplace=True)

# Train-test split theo cc_num (800 users train, 199 users test)
unique_cc_nums = df['cc_num'].unique()
assert len(unique_cc_nums) == 999, "Số lượng user không đạt 999, kiểm tra lại dữ liệu."
np.random.seed(42)
np.random.shuffle(unique_cc_nums)
train_cc_nums = unique_cc_nums[:800]
test_cc_nums = unique_cc_nums[800:]

train = df[df['cc_num'].isin(train_cc_nums)]
test = df[df['cc_num'].isin(test_cc_nums)]

print("Train shape:", train.shape)
print("Test shape:", test.shape)

# Xóa cột trans_num nếu có
for subset in [train, test]:
    if 'trans_num' in subset.columns:
        subset.drop('trans_num', axis=1, inplace=True)

# Tách features và label
y_train = train['is_fraud']
X_train = train.drop('is_fraud', axis=1)
y_test = test['is_fraud']
X_test = test.drop('is_fraud', axis=1)

print('Shape of training data:', (X_train.shape, y_train.shape))
print('Shape of testing data:', (X_test.shape, y_test.shape))

# Scaling dữ liệu
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)
X_train_sc = pd.DataFrame(data=X_train_sc, columns=X_train.columns)
X_test_sc = pd.DataFrame(data=X_test_sc, columns=X_test.columns)

# Sequence creation
def create_sequences_transactional_expansion(df, memory_size):
    sequences, labels = [], []
    for user_id, group in df.groupby('cc_num'):
        group = group.sort_values(by='trans_date_trans_time_numeric')
        values = group.drop(columns=['is_fraud','cc_num']).values
        targets = group['is_fraud'].values
        n = len(group)
        for i in range(n):
            if i < memory_size:
                pad_needed = memory_size - (i + 1)
                pad = np.repeat(values[0:1, :], pad_needed, axis=0)
                seq = np.concatenate((pad, values[:i+1]), axis=0)
            else:
                seq = values[i-memory_size+1:i+1]
            sequences.append(seq)
            labels.append(targets[i])
    return np.array(sequences), np.array(labels)

memory_size = 10
train_seq_df = X_train_sc.copy(); train_seq_df['is_fraud'] = y_train.values
test_seq_df  = X_test_sc.copy();  test_seq_df['is_fraud']  = y_test.values
X_train_seq, y_train_seq = create_sequences_transactional_expansion(train_seq_df, memory_size)
X_test_seq,  y_test_seq  = create_sequences_transactional_expansion(test_seq_df, memory_size)

print("Sequence shape (train):", X_train_seq.shape)
print("Sequence shape (test):",  X_test_seq.shape)

# Dataset & DataLoader
class FraudDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

batch_size = 64
train_dataset = FraudDataset(torch.tensor(X_train_seq, dtype=torch.float32), torch.tensor(y_train_seq, dtype=torch.float32))
test_dataset  = FraudDataset(torch.tensor(X_test_seq,  dtype=torch.float32), torch.tensor(y_test_seq,  dtype=torch.float32))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

# 6️⃣ Định nghĩa Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: probabilities after sigmoid
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.where(targets == 1, inputs, 1 - inputs)
        loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

# 7️⃣ Xây dựng mô hình GRU
class FraudGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(FraudGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)

# Đánh giá mô hình
# 6️⃣ Định nghĩa lại evaluate_model để trả về metrics rõ ràng
def evaluate_model(loader, model, device):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch).squeeze().cpu().numpy()
            all_preds.extend(outputs)
            all_targets.extend(y_batch.cpu().numpy())
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # AUC
    auc = roc_auc_score(all_targets, all_preds)

    # Find best F1 threshold
    thresholds = [i * 0.1 for i in range(1, 10)]
    best_f1, best_t = 0, 0.5
    for t in thresholds:
        f1 = f1_score(all_targets, (all_preds > t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, t

    # Final predicted labels at best threshold
    preds_bin = (all_preds > best_t).astype(int)
    cm = confusion_matrix(all_targets, preds_bin)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    # Compute other metrics
    accuracy  = (tp + tn) / (tp + tn + fp + fn) if (tp+tn+fp+fn)>0 else 0
    precision = tp / (tp + fp)           if (tp+fp)>0 else 0
    recall    = tp / (tp + fn)           if (tp+fn)>0 else 0

    return {
        'threshold': best_t,
        'auc': auc,
        'f1': best_f1,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall
    }

# 8️⃣ Cập nhật train_model để in metrics đầy đủ
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device):
    best_combined_test = -np.inf
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", leave=False):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch}/{num_epochs} — Training Loss: {avg_loss:.4f}")

        # Evaluate on train
        train_metrics = evaluate_model(train_loader, model, device)
        print(f" → [Train metrics] AUC: {train_metrics['auc']:.4f}, "
              f"F1: {train_metrics['f1']:.4f} (thr={train_metrics['threshold']:.2f}), "
              f"Acc: {train_metrics['accuracy']:.4f}, "
              f"Prec: {train_metrics['precision']:.4f}, "
              f"Rec: {train_metrics['recall']:.4f}")

        # Evaluate on test
        test_metrics = evaluate_model(test_loader, model, device)
        print(f" → [Test  metrics] AUC: {test_metrics['auc']:.4f}, "
              f"F1: {test_metrics['f1']:.4f} (thr={test_metrics['threshold']:.2f}), "
              f"Acc: {test_metrics['accuracy']:.4f}, "
              f"Prec: {test_metrics['precision']:.4f}, "
              f"Rec: {test_metrics['recall']:.4f}")

        # Early-stopping on combined (you could also pick any single metric)
        combined_test = (test_metrics['auc'] + test_metrics['f1']) / 2
        if combined_test > best_combined_test:
            best_combined_test = combined_test
            epochs_no_improve = 0
            print(" *** New best test combined score! ***")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= 8:
                print(f"Early stopping at epoch {epoch}")
                break

# ——— Khởi tạo và chạy lại training ———
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FraudGRU(input_size=X_train_seq.shape[2], hidden_size=64, num_layers=2)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    print(f"Using {torch.cuda.device_count()} GPUs.")
model.to(device)

criterion = FocalLoss(alpha=0.25, gamma=2, reduction='mean')
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=100, device=device)
