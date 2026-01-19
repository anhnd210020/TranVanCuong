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

# ====================================
# 1. Data loading & preprocessing
# ====================================
cols_to_keep = ['cc_num', 'is_fraud', 'trans_date_trans_time', 'dob', 'amt', 
                'category', 'job', 'merchant', 'last', 'street', 'city', 'state', 'zip', 'gender']
df = pd.read_csv(r'/home/ducanh/Financial Risk & Fraud Detection/Credit Card Fraud Detection/Datasets/CCFD/CCFD_Data_3Gb.csv')
df = df[cols_to_keep]

df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['trans_date_trans_time_numeric'] = df['trans_date_trans_time'].apply(lambda x: x.timestamp())
df['trans_hour'] = df['trans_date_trans_time'].dt.hour

df['dob'] = pd.to_datetime(df['dob'])
df['cust_age'] = df['dob'].dt.year.apply(lambda x: 2021 - x)
df['cust_age_groups'] = df['cust_age'].apply(
    lambda x: 'below 10' if x < 10 else
              '10-20'     if x < 20 else
              '20-30'     if x < 30 else
              '30-40'     if x < 40 else
              '40-50'     if x < 50 else
              '50-60'     if x < 60 else
              '60-70'     if x < 70 else
              '70-80'     if x < 80 else
              'Above 80'
)

age_piv_2 = pd.pivot_table(df, index='cust_age_groups', columns='is_fraud', values='amt', aggfunc=np.mean)
age_piv_2.sort_values(by=1, ascending=True, inplace=True)
age_dic = {k: v for (k, v) in zip(age_piv_2.index.values, range(len(age_piv_2)))}
df['cust_age_groups'] = df['cust_age_groups'].map(age_dic)

merch_cat = df[df['is_fraud'] == 1].groupby('category')['amt'].mean().sort_values()
merch_cat_dic = {k: v for (k, v) in zip(merch_cat.index.values, range(len(merch_cat)))}
df['category'] = df['category'].map(merch_cat_dic)

job_txn_piv_2 = pd.pivot_table(df, index='job', columns='is_fraud', values='amt', aggfunc=np.mean)
job_cat_dic = {k: v for (k, v) in zip(job_txn_piv_2.index.values, range(len(job_txn_piv_2)))}
df['job'] = df['job'].map(job_cat_dic)

# Factorize text columns
for col in ['merchant', 'last', 'street', 'city', 'zip', 'state']:
    df[f'{col}_num'] = pd.factorize(df[col])[0]

# One-hot encoding for gender
df = pd.get_dummies(df, columns=['gender'], drop_first=True, dtype='int')

# Drop unused columns
drop_cols = ['trans_date_trans_time', 'first', 'last', 'street', 'city', 'state', 'dob', 'merchant']
df.drop(columns=drop_cols, errors='ignore', inplace=True)

# Train/test split by card number
unique_cc = df['cc_num'].unique()
np.random.seed(42)
np.random.shuffle(unique_cc)
sel = unique_cc[:1246]
train_cc, test_cc = sel[:996], sel[996:]
train = df[df['cc_num'].isin(train_cc)]
test  = df[df['cc_num'].isin(test_cc)]

y_train = train['is_fraud']; X_train = train.drop('is_fraud', axis=1)
y_test  = test['is_fraud'];  X_test  = test.drop('is_fraud', axis=1)

# Standard scaling
sc = StandardScaler()
X_train_sc = pd.DataFrame(sc.fit_transform(X_train), columns=X_train.columns)
X_test_sc  = pd.DataFrame(sc.transform(X_test),  columns=X_test.columns)

# Sequence creation with transactional expansion
def create_sequences_transactional_expansion(df, memory_size):
    seqs, labels = [], []
    for _, group in df.groupby('cc_num'):
        group = group.sort_values('trans_date_trans_time_numeric')
        vals   = group.drop(['is_fraud','cc_num'], axis=1).values
        targs  = group['is_fraud'].values
        n = len(group)
        for i in range(n):
            if i < memory_size:
                pad = np.repeat(vals[0:1], memory_size-(i+1), axis=0)
                seq = np.vstack([pad, vals[:i+1]])
            else:
                seq = vals[i-memory_size+1:i+1]
            seqs.append(seq)
            labels.append(targs[i])
    return np.array(seqs), np.array(labels)

train_df = X_train_sc.copy(); train_df['is_fraud'] = y_train.values
test_df  = X_test_sc.copy();  test_df['is_fraud']  = y_test.values
memory_size = 50
X_train_seq, y_train_seq = create_sequences_transactional_expansion(train_df, memory_size)
X_test_seq,  y_test_seq  = create_sequences_transactional_expansion(test_df,  memory_size)

# ====================================
# 2. Dataset & DataLoader
# ====================================
class FraudDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

batch_size = 64
train_loader = DataLoader(FraudDataset(torch.tensor(X_train_seq, dtype=torch.float32), torch.tensor(y_train_seq, dtype=torch.float32)), batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(FraudDataset(torch.tensor(X_test_seq,  dtype=torch.float32), torch.tensor(y_test_seq,  dtype=torch.float32)), batch_size=batch_size, shuffle=False)

# ====================================
# 3. Model definition: switch to GRU
# ====================================
class FraudGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(FraudGRU, self).__init__()
        self.gru     = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc      = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        out, _ = self.gru(x)              # out: (batch, seq_len, hidden_size)
        out = self.fc(out[:, -1, :])      # take last time-step
        return self.sigmoid(out)

# ====================================
# 4. Evaluation & training loops
# ====================================
def evaluate_model(loader, model, device):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch).squeeze(-1).cpu().numpy()
            all_preds.extend(outputs)
            all_targets.extend(y_batch.cpu().numpy())
    all_preds, all_targets = np.array(all_preds), np.array(all_targets)
    auc = roc_auc_score(all_targets, all_preds)
    thresholds = [0.1 * i for i in range(1, 10)]
    best_f1, best_t = 0, 0.5
    for t in thresholds:
        f1 = f1_score(all_targets, (all_preds > t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, t
    combined = (best_f1 + auc) / 2
    preds_bin = (all_preds > best_t).astype(int)
    cm = confusion_matrix(all_targets, preds_bin)
    TP = cm[1,1] if cm.shape == (2,2) else 0
    FP = cm[0,1] if cm.shape == (2,2) else 0
    FN = cm[1,0] if cm.shape == (2,2) else 0
    TN = cm[0,0]
    acc = (TP+TN)/(TP+TN+FP+FN) if (TP+TN+FP+FN)>0 else 0
    prec = TP/(TP+FP) if (TP+FP)>0 else 0
    rec = TP/(TP+FN) if (TP+FN)>0 else 0
    return best_t, best_f1, auc, combined, acc, prec, rec


def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device):
    best_comb_test = -float('inf')
    epochs_no_improve = 0
    best_epoch, best_train_metrics, best_test_metrics = None, None, None

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze(-1)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

        # Evaluate
        train_t, train_f1, train_auc, train_comb, train_acc, train_prec, train_rec = evaluate_model(train_loader, model, device)
        print(f"Train Metrics - Threshold: {train_t:.2f}, F1: {train_f1:.4f}, AUC: {train_auc:.4f}, Combined: {train_comb:.4f}, "
              f"Accuracy: {train_acc:.4f}, Precision: {train_prec:.4f}, Recall: {train_rec:.4f}")

        test_t, test_f1, test_auc, test_comb, test_acc, test_prec, test_rec = evaluate_model(test_loader, model, device)
        print(f"Test Metrics  - Threshold: {test_t:.2f}, F1: {test_f1:.4f}, AUC: {test_auc:.4f}, Combined: {test_comb:.4f}, "
              f"Accuracy: {test_acc:.4f}, Precision: {test_prec:.4f}, Recall: {test_rec:.4f}")

        # Track best
        if test_comb > best_comb_test:
            best_comb_test = test_comb
            best_epoch = epoch + 1
            best_train_metrics = (train_t, train_f1, train_auc, train_comb, train_acc, train_prec, train_rec)
            best_test_metrics  = (test_t, test_f1, test_auc, test_comb, test_acc, test_prec, test_rec)
            print(f"*** New best metrics at epoch {best_epoch} ***")

        # Early stopping on loss
        if avg_loss < best_comb_test:
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= 3:
                print("Early stopping...")
                break

    # Final summary
    print("\n=== Final Best Results ===")
    print(f"Best Epoch: {best_epoch}")
    bt, bf1, ba, bc, bap, bprec, br = best_train_metrics
    print(f"Train Metrics - Threshold: {bt:.2f}, F1: {bf1:.4f}, AUC: {ba:.4f}, Combined: {bc:.4f}, "
          f"Accuracy: {bap:.4f}, Precision: {bprec:.4f}, Recall: {br:.4f}")
    tt, tf1, ta, tc, tap, tprec, tr = best_test_metrics
    print(f"Test Metrics  - Threshold: {tt:.2f}, F1: {tf1:.4f}, AUC: {ta:.4f}, Combined: {tc:.4f}, "
          f"Accuracy: {tap:.4f}, Precision: {tprec:.4f}, Recall: {tr:.4f}")

# ====================================
# 5. Initialize and train (GRU)
# ====================================
input_size  = X_train_seq.shape[2]
hidden_size = 64
num_layers  = 2
model       = FraudGRU(input_size, hidden_size, num_layers)
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 50

train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device)
