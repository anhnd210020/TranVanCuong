# %%
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

# %%
# Data Loading and Pre-processing
df = pd.read_csv(r'/home/ducanh/Credit Card Transactions Fraud Detection/Datasets/combined_data.csv')

# Convert date columns and extract features
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['trans_hour'] = df['trans_date_trans_time'].dt.time.apply(lambda x: str(x)[:2])
df['dob'] = pd.to_datetime(df['dob'])
df['cust_age'] = df['dob'].dt.year.apply(lambda x: 2021 - x)
df['cust_age_groups'] = df['cust_age'].apply(lambda x: 'below 10' if x < 10 
                                              else ('10-20' if x >= 10 and x < 20 
                                              else ('20-30' if x >= 20 and x < 30 
                                              else ('30-40' if x >= 30 and x < 40 
                                              else ('40-50' if x >= 40 and x < 50 
                                              else ('50-60' if x >= 50 and x < 60 
                                              else ('60-70' if x >= 60 and x < 70 
                                              else ('70-80' if x >= 70 and x < 80 else 'Above 80'))))))))

# Drop unneeded columns (keep 'trans_date_trans_time' and 'cc_num')
drop_col = ['Unnamed: 0', 'merchant', 'first', 'last', 'street', 'city', 'state', 
            'lat', 'long', 'dob', 'unix_time', 'cust_age', 'merch_lat', 'merch_long', 'city_pop']
df.drop(drop_col, axis=1, inplace=True)

# Pivot table for cust_age_groups
age_piv_2 = pd.pivot_table(data=df, index='cust_age_groups', columns='is_fraud', values='amt', aggfunc=np.mean)
age_piv_2.sort_values(by=1, ascending=True, inplace=True)
age_dic = {k: v for (k, v) in zip(age_piv_2.index.values, age_piv_2.reset_index().index.values)}
df['cust_age_groups'] = df['cust_age_groups'].map(age_dic)

# Pivot table for category
merch_cat = df[df['is_fraud'] == 1].groupby('category')['amt'].mean().sort_values(ascending=True)
merch_cat_dic = {k: v for (k, v) in zip(merch_cat.index.values, merch_cat.reset_index().index.values)}
df['category'] = df['category'].map(merch_cat_dic)

# Pivot table for job
job_txn_piv_2 = pd.pivot_table(data=df, index='job', columns='is_fraud', values='amt', aggfunc=np.mean)
job_cat_dic = {k: v for (k, v) in zip(job_txn_piv_2.index.values, job_txn_piv_2.reset_index().index.values)}
df['job'] = df['job'].map(job_cat_dic)

df['trans_hour'] = df['trans_hour'].astype('int')
df = pd.get_dummies(data=df, columns=['gender'], drop_first=True, dtype='int')

# Convert trans_date_trans_time to timestamp (numerical) for scaling
df['trans_date_trans_time'] = df['trans_date_trans_time'].apply(lambda x: x.timestamp())

# Train-Test Split
train, test = train_test_split(df, test_size=0.33, random_state=42, stratify=df['is_fraud'])
print("Train shape:", train.shape)
print("Test shape:", test.shape)
# Save if needed
train.to_csv(r'/home/ducanh/Credit Card Transactions Fraud Detection/Datasets/combined_fraudTrain.csv', index=False)
test.to_csv(r'/home/ducanh/Credit Card Transactions Fraud Detection/Datasets/combined_fraudTest.csv', index=False)

# Drop column 'trans_num' if exists (as per your original code)
if 'trans_num' in train.columns:
    train.drop('trans_num', axis=1, inplace=True)
if 'trans_num' in test.columns:
    test.drop('trans_num', axis=1, inplace=True)

# Separate features and label
y_train = train['is_fraud']
X_train = train.drop('is_fraud', axis=1)
y_test = test['is_fraud']
X_test = test.drop('is_fraud', axis=1)
print('Shape of training data:', (X_train.shape, y_train.shape))
print('Shape of testing data:', (X_test.shape, y_test.shape))

# Scaling
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)
X_train_sc = pd.DataFrame(data=X_train_sc, columns=X_train.columns)
X_test_sc = pd.DataFrame(data=X_test_sc, columns=X_test.columns)

# %%
sequence_length = 1000  # Number of transactions in one sequence

def create_sequences_predict_all(df, sequence_length):
    sequences, labels = [], []
    grouped = df.groupby('cc_num')
    for user_id, group in grouped:
        group = group.sort_values(by='trans_date_trans_time')
        values = group.drop(columns=['is_fraud', 'cc_num']).values
        targets = group['is_fraud'].values
        n = len(group)
        for i in range(n):
            if i < sequence_length:
                pad_needed = sequence_length - (i + 1)
                pad = np.repeat(values[0:1, :], pad_needed, axis=0)
                seq = np.concatenate((pad, values[:i+1]), axis=0)
            else:
                seq = values[i-sequence_length+1:i+1]
            sequences.append(seq)
            labels.append(targets[i])
    return np.array(sequences), np.array(labels)

# Append label column and create sequences
train_seq_df = X_train_sc.copy()
train_seq_df['is_fraud'] = y_train.values
test_seq_df = X_test_sc.copy()
test_seq_df['is_fraud'] = y_test.values

X_train_seq, y_train_seq = create_sequences_predict_all(train_seq_df, sequence_length)
X_test_seq, y_test_seq = create_sequences_predict_all(test_seq_df, sequence_length)
print("Sequence shape (train):", X_train_seq.shape)
print("Sequence shape (test):", X_test_seq.shape)

# %%
# Compute time intervals (delta) for each sequence.
# We need the index of 'trans_date_trans_time' in our DataFrame columns.
timestamp_idx = X_train_sc.columns.get_loc('trans_date_trans_time')

def compute_delta(seq, timestamp_idx):
    # seq: numpy array of shape (sequence_length, num_features)
    timestamps = seq[:, timestamp_idx]
    delta = np.diff(timestamps, prepend=timestamps[0])
    return delta.reshape(-1, 1)

# Custom Dataset that returns (sequence, delta_sequence, label)
class FraudDataset(Dataset):
    def __init__(self, X, y, timestamp_idx):
        self.X = X  # shape: (num_sequences, sequence_length, num_features)
        self.y = y
        self.timestamp_idx = timestamp_idx

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        seq = self.X[idx]  # (sequence_length, num_features)
        delta_seq = compute_delta(seq, self.timestamp_idx)  # (sequence_length, 1)
        return (torch.tensor(seq, dtype=torch.float32),
                torch.tensor(delta_seq, dtype=torch.float32),
                torch.tensor(self.y[idx], dtype=torch.float32))

batch_size = 512
train_dataset = FraudDataset(X_train_seq, y_train_seq, timestamp_idx)
test_dataset = FraudDataset(X_test_seq, y_test_seq, timestamp_idx)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# =============================================================================
# 6. Define Time-Aware GRU Model
# =============================================================================
class TimeAwareGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TimeAwareGRUCell, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Time-aware state parameters
        self.W_sh = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_sx = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W_st = nn.Linear(1, hidden_dim, bias=False)
        self.b_s  = nn.Parameter(torch.zeros(hidden_dim))
        
        # Time-aware gate parameters
        self.WTh = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.WTx = nn.Linear(input_dim, hidden_dim, bias=False)
        self.WTs = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.b_T  = nn.Parameter(torch.zeros(hidden_dim))
        
        # GRU reset gate parameters
        self.W_rh = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_rx = nn.Linear(input_dim, hidden_dim, bias=False)
        self.b_r  = nn.Parameter(torch.zeros(hidden_dim))
        
        # GRU update gate parameters
        self.W_zh = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_zx = nn.Linear(input_dim, hidden_dim, bias=False)
        self.b_z  = nn.Parameter(torch.zeros(hidden_dim))
        
        # Candidate hidden state parameters
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_x = nn.Linear(input_dim, hidden_dim, bias=False)
        self.b   = nn.Parameter(torch.zeros(hidden_dim))
    
    def forward(self, x_t, delta_t, h_prev):
        # Compute time-aware state s_t
        s_t = torch.tanh(self.W_sh(h_prev) + self.W_sx(x_t) + self.W_st(delta_t) + self.b_s)
        # Compute time-aware gate T_t
        T_t = torch.sigmoid(self.WTh(h_prev) + self.WTx(x_t) + self.WTs(s_t) + self.b_T)
        
        # Standard GRU gates
        r_t = torch.sigmoid(self.W_rh(h_prev) + self.W_rx(x_t) + self.b_r)
        z_t = torch.sigmoid(self.W_zh(h_prev) + self.W_zx(x_t) + self.b_z)
        # Candidate hidden state (with reset gate)
        h_tilde = torch.tanh(self.W_h(r_t * h_prev) + self.W_x(x_t) + self.b)
        # Update hidden state with time-aware modulation
        h_t = (1 - z_t) * (T_t * h_prev) + z_t * h_tilde
        return h_t

class FraudTimeAwareGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FraudTimeAwareGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.cell = TimeAwareGRUCell(input_dim, hidden_dim)
    
    def forward(self, x_seq, delta_seq, h0=None):
        batch_size, seq_len, _ = x_seq.size()
        if h0 is None:
            h_t = torch.zeros(batch_size, self.hidden_dim, device=x_seq.device)
        else:
            h_t = h0
        h_seq = []
        for t in range(seq_len):
            x_t = x_seq[:, t, :]
            delta_t = delta_seq[:, t, :]
            h_t = self.cell(x_t, delta_t, h_t)
            h_seq.append(h_t.unsqueeze(1))
        h_seq = torch.cat(h_seq, dim=1)  # (batch, seq_len, hidden_dim)
        return h_seq

class FraudTA_GRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(FraudTA_GRU, self).__init__()
        self.tagrus = FraudTimeAwareGRU(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, delta):
        # x: (batch, seq_len, input_size), delta: (batch, seq_len, 1)
        h_seq = self.tagrus(x, delta)
        # Use the last hidden state for classification
        out = self.fc(h_seq[:, -1, :])
        return self.sigmoid(out)

# %%
# 7. Training with Checkpoint Saving and Early Stopping
input_size = X_train_seq.shape[2]  # number of features (including trans_date_trans_time)
hidden_size = 64
model = FraudTA_GRU(input_size, hidden_size)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model.to(device)

def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for X_batch, delta_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            delta_batch = delta_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(X_batch, delta_batch).squeeze()
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    avg_loss = total_loss / len(data_loader)
    binary_preds = np.array(all_preds) > 0.5  # threshold=0.5
    f1 = f1_score(all_labels, binary_preds)
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except Exception:
        auc = 0.0
    return avg_loss, f1, auc

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, patience=8):
    best_combined = -float("inf")
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for X_batch, delta_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            delta_batch = delta_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch, delta_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / len(train_loader)
        
        # Evaluate on validation set
        val_loss, val_f1, val_auc = evaluate(model, val_loader, criterion, device)
        combined_metric = (val_f1 + val_auc) / 2.0

        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}, Combined: {combined_metric:.4f}")

        # Checkpoint saving based on combined metric
        if combined_metric > best_combined:
            best_combined = combined_metric
            best_model_wts = copy.deepcopy(model.state_dict())
            print("  >> New best checkpoint saved!")
        
        # Early stopping based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered!")
                break

    model.load_state_dict(best_model_wts)
    return model

num_epochs = 30
# Here we use test_loader as validation. In practice, you might use a separate validation set.
model = train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device, patience=8)

# %%
# 8. Prediction and Evaluation
model.eval()
y_pred_train_proba = []
with torch.no_grad():
    for X_batch, delta_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        delta_batch = delta_batch.to(device)
        outputs = model(X_batch, delta_batch).squeeze().cpu().numpy()
        y_pred_train_proba.extend(outputs)
y_pred_train_proba = np.array(y_pred_train_proba)

y_pred_test_proba = []
with torch.no_grad():
    for X_batch, delta_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        delta_batch = delta_batch.to(device)
        outputs = model(X_batch, delta_batch).squeeze().cpu().numpy()
        y_pred_test_proba.extend(outputs)
y_pred_test_proba = np.array(y_pred_test_proba)

# Create DataFrames for results
y_train_results = pd.DataFrame(y_pred_train_proba, columns=['pred_fraud'])
y_train_results['pred_not_fraud'] = 1 - y_train_results['pred_fraud']
y_train_results['y_train_actual'] = y_train_seq

y_test_results = pd.DataFrame(y_pred_test_proba, columns=['pred_fraud'])
y_test_results['pred_not_fraud'] = 1 - y_test_results['pred_fraud']
y_test_results['y_test_actual'] = y_test_seq

numbers = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for i in numbers:
    y_train_results[i] = y_train_results['pred_fraud'].map(lambda x: 1 if x > i else 0)
    y_test_results[i] = y_test_results['pred_fraud'].map(lambda x: 1 if x > i else 0)

cutoff_df = pd.DataFrame(columns=['Threshold', 'Accuracy', 'precision_score', 'recall_score', 'F1_score'])
for i in numbers:
    cm1 = confusion_matrix(y_train_results['y_train_actual'], y_train_results[i])
    TP, FP, FN, TN = cm1[1,1], cm1[0,1], cm1[1,0], cm1[0,0]
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score_value = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    cutoff_df.loc[i] = [i, accuracy, precision, recall, f1_score_value]

print("Train Evaluation:")
print(cutoff_df)

best_idx = cutoff_df['F1_score'].idxmax()
best_threshold = cutoff_df.loc[best_idx, 'Threshold']
best_accuracy = cutoff_df.loc[best_idx, 'Accuracy']
best_precision = cutoff_df.loc[best_idx, 'precision_score']
best_recall = cutoff_df.loc[best_idx, 'recall_score']
best_f1_score = cutoff_df.loc[best_idx, 'F1_score']
best_auc = roc_auc_score(y_train_results['y_train_actual'], y_train_results['pred_fraud'])

print(f'Best Threshold (Train): {best_threshold:.4f}')
print(f'Best Accuracy (Train): {best_accuracy:.4f}')
print(f'Best Precision (Train): {best_precision:.4f}')
print(f'Best Recall (Train): {best_recall:.4f}')
print(f'Best F1 Score (Train): {best_f1_score:.4f}')
print(f'Best ROC_AUC Score (Train): {best_auc:.4f}')

cutoff_df_test = pd.DataFrame(columns=['Threshold', 'Accuracy', 'precision_score', 'recall_score', 'F1_score'])
for i in numbers:
    cm1 = confusion_matrix(y_test_results['y_test_actual'], y_test_results[i])
    TP, FP, FN, TN = cm1[1,1], cm1[0,1], cm1[1,0], cm1[0,0]
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score_value = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    cutoff_df_test.loc[i] = [i, accuracy, precision, recall, f1_score_value]

print("Test Evaluation:")
print(cutoff_df_test)

best_idx_test = cutoff_df_test['F1_score'].idxmax()
best_threshold_test = cutoff_df_test.loc[best_idx_test, 'Threshold']
best_accuracy_test = cutoff_df_test.loc[best_idx_test, 'Accuracy']
best_precision_test = cutoff_df_test.loc[best_idx_test, 'precision_score']
best_recall_test = cutoff_df_test.loc[best_idx_test, 'recall_score']
best_f1_score_test = cutoff_df_test.loc[best_idx_test, 'F1_score']
best_auc_test = roc_auc_score(y_test_results['y_test_actual'], y_test_results['pred_fraud'])

print(f'Best Threshold (Test): {best_threshold_test:.4f}')
print(f'Best Accuracy (Test): {best_accuracy_test:.4f}')
print(f'Best Precision (Test): {best_precision_test:.4f}')
print(f'Best Recall (Test): {best_recall_test:.4f}')
print(f'Best F1 Score (Test): {best_f1_score_test:.4f}')
print(f'Best ROC_AUC Score (Test): {best_auc_test:.4f}')