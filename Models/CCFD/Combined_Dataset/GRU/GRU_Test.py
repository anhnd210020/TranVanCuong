import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from tqdm import tqdm
import math

# =======================================================
# 1. Định nghĩa lớp Shift_gcn
# =======================================================
class Shift_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes=1, coff_embedding=4, num_subset=3):
        super(Shift_gcn, self).__init__()
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

        self.Feature_Mask = nn.Parameter(torch.zeros(1, num_nodes, in_channels))
        nn.init.constant_(self.Feature_Mask, 0)

        self.bn = nn.BatchNorm1d(num_nodes * out_channels)
        self.relu = nn.ReLU()

        idx_in = np.empty(num_nodes * in_channels, dtype=np.int64)
        for i in range(num_nodes):
            for j in range(in_channels):
                idx_in[i * in_channels + j] = (i * in_channels + j + j * in_channels) % (in_channels * num_nodes)
        self.register_buffer('shift_in', torch.from_numpy(idx_in))

        idx_out = np.empty(num_nodes * out_channels, dtype=np.int64)
        for i in range(num_nodes):
            for j in range(out_channels):
                idx_out[i * out_channels + j] = (i * out_channels + j - j * out_channels) % (out_channels * num_nodes)
        self.register_buffer('shift_out', torch.from_numpy(idx_out))

    def forward(self, x0):
        x0_proc = x0.permute(0, 3, 1, 2).contiguous()
        n, c, t, v = x0_proc.size()

        x = x0_proc.permute(0, 2, 3, 1).contiguous().view(n * t, v * c)
        x = torch.index_select(x, 1, self.shift_in)
        x = x.view(n * t, v, c)

        x = x * (torch.tanh(self.Feature_Mask) + 1)
        x = torch.einsum('nwc,cd->nwd', x, self.Linear_weight) + self.Linear_bias

        x = x.view(n * t, -1)
        x = torch.index_select(x, 1, self.shift_out)
        x = self.bn(x)
        x = x.view(n, t, v, self.out_channels).permute(0, 3, 1, 2).contiguous()

        shortcut = self.down(x0_proc)
        x = self.relu(x + shortcut)
        return x

# =======================================================
# 2. Tiền xử lý dữ liệu và tạo sequence
# =======================================================
df = pd.read_csv(r'/home/ducanh/Financial Risk & Fraud Detection/Credit Card Fraud Detection/Datasets/CCFD/Combined_Data/combined_data.csv')

# Xử lý thời gian
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['trans_date_trans_time_numeric'] = df['trans_date_trans_time'].apply(lambda x: x.timestamp())
df['trans_hour'] = df['trans_date_trans_time'].dt.time.apply(lambda x: str(x)[:2])

# Xử lý ngày sinh và tính tuổi
df['dob'] = pd.to_datetime(df['dob'])
df['cust_age'] = df['dob'].dt.year.apply(lambda x: 2021 - x)
df['cust_age_groups'] = df['cust_age'].apply(lambda x: 'below 10' if x < 10 else 
                                             ('10-20' if x >= 10 and x < 20 else 
                                             ('20-30' if x >= 20 and x < 30 else 
                                             ('30-40' if x >= 30 and x < 40 else 
                                             ('40-50' if x >= 40 and x < 50 else 
                                             ('50-60' if x >= 50 and x < 60 else 
                                             ('60-70' if x >= 60 and x < 70 else 
                                             ('70-80' if x >= 70 and x < 80 else 'Above 80'))))))))

# Mapping cho cust_age_groups
age_piv_2 = pd.pivot_table(data=df,
                           index='cust_age_groups',
                           columns='is_fraud',
                           values='amt',
                           aggfunc=np.mean)
age_piv_2.sort_values(by=1, ascending=True, inplace=True)
age_dic = {k: v for (k, v) in zip(age_piv_2.index.values, age_piv_2.reset_index().index.values)}
df['cust_age_groups'] = df['cust_age_groups'].map(age_dic)

# Mapping cho category
merch_cat = df[df['is_fraud'] == 1].groupby('category')['amt'].mean().sort_values(ascending=True)
merch_cat_dic = {k: v for (k, v) in zip(merch_cat.index.values, merch_cat.reset_index().index.values)}
df['category'] = df['category'].map(merch_cat_dic)

# Mapping cho job
job_txn_piv_2 = pd.pivot_table(data=df,
                               index='job',
                               columns='is_fraud',
                               values='amt',
                               aggfunc=np.mean)
job_cat_dic = {k: v for (k, v) in zip(job_txn_piv_2.index.values, job_txn_piv_2.reset_index().index.values)}
df['job'] = df['job'].map(job_cat_dic)

df['merchant_num'] = pd.factorize(df['merchant'])[0]
df['last_num'] = pd.factorize(df['last'])[0]
df['street_num'] = pd.factorize(df['street'])[0]
df['city_num'] = pd.factorize(df['city'])[0]
df['zip_num'] = pd.factorize(df['zip'])[0]
df['state_num'] = pd.factorize(df['state'])[0]

df = pd.get_dummies(data=df, columns=['gender'], drop_first=True, dtype='int')

drop_cols = ['Unnamed: 0', 'trans_date_trans_time', 'merchant', 'first', 'last', 'street', 'city', 'state', 
             'lat', 'long', 'dob', 'unix_time', 'merch_lat', 'merch_long', 'city_pop', 'trans_num']
df.drop(columns=drop_cols, errors='ignore', inplace=True)

unique_cc = df['cc_num'].unique()
np.random.seed(42)
np.random.shuffle(unique_cc)
train_ids = unique_cc[:800]
test_ids  = unique_cc[800:]
train_df = df[df['cc_num'].isin(train_ids)]
test_df  = df[df['cc_num'].isin(test_ids)]

y_train = train_df['is_fraud'].values
X_train = train_df.drop(columns=['is_fraud', 'cc_num']).values
y_test  = test_df['is_fraud'].values
X_test  = test_df.drop(columns=['is_fraud', 'cc_num']).values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

def create_sequences(df_X, df_y, cc_series, memory_size):
    seqs, labs = [], []
    data = np.concatenate([cc_series.values.reshape(-1,1), df_X, df_y.reshape(-1,1)], axis=1)
    for user in np.unique(cc_series):
        user_data = data[data[:,0]==user]
        X_vals = user_data[:,1:-1]
        y_vals = user_data[:,-1]
        n = len(X_vals)
        for i in range(n):
            if i < memory_size:
                pad = np.repeat(X_vals[0:1], memory_size - (i+1), axis=0)
                seq = np.vstack([pad, X_vals[:i+1]])
            else:
                seq = X_vals[i-memory_size+1:i+1]
            seqs.append(seq)
            labs.append(y_vals[i])
    return np.array(seqs), np.array(labs)

memory_size = 50
X_train_seq, y_train_seq = create_sequences(X_train, y_train, train_df['cc_num'], memory_size)
X_test_seq,  y_test_seq  = create_sequences(X_test,  y_test,  test_df['cc_num'], memory_size)

class FraudDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

batch_size = 64
train_loader = DataLoader(FraudDataset(X_train_seq, y_train_seq), batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(FraudDataset(X_test_seq,  y_test_seq),  batch_size=batch_size, shuffle=False)

class FraudGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(FraudGRU, self).__init__()
        self.shift_gcn = Shift_gcn(in_channels=input_size, out_channels=hidden_size, num_nodes=1)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = x.unsqueeze(2)
        x = self.shift_gcn(x)
        x = x.squeeze(3).permute(0, 2, 1)
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)

def evaluate_model(loader, model, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch).squeeze().cpu().numpy()
            preds.extend(outputs)
            targets.extend(y_batch.numpy())
    preds = np.array(preds)
    targets = np.array(targets)
    auc = roc_auc_score(targets, preds)
    best_f1, best_thr = 0, 0.5
    for t in [0.1 * i for i in range(1,10)]:
        f1 = f1_score(targets, (preds>t).astype(int))
        if f1 > best_f1:
            best_f1, best_thr = f1, t
    cm = confusion_matrix(targets, (preds>best_thr).astype(int))
    TN, FP, FN, TP = cm.ravel() if cm.size==4 else (0,0,0,0)
    acc = (TP+TN)/(TP+TN+FP+FN)
    prec = TP/(TP+FP) if TP+FP>0 else 0
    rec  = TP/(TP+FN) if TP+FN>0 else 0
    combined = (best_f1 + auc) / 2
    return best_thr, best_f1, auc, combined, acc, prec, rec

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device):
    best_combined = -float('inf')
    epochs_no_improve = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

        # Evaluate on test set
        thr, f1, auc, combined, acc, prec, rec = evaluate_model(test_loader, model, device)
        print(f"Test Metrics - Threshold: {thr:.2f}, AUC: {auc:.4f}, F1: {f1:.4f}, "
              f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")

        if combined > best_combined:
            best_combined = combined
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= 3:
                print("Early stopping")
                break
    print("Training hoàn tất. Best combined metric:", best_combined)

input_size = X_train_seq.shape[2]
hidden_size = 64
num_layers = 2
model = FraudGRU(input_size, hidden_size, num_layers)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model.to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 50

train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device)