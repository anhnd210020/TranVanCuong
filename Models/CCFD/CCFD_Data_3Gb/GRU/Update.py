import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math 
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from tqdm import tqdm

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='none'):
        super(FocalLoss, self).__init__()
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
        else:
            return focal_loss

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, bce_weight=1.0, focal_weight=1.0, reduction='mean'):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.reduction = reduction
        self.focal_loss = FocalLoss(alpha, gamma, reduction='none')

    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy(inputs, targets, reduction='none')
        focal = self.focal_loss(inputs, targets)
        combined = self.bce_weight * bce + self.focal_weight * focal
        if self.reduction == 'mean':
            return combined.mean()
        elif self.reduction == 'sum':
            return combined.sum()
        else:
            return combined

class Shift_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A=None, num_nodes=25, coff_embedding=4, num_subset=3):
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

df['merchant_num'] = pd.factorize(df['merchant'])[0]
df['last_num']     = pd.factorize(df['last'])[0]
df['street_num']   = pd.factorize(df['street'])[0]
df['city_num']     = pd.factorize(df['city'])[0]
df['zip_num']      = pd.factorize(df['zip'])[0]
df['state_num']    = pd.factorize(df['state'])[0]

df = pd.get_dummies(df, columns=['gender'], drop_first=True, dtype='int')
drop_cols = ['trans_date_trans_time', 'first', 'last', 'street', 'city', 'state', 'dob', 'merchant']
df.drop(columns=drop_cols, errors='ignore', inplace=True)

unique_cc = df['cc_num'].unique()
np.random.seed(42)
np.random.shuffle(unique_cc)
sel = unique_cc[:1246]
train_cc, test_cc = sel[:996], sel[996:]
train = df[df['cc_num'].isin(train_cc)]
test  = df[df['cc_num'].isin(test_cc)]

y_train = train['is_fraud']; X_train = train.drop('is_fraud', axis=1)
y_test  = test['is_fraud'];  X_test  = test.drop('is_fraud', axis=1)

sc = StandardScaler()
X_train_sc = pd.DataFrame(sc.fit_transform(X_train), columns=X_train.columns)
X_test_sc  = pd.DataFrame(sc.transform(X_test), columns=X_test.columns)


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

memory_size = 1854
X_train_seq, y_train_seq = create_sequences_transactional_expansion(train_df, memory_size)
X_test_seq,  y_test_seq  = create_sequences_transactional_expansion(test_df, memory_size)
# =======================================================
# 3. Định nghĩa Dataset cho PyTorch
# =======================================================
class FraudDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

batch_size = 64
train_loader = DataLoader(FraudDataset(torch.tensor(X_train_seq, dtype=torch.float32),
                                       torch.tensor(y_train_seq, dtype=torch.float32)),
                          batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(FraudDataset(torch.tensor(X_test_seq, dtype=torch.float32),
                                       torch.tensor(y_test_seq, dtype=torch.float32)),
                          batch_size=batch_size, shuffle=False)

# ----------------------------
# Model definition
# ----------------------------
class FraudGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(FraudGRU, self).__init__()
        self.shift_gcn = Shift_gcn(in_channels=input_size, out_channels=hidden_size, num_nodes=1)
        self.gru       = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc        = nn.Linear(hidden_size, 1)
        self.sigmoid   = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(2)                        # (batch, seq, 1, features)
        x = self.shift_gcn(x)
        x = x.squeeze(3).permute(0, 2, 1)         # (batch, seq, hidden)
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)
# =======================================================
# 5. Hàm đánh giá mô hình
# =======================================================
def evaluate_model(loader, model, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            # only squeeze the last dimension
            outputs = model(X_batch).squeeze(-1).cpu().numpy()
            all_preds.extend(outputs)
            all_targets.extend(y_batch.cpu().numpy())
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    auc = roc_auc_score(all_targets, all_preds)
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
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

# =======================================================
# 6. Hàm huấn luyện mô hình tích hợp tqdm
# =======================================================
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device):
    best_comb_test = -float('inf')
    epochs_no_improve = 0
    best_epoch = None
    best_train_metrics, best_test_metrics = None, None

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
        print(f"Train Metrics - Best Threshold: {train_t:.2f}, F1: {train_f1:.4f}, AUC: {train_auc:.4f}, Combined: {train_comb:.4f}, "
              f"Accuracy: {train_acc:.4f}, Precision: {train_prec:.4f}, Recall: {train_rec:.4f}")

        test_t, test_f1, test_auc, test_comb, test_acc, test_prec, test_rec = evaluate_model(test_loader, model, device)
        print(f"Test Metrics  - Best Threshold: {test_t:.2f}, F1: {test_f1:.4f}, AUC: {test_auc:.4f}, Combined: {test_comb:.4f}, "
              f"Accuracy: {test_acc:.4f}, Precision: {test_prec:.4f}, Recall: {test_rec:.4f}")

        # Track best
        if test_comb > best_comb_test:
            best_comb_test = test_comb
            best_epoch = epoch + 1
            best_train_metrics = (train_t, train_f1, train_auc, train_comb, train_acc, train_prec, train_rec)
            best_test_metrics =  (test_t, test_f1, test_auc, test_comb, test_acc, test_prec, test_rec)
            print(f"*** New best metrics at epoch {best_epoch} ***")

        # Early stopping on loss
        if avg_loss < best_comb_test:
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= 3:
                print("Early stopping...")
                break

    # Final best
    print("\n=== Final Best Results ===")
    print(f"Best Epoch: {best_epoch}")
    bt, bf1, ba, bc, bap, bprec, br = best_train_metrics
    print(f"Train Metrics - Threshold: {bt:.2f}, F1: {bf1:.4f}, AUC: {ba:.4f}, Combined: {bc:.4f}, "
          f"Accuracy: {bap:.4f}, Precision: {bprec:.4f}, Recall: {br:.4f}")
    tt, tf1, ta, tc, tap, tprec, tr = best_test_metrics
    print(f"Test Metrics  - Threshold: {tt:.2f}, F1: {tf1:.4f}, AUC: {ta:.4f}, Combined: {tc:.4f}, "
          f"Accuracy: {tap:.4f}, Precision: {tprec:.4f}, Recall: {tr:.4f}")

# =======================================================
# 7. Khởi tạo mô hình và bắt đầu huấn luyện
# =======================================================
input_size = X_train_seq.shape[2]
hidden_size = 64
num_layers  = 2

model = FraudGRU(input_size, hidden_size, num_layers)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model.to(device)

criterion = CombinedLoss(alpha=0.25, gamma=2, bce_weight=1.0, focal_weight=1.0, reduction='mean')
optimizer = optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 50

train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device)