import random
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

# =======================================================
# 0. Loss functions
# =======================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='none'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: after sigmoid, shape (batch,)
        # targets: shape (batch,)
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

# =======================================================
# 1. Shift‐GCN block
# =======================================================
class Shift_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes=1):
        super(Shift_gcn, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_nodes = num_nodes

        # down‐projection if needed for residual
        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        # learnable linear mapping
        self.Linear_weight = nn.Parameter(torch.zeros(in_channels, out_channels))
        nn.init.normal_(self.Linear_weight, 0, math.sqrt(1.0 / out_channels))
        self.Linear_bias = nn.Parameter(torch.zeros(1, 1, out_channels))
        self.Feature_Mask = nn.Parameter(torch.zeros(1, num_nodes, in_channels))

        self.bn = nn.BatchNorm1d(num_nodes * out_channels)
        self.relu = nn.ReLU()

        # prepare shift indices
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

    def forward(self, x0, edge_index=None):
        # x0: (B, T, V, C_in)
        x = x0.permute(0, 3, 1, 2).contiguous()        # (B, C_in, T, V)
        B, C, T, V = x.shape

        # shift in
        x = x.permute(0, 2, 3, 1).contiguous().view(B * T, V * C)
        x = torch.index_select(x, 1, self.shift_in).view(B * T, V, C)

        # feature gating + linear
        x = x * (torch.tanh(self.Feature_Mask) + 1)
        x = torch.einsum('ntc,cd->ntd', x, self.Linear_weight) + self.Linear_bias

        # shift out + BN + reshape
        x = x.view(B * T, -1)
        x = torch.index_select(x, 1, self.shift_out)
        x = self.bn(x)
        x = x.view(B, T, V, self.out_channels).permute(0, 3, 1, 2).contiguous()

        # residual & activate
        res = self.down(x0.permute(0, 3, 1, 2))
        x = x + res
        return self.relu(x)

# =======================================================
# 2. Data preprocessing & sequence creation
# =======================================================
df = pd.read_csv('/home/ducanh/Financial Risk & Fraud Detection/Credit Card Fraud Detection/Datasets/CCFD/Combined_Data/combined_data.csv')

# timestamps & hour
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['trans_date_trans_time_numeric'] = df['trans_date_trans_time'].map(pd.Timestamp.timestamp)
df['trans_hour'] = df['trans_date_trans_time'].dt.hour.astype(float)

# age & buckets
df['dob'] = pd.to_datetime(df['dob'])
df['cust_age'] = 2021 - df['dob'].dt.year
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

# Factorize các cột danh mục
df['merchant_num'] = pd.factorize(df['merchant'])[0]
df['last_num'] = pd.factorize(df['last'])[0]
df['street_num'] = pd.factorize(df['street'])[0]
df['city_num'] = pd.factorize(df['city'])[0]
df['zip_num'] = pd.factorize(df['zip'])[0]
df['state_num'] = pd.factorize(df['state'])[0]

# One-hot encoding cho giới tính
df = pd.get_dummies(data=df, columns=['gender'], drop_first=True, dtype='int')

# Drop các cột không cần thiết
drop_cols = ['Unnamed: 0', 'trans_date_trans_time', 'merchant', 'first', 'last', 'street', 'city', 'state', 
             'lat', 'long', 'dob', 'unix_time', 'merch_lat', 'merch_long', 'city_pop', 'trans_num']
df.drop(columns=drop_cols, errors='ignore', inplace=True)


# split by cc_num
unique_cc = df['cc_num'].unique()
np.random.seed(42)
np.random.shuffle(unique_cc)
train_cc, test_cc = unique_cc[:800], unique_cc[800:]
train_df = df[df['cc_num'].isin(train_cc)].copy()
test_df  = df[df['cc_num'].isin(test_cc)].copy()

# separate label and features
y_train = train_df['is_fraud'].values
X_train = train_df.drop(['is_fraud','cc_num'], axis=1)
y_test  = test_df['is_fraud'].values
X_test  = test_df.drop(['is_fraud','cc_num'], axis=1)

# scale
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# create transactional‐expansion sequences
def create_sequences(df_scaled, y, memory_size):
    seqs, labs = [], []
    df_scaled = pd.DataFrame(df_scaled, columns=X_train.columns)
    df_scaled['is_fraud'] = y
    df_scaled['cc_num'] = np.concatenate([train_df['cc_num'].values, test_df['cc_num'].values])[:len(y)]
    for cc, group in df_scaled.groupby('cc_num'):
        vals = group.sort_values('trans_date_trans_time_numeric').drop(['is_fraud','cc_num'],axis=1).values
        labs_group = group['is_fraud'].values
        for i in range(len(group)):
            if i < memory_size:
                pad = np.repeat(vals[:1], memory_size - (i+1), axis=0)
                seq = np.vstack([pad, vals[:i+1]])
            else:
                seq = vals[i-memory_size+1:i+1]
            seqs.append(seq)
            labs.append(labs_group[i])
    return np.array(seqs), np.array(labs)

memory_size = 50
X_train_seq, y_train_seq = create_sequences(X_train_sc, y_train, memory_size)
X_test_seq,  y_test_seq  = create_sequences(X_test_sc,  y_test,  memory_size)

# =======================================================
# 3. Dataset classes & DataLoaders
# =======================================================
class FraudDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class FraudTripletDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        # index by label
        self.by_label = {0: [], 1: []}
        for i, lbl in enumerate(y):
            self.by_label[int(lbl)].append(i)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        anchor = self.X[idx]
        lbl = int(self.y[idx])
        # positive
        pos_idx = idx
        while pos_idx == idx:
            pos_idx = random.choice(self.by_label[lbl])
        # negative
        neg_lbl = 1 - lbl
        neg_idx = random.choice(self.by_label[neg_lbl])
        return (
            torch.tensor(anchor, dtype=torch.float32),
            torch.tensor(self.X[pos_idx], dtype=torch.float32),
            torch.tensor(self.X[neg_idx], dtype=torch.float32)
        ), torch.tensor(lbl, dtype=torch.float32)

batch_size = 64
# for evaluation
train_eval_loader = DataLoader(FraudDataset(X_train_seq, y_train_seq),
                               batch_size=batch_size, shuffle=False)
test_loader       = DataLoader(FraudDataset(X_test_seq,  y_test_seq),
                               batch_size=batch_size, shuffle=False)
# for training with triplets
train_triplet_loader = DataLoader(FraudTripletDataset(X_train_seq, y_train_seq),
                                  batch_size=batch_size, shuffle=True)

# =======================================================
# 4. FraudGRU model (returns prob & embedding)
# =======================================================
class FraudGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 day_index=None, hour_index=None,
                 day_scale=2.0, hour_scale=2.0):
        super(FraudGRU, self).__init__()
        self.shift_gcn = Shift_gcn(in_channels=input_size,
                                   out_channels=hidden_size,
                                   num_nodes=1)
        self.gru = nn.GRU(hidden_size, hidden_size,
                          num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.day_index = day_index
        self.hour_index = hour_index
        self.day_scale = day_scale
        self.hour_scale = hour_scale

    def forward(self, x):
        # x: (B, seq, D)
        if self.day_index is not None:
            x[:, :, self.day_index] *= self.day_scale
        if self.hour_index is not None:
            x[:, :, self.hour_index] *= self.hour_scale

        x = x.unsqueeze(2)                     # (B, seq, 1, D)
        x = self.shift_gcn(x)                  # (B, H, seq, 1)
        x = x.squeeze(3).permute(0, 2, 1)      # (B, seq, H)
        out, _ = self.gru(x)                   # (B, seq, H)
        h_last = out[:, -1, :]                 # (B, H)
        logit = self.fc(h_last)                # (B, 1)
        prob  = self.sigmoid(logit)            # (B, 1)
        return prob.squeeze(1), h_last         # -> (B,), (B, H)

# =======================================================
# 5. Evaluation
# =======================================================
def evaluate_model(loader, model, device):
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            probs, _ = model(X_batch)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(y_batch.numpy())
    all_probs  = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    auc = roc_auc_score(all_labels, all_probs)
    best_f1, best_thr = 0, 0.5
    for t in np.linspace(0.1, 0.9, 9):
        preds = (all_probs > t).astype(int)
        f1 = f1_score(all_labels, preds)
        if f1 > best_f1:
            best_f1, best_thr = f1, t

    combined = (best_f1 + auc) / 2
    preds = (all_probs > best_thr).astype(int)
    cm = confusion_matrix(all_labels, preds)
    TN, FP, FN, TP = cm.ravel() if cm.size == 4 else (0,0,0,0)
    acc  = (TP + TN) / (TP + TN + FP + FN)
    prec = TP / (TP + FP) if (TP + FP)>0 else 0
    rec  = TP / (TP + FN) if (TP + FN)>0 else 0

    return best_thr, best_f1, auc, combined, acc, prec, rec

# =======================================================
# 6. Training with contrastive loss
# =======================================================
def train_model(model,
                train_loader,
                eval_loader,
                test_loader,
                cls_criterion,
                triplet_criterion,
                optimizer,
                num_epochs,
                device,
                contrastive_weight=0.5):
    best_test_combined = -float('inf')
    for epoch in range(1, num_epochs+1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", leave=False)
        for (anc, pos, neg), y in pbar:
            anc, pos, neg, y = anc.to(device), pos.to(device), neg.to(device), y.to(device)
            optimizer.zero_grad()
            prob_a, emb_a = model(anc)
            _,      emb_p = model(pos)
            _,      emb_n = model(neg)

            cls_loss = cls_criterion(prob_a, y)
            cont_loss= triplet_criterion(emb_a, emb_p, emb_n)
            loss = cls_loss + contrastive_weight * cont_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch} — Avg Loss: {avg_loss:.4f}")

        # eval on training set
        tr_thr, tr_f1, tr_auc, tr_comb, tr_acc, tr_prec, tr_rec = \
            evaluate_model(eval_loader, model, device)
        print(f" Train  — Thr {tr_thr:.2f}, F1 {tr_f1:.4f}, AUC {tr_auc:.4f}, "
              f"Comb {tr_comb:.4f}, Acc {tr_acc:.4f}, Prec {tr_prec:.4f}, Rec {tr_rec:.4f}")

        # eval on test set
        te_thr, te_f1, te_auc, te_comb, te_acc, te_prec, te_rec = \
            evaluate_model(test_loader, model, device)
        print(f" Test   — Thr {te_thr:.2f}, F1 {te_f1:.4f}, AUC {te_auc:.4f}, "
              f"Comb {te_comb:.4f}, Acc {te_acc:.4f}, Prec {te_prec:.4f}, Rec {te_rec:.4f}")

        if te_comb > best_test_combined:
            best_test_combined = te_comb
            print(f"*** New best Test combined metric: {te_comb:.4f} at epoch {epoch} ***")

# =======================================================
# 7. Instantiate & run
# =======================================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

input_size  = X_train_seq.shape[2]
hidden_size = 64
num_layers  = 2
day_idx     = X_train.columns.get_loc('day_of_week')
hour_idx    = X_train.columns.get_loc('trans_hour')

model = FraudGRU(input_size, hidden_size, num_layers,
                 day_index=day_idx, hour_index=hour_idx,
                 day_scale=2.0, hour_scale=2.0)
model.to(device)

cls_criterion     = CombinedLoss(alpha=0.25, gamma=2,
                                 bce_weight=1.0, focal_weight=1.0,
                                 reduction='mean')
triplet_criterion = nn.TripletMarginLoss(margin=1.0, p=2)
optimizer         = optim.Adam(model.parameters(), lr=1e-3)
num_epochs        = 50
contrastive_wt    = 0.5  # tune this

train_model(model,
            train_triplet_loader,
            train_eval_loader,
            test_loader,
            cls_criterion,
            triplet_criterion,
            optimizer,
            num_epochs,
            device,
            contrastive_weight=contrastive_wt)
