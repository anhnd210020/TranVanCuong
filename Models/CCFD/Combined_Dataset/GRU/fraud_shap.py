import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math 
import matplotlib.pyplot as plt
import shap
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

class CombinedLossUnc(nn.Module):
    """Uncertainty weighting (Kendall & Gal CVPR 2018)"""
    def __init__(self, alpha=0.25, gamma=2):
        super(CombinedLossUnc, self).__init__()
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

# Factorize các cột danh mục
df['merchant_num'] = pd.factorize(df['merchant'])[0]
df['last_num'] = pd.factorize(df['last'])[0]
df['street_num'] = pd.factorize(df['street'])[0]
df['city_num'] = pd.factorize(df['city'])[0]
df['zip_num'] = pd.factorize(df['zip'])[0]
df['state_num'] = pd.factorize(df['state'])[0]

df = pd.get_dummies(data=df, columns=['gender'], drop_first=True, dtype='int')

drop_cols = ['Unnamed: 0', 'trans_date_trans_time', 'merchant', 'first', 'last', 'street', 'city', 'state', 
             'lat', 'long', 'dob', 'unix_time', 'merch_lat', 'merch_long', 'city_pop']
df.drop(columns=drop_cols, errors='ignore', inplace=True)

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

if 'trans_num' in train.columns:
    train.drop('trans_num', axis=1, inplace=True)
if 'trans_num' in test.columns:
    test.drop('trans_num', axis=1, inplace=True)

y_train = train['is_fraud']
X_train = train.drop('is_fraud', axis=1)
y_test = test['is_fraud']
X_test = test.drop('is_fraud', axis=1)

print('Shape of training data:', (X_train.shape, y_train.shape))
print('Shape of testing data:', (X_test.shape, y_test.shape))

# Chuẩn hóa dữ liệu
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)
X_train_sc = pd.DataFrame(data=X_train_sc, columns=X_train.columns)
X_test_sc = pd.DataFrame(data=X_test_sc, columns=X_test.columns)

# Hàm tạo sequence theo phương pháp Transactional Expansion
def create_sequences_transactional_expansion(df, memory_size):
    sequences, labels = [], []
    grouped = df.groupby('cc_num')
    for user_id, group in grouped:
        group = group.sort_values(by='trans_date_trans_time_numeric')
        values = group.drop(columns=['is_fraud', 'cc_num']).values
        targets = group['is_fraud'].values
        n = len(group)
        for i in range(n):
            if i < memory_size:
                pad_needed = memory_size - (i + 1)
                pad = np.repeat(values[0:1, :], pad_needed, axis=0)
                seq = np.concatenate((pad, values[:i+1]), axis=0)
            else:
                seq = values[i - memory_size + 1: i + 1]
            sequences.append(seq)
            labels.append(targets[i])
    return np.array(sequences), np.array(labels)

train_seq_df = X_train_sc.copy()
train_seq_df['is_fraud'] = y_train.values
test_seq_df = X_test_sc.copy()
test_seq_df['is_fraud'] = y_test.values

memory_size = 50
X_train_seq, y_train_seq = create_sequences_transactional_expansion(train_seq_df, memory_size)
X_test_seq, y_test_seq = create_sequences_transactional_expansion(test_seq_df, memory_size)

print("Sequence shape (train):", X_train_seq.shape)
print("Sequence shape (test):", X_test_seq.shape)

# =======================================================
# 3. Định nghĩa Dataset cho PyTorch
# =======================================================
class FraudDataset(Dataset):
    def __init__(self, X, y):
        self.X = X  # (num_sequences, sequence_length, num_features)
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

# =======================================================
# 4. Định nghĩa mô hình FraudGRU với nhấn mạnh các features day_of_week và hour
# =======================================================
class FraudGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, day_index=None, hour_index=None, day_scale=2.0, hour_scale=2.0):
        super(FraudGRU, self).__init__()
        self.shift_gcn = Shift_gcn(in_channels=input_size, 
                                   out_channels=hidden_size, 
                                   num_nodes=1)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.day_index = day_index
        self.hour_index = hour_index
        self.day_scale = day_scale
        self.hour_scale = hour_scale

    def forward(self, x):
        x = x.unsqueeze(2)  # (batch, seq, 1, input_size)
        x = self.shift_gcn(x)
        x = x.squeeze(3).permute(0, 2, 1)  # (batch, seq, hidden_size)
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
            outputs = model(X_batch).squeeze().cpu().numpy()
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
    best_loss = float('inf')
    best_combined_metric_test = -float('inf')
    epochs_without_improvement = 0
    best_epoch = None
    best_train_metrics = None
    best_test_metrics = None

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for X_batch, y_batch in progress_bar:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        average_loss = total_loss / len(train_loader)
        print(f'\nEpoch {epoch+1}, Loss: {average_loss:.4f}')

        print(f"  log_sigma_bce:   {criterion.log_sigma_bce.item():.4f}")
        print(f"  log_sigma_focal: {criterion.log_sigma_focal.item():.4f}")
        
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

# =======================================================
# 7. Khởi tạo mô hình và bắt đầu huấn luyện
# =======================================================
input_size = X_train_seq.shape[2]  # Số lượng features (bao gồm các feature nhấn mạnh)
hidden_size = 64
num_layers = 2

# Xác định chỉ số của day_of_week và trans_hour trong DataFrame chuẩn hóa
day_index = X_train_sc.columns.get_loc('day_of_week')
hour_index = X_train_sc.columns.get_loc('trans_hour')

model = FraudGRU(input_size, hidden_size, num_layers,
                 day_index=day_index, hour_index=hour_index,
                 day_scale=2.0, hour_scale=2.0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = CombinedLossUnc(alpha=0.25, gamma=2).to(device)
optimizer = optim.Adam(
    list(model.parameters()) + list(criterion.parameters()),
    lr=1e-3
)
train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=1, device=device)
# ==================== AFTER TRAINING ====================
# Sau khi đã huấn luyện xong, tiến hành SHAP explainability
import copy

print("🔍 Starting SHAP explainability analysis...")

# 🚫 Disable cuDNN để tránh lỗi RNN backward
torch.backends.cudnn.enabled = False

# Bản sao model để tránh SHAP ảnh hưởng model gốc
model_shap = copy.deepcopy(model).to(device)
model_shap.eval()

# Lấy một batch nhỏ từ test_loader
X_batch, _ = next(iter(test_loader))
X_batch = X_batch.to(device)

# Dùng 100 mẫu làm background
background = X_batch[:100]

# Dùng DeepExplainer nếu ổn định, fallback nếu lỗi
try:
    explainer = shap.DeepExplainer(model_shap, background)
except Exception as e:
    print("⚠️ DeepExplainer failed, switching to GradientExplainer...", e)
    explainer = shap.GradientExplainer(model_shap, background)

# Chọn 50 mẫu để tính SHAP values
X_test_sample = X_batch[:50]

# Tính SHAP values (shape: [1, 50, T, F])
shap_values = explainer.shap_values(X_test_sample)

# Trung bình SHAP theo thời gian (dim=2), kết quả: (50, F)
shap_avg = shap_values[0].mean(axis=2)

# Tên các feature
feature_names = X_train_sc.columns.tolist()
X_mean = X_test_sample.mean(dim=1).cpu().numpy()

# Vẽ và lưu SHAP summary plot
shap.summary_plot(shap_avg, features=X_mean, feature_names=feature_names, show=False)

save_path = "/home/ducanh/Financial Risk & Fraud Detection/Credit Card Fraud Detection/Models/CCFD/Combined_Dataset/DL models/GRU/shap_summary.png"
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.close()

print(f"✅ SHAP summary plot saved to: {save_path}")
