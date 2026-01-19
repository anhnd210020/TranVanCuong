import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
from sklearn.preprocessing import StandardScaler

########################
# Utility: readable size
########################

def bytes_readable(num: int) -> str:
    """Convert bytes to human-readable string (KB/MB/GB)."""
    for unit in ["B", "KB", "MB", "GB", "TB", "PB"]:
        if num < 1024:
            return f"{num:.2f} {unit}"
        num /= 1024
    return f"{num:.2f} EB"

########################################################
# 1. Định nghĩa Shift_gcn & FraudGRU
########################################################

class Shift_gcn(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_nodes: int = 1):
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
        nn.init.normal_(self.Linear_weight, 0, np.sqrt(1.0 / out_channels))
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
        self.register_buffer("shift_in", torch.from_numpy(idx_in))
        idx_out = np.empty(num_nodes * out_channels, dtype=np.int64)
        for i in range(num_nodes):
            for j in range(out_channels):
                idx_out[i * out_channels + j] = (i * out_channels + j - j * out_channels) % (out_channels * num_nodes)
        self.register_buffer("shift_out", torch.from_numpy(idx_out))

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        x = x0.permute(0, 3, 1, 2).contiguous()
        B, C, T, V = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(B * T, V * C)
        x = torch.index_select(x, 1, self.shift_in).view(B * T, V, C)
        x = x * (torch.tanh(self.Feature_Mask) + 1)
        x = torch.einsum("nwc,cd->nwd", x, self.Linear_weight) + self.Linear_bias
        x = x.view(B * T, -1)
        x = torch.index_select(x, 1, self.shift_out)
        x = self.bn(x)
        x = x.view(B, T, V, self.out_channels).permute(0, 3, 1, 2)
        shortcut = self.down(x0.permute(0, 3, 1, 2))
        return self.relu(x + shortcut)

class FraudGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.shift_gcn = Shift_gcn(input_size, hidden_size, num_nodes=1)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(2)
        x = self.shift_gcn(x)
        x = x.squeeze(3).permute(0, 2, 1)
        out, _ = self.gru(x)
        return self.sigmoid(self.fc(out[:, -1, :]))

########################################################
# 2. Tiền xử lý & sinh sequence + log dung lượng
########################################################

@st.cache_data(show_spinner="Đang tiền xử lý và sinh sequence…")
def preprocess_and_sequences(df: pd.DataFrame, memory_size: int = 30):
    # Copy original DataFrame
    df_proc = df.copy()

    # 1. Chuyển đổi và tạo thông tin ngày giờ
    df_proc['trans_date_trans_time']         = pd.to_datetime(df_proc['trans_date_trans_time'])
    df_proc['trans_date_trans_time_numeric'] = df_proc['trans_date_trans_time'].apply(lambda x: x.timestamp())

    # --- Bổ sung ở đây: tạo và mã hóa day_of_week ---
    df_proc['day_of_week'] = df_proc['trans_date_trans_time'].dt.day_name()
    day_mapping = {
        'Monday':    0,
        'Tuesday':   1,
        'Wednesday': 2,
        'Thursday':  3,
        'Friday':    4,
        'Saturday':  5,
        'Sunday':    6
    }
    df_proc['day_of_week'] = df_proc['day_of_week'].map(day_mapping)
    df_proc['trans_hour'] = df_proc['trans_date_trans_time'].dt.hour

    # 2. Xử lý ngày sinh và tuổi
    df_proc['dob'] = pd.to_datetime(df_proc['dob'])
    df_proc['cust_age'] = df_proc['dob'].dt.year.apply(lambda x: 2021 - x)
    df_proc['cust_age_groups'] = df_proc['cust_age'].apply(
        lambda x: 'below 10' if x < 10 else 
                  ('10-20' if x < 20 else 
                   ('20-30' if x < 30 else 
                    ('30-40' if x < 40 else 
                     ('40-50' if x < 50 else 
                      ('50-60' if x < 60 else 
                       ('60-70' if x < 70 else 
                        ('70-80' if x < 80 else 'Above 80'))))))))
    

    # Mapping cho cust_age_groups theo mean fraud
    age_piv_2 = pd.pivot_table(
        data=df_proc,
        index='cust_age_groups',
        columns='is_fraud',
        values='amt',
        aggfunc=np.mean
    )
    age_piv_2.sort_values(by=1, ascending=True, inplace=True)
    age_dic = {k: v for (k, v) in zip(age_piv_2.index.values, age_piv_2.reset_index().index.values)}
    df_proc['cust_age_groups'] = df_proc['cust_age_groups'].map(age_dic)

    # Mapping cho category
    merch_cat = df_proc[df_proc['is_fraud'] == 1].groupby('category')['amt'].mean().sort_values()
    merch_cat_dic = {k: v for (k, v) in zip(merch_cat.index.values, merch_cat.reset_index().index.values)}
    df_proc['category'] = df_proc['category'].map(merch_cat_dic)

    # Mapping cho job
    job_txn_piv_2 = pd.pivot_table(
        data=df_proc,
        index='job',
        columns='is_fraud',
        values='amt',
        aggfunc=np.mean
    )
    job_cat_dic = {k: v for (k, v) in zip(job_txn_piv_2.index.values, job_txn_piv_2.reset_index().index.values)}
    df_proc['job'] = df_proc['job'].map(job_cat_dic)

    # Factorize các cột danh mục
    df_proc['merchant_num'] = pd.factorize(df_proc['merchant'])[0]
    df_proc['last_num']     = pd.factorize(df_proc['last'])[0]
    df_proc['street_num']   = pd.factorize(df_proc['street'])[0]
    df_proc['city_num']     = pd.factorize(df_proc['city'])[0]
    df_proc['zip_num']      = pd.factorize(df_proc['zip'])[0]
    df_proc['state_num']    = pd.factorize(df_proc['state'])[0]

    # One-hot gender
    df_proc = pd.get_dummies(data=df_proc, columns=['gender'], drop_first=True, dtype='int')

    # 3. Drop các cột không dùng
    drop_cols = [
        'Unnamed: 0', 'trans_date_trans_time', 'merchant', 'first', 'last',
        'street', 'city', 'state', 'lat', 'long', 'dob', 'unix_time',
        'merch_lat', 'merch_long', 'city_pop', 'trans_num'
    ]
    df_proc.drop(columns=drop_cols, errors='ignore', inplace=True)

    # Lấy feature_cols và scale
    feature_cols = [c for c in df_proc.columns if c not in ['cc_num', 'is_fraud']]
    numeric_df = df_proc[feature_cols].astype(float)
    scaler = StandardScaler().fit(numeric_df)
    scaled = scaler.transform(numeric_df).astype(np.float32)

    # Log dung lượng
    raw_bytes = df.memory_usage(deep=True).sum()
    proc_bytes = df_proc.memory_usage(deep=True).sum()
    st.info(f"CSV gốc: {bytes_readable(raw_bytes)}, Xử lý: {bytes_readable(proc_bytes)}")

    # Sinh sequence
    sequences, idxs = [], []
    for _, grp in df_proc.sort_values('trans_date_trans_time_numeric').groupby('cc_num'):
        vals = scaled[grp.index]
        for i in range(len(vals)):
            start = max(0, i-memory_size+1)
            seq = vals[start:i+1]
            if len(seq) < memory_size:
                pad = np.repeat(seq[:1], memory_size-len(seq), axis=0)
                seq = np.vstack([pad, seq])
            sequences.append(seq)
            idxs.append(grp.index[i])
    sequences = np.stack(sequences)
    st.info(f"Sequences: shape={sequences.shape}, mem={bytes_readable(sequences.nbytes)}")
    st.write("### Features sử dụng:")
    for i, feat in enumerate(feature_cols, 1):
        st.write(f"{i}. {feat}")
    return df_proc, sequences, idxs, feature_cols

########################################################
# 3. Load model & UI
########################################################

@st.cache_resource(show_spinner="Đang tải mô hình…")
def load_model(model_path: str, input_size: int):
    model = FraudGRU(input_size, hidden_size=64, num_layers=2)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

# Config paths
DATA_PATH = "/home/ducanh/Financial Risk & Fraud Detection/Credit Card Fraud Detection/Datasets/CCFD/Combined_Data/combined_data.csv"
MODEL_PATH = "/home/ducanh/Financial Risk & Fraud Detection/Credit Card Fraud Detection/Models/CCFD/Combined_Dataset/DL models/GRU/fraudgru.pth"

st.title("Credit Card Fraud Detection App")
if not os.path.exists(DATA_PATH):
    st.error(f"Không tìm thấy dữ liệu tại {DATA_PATH}")
else:
    df = pd.read_csv(DATA_PATH)
    st.write(f"Dữ liệu: {df.shape[0]} dòng x {df.shape[1]} cột")
    df_proc, sequences, idxs, feature_cols = preprocess_and_sequences(df, memory_size=30)
    model = load_model(MODEL_PATH, sequences.shape[2])
    # predict batch
    bs = 1024
    probs=[]
    for i in range(0, len(sequences), bs):
        batch = torch.tensor(sequences[i:i+bs], dtype=torch.float32)
        with torch.no_grad(): probs += model(batch).squeeze().tolist()
    df_proc['fraud_prob']=np.nan
    df_proc.loc[idxs,'fraud_prob']=probs
    thresh=st.slider("Threshold",0.0,1.0,0.5)
    df_proc['pred']= (df_proc['fraud_prob']>thresh).astype(int)
    st.subheader("Kết quả mẫu")
    st.dataframe(df_proc.loc[idxs,['cc_num','fraud_prob','pred']].head())
    csv=df_proc.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV",csv,"pred.csv")
