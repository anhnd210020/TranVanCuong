import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
from sklearn.preprocessing import StandardScaler
from st_aggrid import AgGrid, GridOptionsBuilder

# ---------- Utility: Đổi dung lượng bytes thành dạng dễ đọc ----------
def bytes_readable(num: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB", "PB"]:
        if num < 1024:
            return f"{num:.2f} {unit}"
        num /= 1024
    return f"{num:.2f} EB"

# ---------- Mô hình Shift-GCN ----------
class Shift_gcn(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_nodes: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_nodes = num_nodes
        self.down = (
            nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1), nn.BatchNorm2d(out_channels))
            if in_channels != out_channels else lambda x: x
        )
        self.Linear_weight = nn.Parameter(torch.zeros(in_channels, out_channels))
        nn.init.normal_(self.Linear_weight, 0, np.sqrt(1.0 / out_channels))
        self.Linear_bias = nn.Parameter(torch.zeros(1, 1, out_channels))
        self.Feature_Mask = nn.Parameter(torch.zeros(1, num_nodes, in_channels))
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

    def forward(self, x0):
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

# ---------- Mô hình FraudGRU ----------
class FraudGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.shift_gcn = Shift_gcn(input_size, hidden_size, num_nodes=1)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(2)
        x = self.shift_gcn(x)
        x = x.squeeze(3).permute(0, 2, 1)
        out, _ = self.gru(x)
        return self.sigmoid(self.fc(out[:, -1, :]))

# ---------- Tiền xử lý & Sinh sequence ----------
@st.cache_data(show_spinner="Đang tiền xử lý và sinh sequence…")
def preprocess_and_sequences(df: pd.DataFrame, memory_size: int = 30):
    df_proc = df.copy()
    df_proc['trans_date_trans_time'] = pd.to_datetime(df_proc['trans_date_trans_time'])
    df_proc['trans_date_trans_time_numeric'] = df_proc['trans_date_trans_time'].apply(lambda x: x.timestamp())
    df_proc['day_of_week'] = df_proc['trans_date_trans_time'].dt.day_name().map({
        'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
        'Friday': 4, 'Saturday': 5, 'Sunday': 6
    })
    df_proc['trans_hour'] = df_proc['trans_date_trans_time'].dt.hour
    df_proc['dob'] = pd.to_datetime(df_proc['dob'])
    df_proc['cust_age'] = df_proc['dob'].dt.year.apply(lambda x: 2021 - x)
    df_proc['cust_age_groups'] = df_proc['cust_age'].apply(lambda x: 'below 10' if x < 10 else (
        '10-20' if x < 20 else ('20-30' if x < 30 else ('30-40' if x < 40 else (
        '40-50' if x < 50 else ('50-60' if x < 60 else ('60-70' if x < 70 else (
        '70-80' if x < 80 else 'Above 80'))))))))
    age_piv = pd.pivot_table(df_proc, index='cust_age_groups', columns='is_fraud', values='amt', aggfunc=np.mean).sort_values(by=1)
    df_proc['cust_age_groups'] = df_proc['cust_age_groups'].map({k:i for i,k in enumerate(age_piv.index)})
    merch_map = {k:i for i,k in enumerate(df_proc[df_proc['is_fraud']==1].groupby('category')['amt'].mean().sort_values().index)}
    df_proc['category'] = df_proc['category'].map(merch_map)
    job_map = {k:i for i,k in enumerate(df_proc.groupby('job')['amt'].mean().sort_values().index)}
    df_proc['job'] = df_proc['job'].map(job_map)
    for col in ['merchant','last','street','city','zip','state']:
        df_proc[f'{col}_num'] = pd.factorize(df_proc[col])[0]
    df_proc = pd.get_dummies(df_proc, columns=['gender'], drop_first=True)
    df_proc.drop(columns=[
        'Unnamed: 0','trans_date_trans_time','merchant','first','last','street','city','state',
        'lat','long','dob','unix_time','merch_lat','merch_long','city_pop','trans_num'
    ], errors='ignore', inplace=True)

    feature_cols = [c for c in df_proc.columns if c not in ['cc_num','is_fraud']]
    scaler = StandardScaler().fit(df_proc[feature_cols].astype(float))
    scaled = scaler.transform(df_proc[feature_cols]).astype(np.float32)

    st.info(f"CSV gốc: {bytes_readable(df.memory_usage(deep=True).sum())}, Sau xử lý: {bytes_readable(df_proc.memory_usage(deep=True).sum())}")

    sequences, idxs = [], []
    for _, grp in df_proc.sort_values('trans_date_trans_time_numeric').groupby('cc_num'):
        vals = scaled[grp.index]
        for i in range(len(vals)):
            pad_len = max(0, 30 - (i + 1))
            pad = np.repeat(vals[0:1], pad_len, axis=0)
            seq = np.vstack([pad, vals[:i + 1]]) if pad_len else vals[i + 1 - 30:i + 1]
            sequences.append(seq)
            idxs.append(grp.index[i])
    sequences = np.stack(sequences)
    st.info(f"Kích thước sequences: {sequences.shape}, RAM: {bytes_readable(sequences.nbytes)}")
    st.markdown("### Features sử dụng:")
    cols = st.columns(3)
    for i, f in enumerate(feature_cols):
        cols[i % 3].markdown(f"- {f}")
    return df_proc, sequences, idxs, feature_cols

# ---------- Load model ----------
@st.cache_resource(show_spinner="Đang tải mô hình…")
def load_model(model_path: str, input_size: int):
    model = FraudGRU(input_size)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

# ---------- Cấu hình đường dẫn ----------
DATA_PATH = "/home/ducanh/Financial Risk & Fraud Detection/Credit Card Fraud Detection/Datasets/CCFD/Combined_Data/combined_data.csv"
MODEL_PATH = "/home/ducanh/Financial Risk & Fraud Detection/Credit Card Fraud Detection/Models/CCFD/Combined_Dataset/DL models/GRU/fraudgru.pth"

# ---------- Giao diện chính ----------
st.title("Credit Card Fraud Detection App")

if not os.path.exists(DATA_PATH):
    st.error(f"Không tìm thấy file dữ liệu tại {DATA_PATH}")
else:
    df = pd.read_csv(DATA_PATH)
    st.write(f"Dữ liệu gồm {df.shape[0]} dòng x {df.shape[1]} cột")

    df_proc, sequences, idxs, feature_cols = preprocess_and_sequences(df, memory_size=30)
    model = load_model(MODEL_PATH, sequences.shape[2])

    # Dự đoán theo batch
    probs = []
    bs = 1024
    for i in range(0, len(sequences), bs):
        batch = torch.tensor(sequences[i:i+bs], dtype=torch.float32)
        with torch.no_grad():
            probs += model(batch).squeeze().tolist()

    df_proc['fraud_prob'] = np.nan
    df_proc.loc[idxs, 'fraud_prob'] = probs
    thresh = 0.5
    df_proc['pred'] = (df_proc['fraud_prob'] > thresh).astype(int)
    num_fraud = df_proc.loc[idxs, 'pred'].sum()
    st.success(f"Số giao dịch bị đánh dấu là gian lận: {num_fraud} / {len(idxs)}")

    st.subheader("Kết quả mẫu")
    display_cols = ['cc_num', 'fraud_prob', 'pred']
    sample_df = df_proc.loc[idxs, display_cols].copy()
    gb = GridOptionsBuilder.from_dataframe(sample_df)
    gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=15)
    gb.configure_default_column(filter=True, sortable=True, resizable=True)
    gb.configure_column("fraud_prob", type=["numericColumn", "numberColumnFilter", "customNumericFormat"], precision=4)
    gridOptions = gb.build()
    AgGrid(sample_df, gridOptions=gridOptions, theme="streamlit", height=400, fit_columns_on_grid_load=True)
    csv = df_proc.to_csv(index=False).encode('utf-8')
    st.download_button("Tải CSV kết quả", csv, "fraud_result.csv")
