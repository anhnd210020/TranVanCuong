import os
import numpy as np
import pandas as pd
import torch

# =======================================================
# CONFIG
# =======================================================
CSV_PATH = r"TranVanCuong/combined_data.csv"
SAVE_PATH = r"seq_cache.pt"
MEMORY_SIZE = 2000

np.random.seed(42)

# =======================================================
# 1) LOAD CSV
# =======================================================
print("Loading CSV...")
df = pd.read_csv(CSV_PATH)

df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['trans_date_trans_time_numeric'] = df['trans_date_trans_time'].apply(lambda x: x.timestamp())

# --- day_of_week ---
df['day_of_week'] = df['trans_date_trans_time'].dt.day_name()
day_mapping = {
    'Monday':    0,
    'Tuesday':   1,
    'Wednesday': 2,
    'Thursday':  3,
    'Friday':    4,
    'Saturday':  5,
    'Sunday':    6
}
df['day_of_week'] = df['day_of_week'].map(day_mapping)

# --- trans_hour (GIỮ LOGIC CỦA BẠN, NHƯNG ÉP SỐ) ---
df['trans_hour'] = df['trans_date_trans_time'].dt.time.apply(lambda x: str(x)[:2]).astype(np.int16)

# =======================================================
# 2) AGE + AGE GROUPS
# =======================================================
df['dob'] = pd.to_datetime(df['dob'])
df['cust_age'] = df['dob'].dt.year.apply(lambda x: 2021 - x)

df['cust_age_groups'] = df['cust_age'].apply(
    lambda x: 'below 10' if x < 10 else
    ('10-20' if x >= 10 and x < 20 else
     ('20-30' if x >= 20 and x < 30 else
      ('30-40' if x >= 30 and x < 40 else
       ('40-50' if x >= 40 and x < 50 else
        ('50-60' if x >= 50 and x < 60 else
         ('60-70' if x >= 60 and x < 70 else
          ('70-80' if x >= 70 and x < 80 else 'Above 80'))))))))

age_piv_2 = pd.pivot_table(
    data=df,
    index='cust_age_groups',
    columns='is_fraud',
    values='amt',
    aggfunc=np.mean
)

# nếu thiếu fraud column 1 thì tránh lỗi
if 1 not in age_piv_2.columns:
    age_piv_2[1] = 0.0

age_piv_2.sort_values(by=1, ascending=True, inplace=True)
age_dic = {k: v for (k, v) in zip(age_piv_2.index.values, age_piv_2.reset_index().index.values)}
df['cust_age_groups'] = df['cust_age_groups'].map(age_dic)

# =======================================================
# 3) MAPPING category/job
# =======================================================
merch_cat = df[df['is_fraud'] == 1].groupby('category')['amt'].mean().sort_values(ascending=True)
merch_cat_dic = {k: v for (k, v) in zip(merch_cat.index.values, merch_cat.reset_index().index.values)}
df['category'] = df['category'].map(merch_cat_dic)

job_txn_piv_2 = pd.pivot_table(
    data=df,
    index='job',
    columns='is_fraud',
    values='amt',
    aggfunc=np.mean
)

job_cat_dic = {k: v for (k, v) in zip(job_txn_piv_2.index.values, job_txn_piv_2.reset_index().index.values)}
df['job'] = df['job'].map(job_cat_dic)

# =======================================================
# 4) FACTORIZE
# =======================================================
df['merchant_num'] = pd.factorize(df['merchant'])[0]
df['last_num']     = pd.factorize(df['last'])[0]
df['street_num']   = pd.factorize(df['street'])[0]
df['city_num']     = pd.factorize(df['city'])[0]
df['zip_num']      = pd.factorize(df['zip'])[0]
df['state_num']    = pd.factorize(df['state'])[0]

# one-hot gender
df = pd.get_dummies(data=df, columns=['gender'], drop_first=True, dtype='int')

# drop cols
drop_cols = [
    'Unnamed: 0', 'trans_date_trans_time', 'merchant', 'first', 'last', 'street', 'city', 'state',
    'lat', 'long', 'dob', 'unix_time', 'merch_lat', 'merch_long', 'city_pop', 'trans_num'
]
df.drop(columns=drop_cols, errors='ignore', inplace=True)

# fill NaN để đảm bảo numeric
df = df.fillna(-1)

# =======================================================
# 5) SPLIT BY cc_num (999 users)
# =======================================================
unique_cc_nums = df['cc_num'].unique()
assert len(unique_cc_nums) == 999, "Số lượng user không đạt 999, kiểm tra lại dữ liệu."

np.random.shuffle(unique_cc_nums)
train_cc_nums = unique_cc_nums[:800]
test_cc_nums  = unique_cc_nums[800:]

train = df[df['cc_num'].isin(train_cc_nums)].copy()
test  = df[df['cc_num'].isin(test_cc_nums)].copy()

print("Train shape:", train.shape)
print("Test shape :", test.shape)

# =======================================================
# 6) DEFINE FEATURES + STANDARDIZE (CPU mean/std)
# =======================================================
y_train = train["is_fraud"].astype(np.float32)
X_train = train.drop("is_fraud", axis=1)
y_test  = test["is_fraud"].astype(np.float32)
X_test  = test.drop("is_fraud", axis=1)

feature_cols = [c for c in X_train.columns if c != "cc_num"]
print(f"Total features used: {len(feature_cols)}")

# mean/std từ TRAIN
Xtr_np = X_train[feature_cols].to_numpy(np.float32, copy=True)
mean = Xtr_np.mean(axis=0, keepdims=True)
std  = Xtr_np.std(axis=0, keepdims=True)
std[std < 1e-6] = 1e-6

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# =======================================================
# 7) BUILD PER-USER ARRAYS (CPU)  + INDEX MAP
# =======================================================
def build_user_tensors(df_raw, feature_cols, mean, std, x_dtype=np.float32):
    """
    Return:
      - user_ids: list of cc_num
      - X_list: list of torch.Tensor [n_i, F] (scaled)
      - y_list: list of torch.Tensor [n_i]
      - idx_user: np.int32 array length N_total (maps global idx -> user_index)
      - idx_pos : np.int32 array length N_total (maps global idx -> position in that user)
    """
    user_ids = []
    X_list = []
    y_list = []

    idx_user = []
    idx_pos = []

    grouped = df_raw.groupby("cc_num", sort=True)
    for u_idx, (cc, group) in enumerate(grouped):
        group = group.sort_values(by="trans_date_trans_time_numeric")

        X = group[feature_cols].to_numpy(np.float32, copy=True)
        y = group["is_fraud"].to_numpy(np.float32, copy=True)

        # scale
        X = (X - mean) / std

        if x_dtype == np.float16:
            X = X.astype(np.float16, copy=False)

        Xt = torch.from_numpy(X)              # [n_i, F]
        yt = torch.from_numpy(y)              # [n_i]

        user_ids.append(cc)
        X_list.append(Xt)
        y_list.append(yt)

        n = len(group)
        idx_user.extend([u_idx] * n)
        idx_pos.extend(list(range(n)))

    idx_user = np.asarray(idx_user, dtype=np.int32)
    idx_pos  = np.asarray(idx_pos, dtype=np.int32)
    return user_ids, X_list, y_list, idx_user, idx_pos


class WindowDataset(Dataset):
    def __init__(self, X_list, y_list, idx_user, idx_pos, memory_size, pad_mode="repeat_first"):
        self.X_list = X_list
        self.y_list = y_list
        self.idx_user = idx_user
        self.idx_pos = idx_pos
        self.M = memory_size
        assert pad_mode in ["repeat_first", "zeros"]
        self.pad_mode = pad_mode

        # feature dim
        self.F = X_list[0].shape[1]

    def __len__(self):
        return len(self.idx_user)

    def __getitem__(self, i):
        u = int(self.idx_user[i])
        t = int(self.idx_pos[i])

        X_u = self.X_list[u]     # [n_i, F]
        y_u = self.y_list[u]     # [n_i]

        start = t - self.M + 1
        if start >= 0:
            seq = X_u[start:t+1]  # [M, F]
        else:
            # need pad
            need = -start
            if self.pad_mode == "repeat_first":
                pad_row = X_u[0:1].expand(need, -1)  # [need, F]
            else:
                pad_row = torch.zeros((need, self.F), dtype=X_u.dtype)

            seq = torch.cat([pad_row, X_u[0:t+1]], dim=0)  # [M, F]

        label = y_u[t]
        # return float32 for model stability
        return seq.to(torch.float32), label.to(torch.float32)


print("Building per-user train tensors...")
train_user_ids, X_train_users, y_train_users, train_idx_user, train_idx_pos = build_user_tensors(
    train, feature_cols, mean, std, x_dtype=np.float16  # <--- dùng float16 để giảm RAM (tùy bạn)
)

print("Building per-user test tensors...")
test_user_ids, X_test_users, y_test_users, test_idx_user, test_idx_pos = build_user_tensors(
    test, feature_cols, mean, std, x_dtype=np.float16
)

train_ds = WindowDataset(X_train_users, y_train_users, train_idx_user, train_idx_pos,
                         memory_size=MEMORY_SIZE, pad_mode="repeat_first")
test_ds  = WindowDataset(X_test_users,  y_test_users,  test_idx_user,  test_idx_pos,
                         memory_size=MEMORY_SIZE, pad_mode="repeat_first")

print("Total train samples:", len(train_ds))
print("Total test samples :", len(test_ds))

# =======================================================
# 8) SAVE LIGHT CACHE (NO BIG (N,M,F))
# =======================================================
print(f"Saving lightweight cache to {SAVE_PATH} ...")
cache = {
    "memory_size": MEMORY_SIZE,
    "feature_cols": feature_cols,
    "mean": torch.tensor(mean, dtype=torch.float32),
    "std": torch.tensor(std, dtype=torch.float32),

    # store per-user tensors (much smaller than full sequences)
    "X_train_users": X_train_users,
    "y_train_users": y_train_users,
    "train_idx_user": torch.from_numpy(train_idx_user),
    "train_idx_pos": torch.from_numpy(train_idx_pos),

    "X_test_users": X_test_users,
    "y_test_users": y_test_users,
    "test_idx_user": torch.from_numpy(test_idx_user),
    "test_idx_pos": torch.from_numpy(test_idx_pos),
}
torch.save(cache, SAVE_PATH)

import os
file_size_mb = os.path.getsize(SAVE_PATH) / (1024 * 1024)
print(f"Saved: {SAVE_PATH} ({file_size_mb:.2f} MB)")
print("Done.")

