import os
import numpy as np
import pandas as pd
import torch

# =======================================================
# CONFIG
# =======================================================
CSV_PATH = r"/home/ducanhhh/Fraud-detection-in-credit-card/Credit Card Fraud Detection/Datasets/combined_data.csv"
SAVE_PATH = r"seq_cache.pt"
MEMORY_SIZE = 100

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

# =======================================================
# 7) CREATE SEQUENCES (CPU)
# =======================================================
def create_sequences_transactional_expansion_cpu(df_raw, feature_cols, memory_size, mean, std):
    sequences = []
    labels = []

    grouped = df_raw.groupby("cc_num")
    for _, group in grouped:
        group = group.sort_values(by="trans_date_trans_time_numeric")

        X = group[feature_cols].to_numpy(np.float32, copy=True)
        y = group["is_fraud"].to_numpy(np.float32, copy=True)

        # scale CPU
        X = (X - mean) / std

        n = len(group)
        for i in range(n):
            if i < memory_size:
                pad_needed = memory_size - (i + 1)
                pad = np.repeat(X[0:1], pad_needed, axis=0)
                seq = np.concatenate([pad, X[:i+1]], axis=0)
            else:
                seq = X[i - memory_size + 1: i + 1]

            sequences.append(seq)
            labels.append(y[i])

    return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.float32)

print("Building train sequences...")
X_train_seq, y_train_seq = create_sequences_transactional_expansion_cpu(train, feature_cols, MEMORY_SIZE, mean, std)
print("Building test sequences...")
X_test_seq,  y_test_seq  = create_sequences_transactional_expansion_cpu(test, feature_cols, MEMORY_SIZE, mean, std)

print("Train seq shape:", X_train_seq.shape)
print("Test  seq shape:", X_test_seq.shape)

# =======================================================
# 8) SAVE TO .pt
# =======================================================
print(f"Saving to {SAVE_PATH} ...")

cache = {
    "memory_size": MEMORY_SIZE,
    "feature_cols": feature_cols,
    "mean": torch.tensor(mean, dtype=torch.float32),
    "std": torch.tensor(std, dtype=torch.float32),

    "X_train_seq": torch.tensor(X_train_seq, dtype=torch.float32),
    "y_train_seq": torch.tensor(y_train_seq, dtype=torch.float32),

    "X_test_seq":  torch.tensor(X_test_seq, dtype=torch.float32),
    "y_test_seq":  torch.tensor(y_test_seq, dtype=torch.float32),
}

torch.save(cache, SAVE_PATH)

file_size_mb = os.path.getsize(SAVE_PATH) / (1024 * 1024)
print(f"Saved: {SAVE_PATH} ({file_size_mb:.2f} MB)")
print("Done.")
