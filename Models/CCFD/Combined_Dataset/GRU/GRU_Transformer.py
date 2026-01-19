# %%
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix

# %%
# Read dataset
df = pd.read_csv(r'/home/ducanh/Credit Card Transactions Fraud Detection/Datasets/combined_data.csv')

# Process date/time columns
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['trans_hour'] = df['trans_date_trans_time'].dt.time.apply(lambda x: str(x)[:2])

df['dob'] = pd.to_datetime(df['dob'])
df['cust_age'] = df['dob'].dt.year.apply(lambda x: 2021 - x)
df['cust_age_groups'] = df['cust_age'].apply(lambda x: 'below 10' if x < 10 else (
    '10-20' if x >= 10 and x < 20 else (
    '20-30' if x >= 20 and x < 30 else (
    '30-40' if x >= 30 and x < 40 else (
    '40-50' if x >= 40 and x < 50 else (
    '50-60' if x >= 50 and x < 60 else (
    '60-70' if x >= 60 and x < 70 else (
    '70-80' if x >= 70 and x < 80 else 'Above 80'))))))))

# Drop unnecessary columns (except 'trans_date_trans_time' and 'cc_num')
drop_col = ['Unnamed: 0', 'merchant', 'first', 'last', 'street', 'city', 'state', 'lat',
            'long','dob', 'unix_time', 'cust_age', 'merch_lat', 'merch_long', 'city_pop']
df.drop(drop_col, axis=1, inplace=True)

# Pivot table for cust_age_groups
age_piv_2 = pd.pivot_table(data=df,
                           index='cust_age_groups',
                           columns='is_fraud',
                           values='amt',
                           aggfunc=np.mean)
age_piv_2.sort_values(by=1, ascending=True, inplace=True)
age_dic = {k: v for (k, v) in zip(age_piv_2.index.values, age_piv_2.reset_index().index.values)}
df['cust_age_groups'] = df['cust_age_groups'].map(age_dic)

# Pivot table for category
merch_cat = df[df['is_fraud'] == 1].groupby('category')['amt'].mean().sort_values(ascending=True)
merch_cat_dic = {k: v for (k, v) in zip(merch_cat.index.values, merch_cat.reset_index().index.values)}
df['category'] = df['category'].map(merch_cat_dic)

# Pivot table for job
job_txn_piv_2 = pd.pivot_table(data=df,
                               index='job',
                               columns='is_fraud',
                               values='amt',
                               aggfunc=np.mean)
job_cat_dic = {k: v for (k, v) in zip(job_txn_piv_2.index.values, job_txn_piv_2.reset_index().index.values)}
df['job'] = df['job'].map(job_cat_dic)

df['trans_hour'] = df['trans_hour'].astype('int')
df = pd.get_dummies(data=df, columns=['gender'], drop_first=True, dtype='int')

# Convert trans_date_trans_time to timestamp for scaling
df['trans_date_trans_time'] = df['trans_date_trans_time'].apply(lambda x: x.timestamp())

# 2️⃣ Train-test split
train, test = train_test_split(df, test_size=0.33, random_state=42, stratify=df['is_fraud'])
print("Train shape:", train.shape)
print("Test shape:", test.shape)

# Save train and test data if needed
train.to_csv(r'/home/ducanh/Credit Card Transactions Fraud Detection/Datasets/combined_fraudTrain.csv', index=False)
test.to_csv(r'/home/ducanh/Credit Card Transactions Fraud Detection/Datasets/combined_fraudTest.csv', index=False)

# Drop the 'trans_num' column from both sets
train.drop('trans_num', axis=1, inplace=True)
test.drop('trans_num', axis=1, inplace=True)

# Split features and label
y_train = train['is_fraud']
X_train = train.drop('is_fraud', axis=1)

y_test = test['is_fraud']
X_test = test.drop('is_fraud', axis=1)

print('Shape of training data:', (X_train.shape, y_train.shape))
print('Shape of testing data:', (X_test.shape, y_test.shape))

# 3️⃣ Scaling the data
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

# Convert back to DataFrame
X_train_sc = pd.DataFrame(data=X_train_sc, columns=X_train.columns)
X_test_sc = pd.DataFrame(data=X_test_sc, columns=X_test.columns)

# %%
# Define the sequence length
sequence_length = 1000  # Number of transactions per sequence

def create_sequences_predict_all(df, sequence_length):
    sequences, labels = [], []
    # Group by credit card number
    grouped = df.groupby('cc_num')
    for user_id, group in grouped:
        # Sort by transaction time (already in timestamp format)
        group = group.sort_values(by='trans_date_trans_time')
        # Drop 'is_fraud' and 'cc_num'
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

# Append is_fraud column back to scaled DataFrames for sequence creation
train_seq_df = X_train_sc.copy()
train_seq_df['is_fraud'] = y_train.values

test_seq_df = X_test_sc.copy()
test_seq_df['is_fraud'] = y_test.values

X_train_seq, y_train_seq = create_sequences_predict_all(train_seq_df, sequence_length)
X_test_seq, y_test_seq = create_sequences_predict_all(test_seq_df, sequence_length)

print("Sequence shape (train):", X_train_seq.shape)
print("Sequence shape (test):", X_test_seq.shape)

# %%
# Create a custom Dataset for the sequences
class FraudDataset(Dataset):
    def __init__(self, X, y):
        self.X = X  # Shape: (num_sequences, sequence_length, num_features)
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

batch_size = 32
train_dataset = FraudDataset(torch.tensor(X_train_seq, dtype=torch.float32),
                               torch.tensor(y_train_seq, dtype=torch.float32))
test_dataset = FraudDataset(torch.tensor(X_test_seq, dtype=torch.float32),
                              torch.tensor(y_test_seq, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# %%
# Hybrid GRU-Transformer model for Fraud Detection with batch_first enabled in Transformer
class FraudGRUTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, gru_layers, transformer_layers, num_heads, transformer_dim, dropout):
        super(FraudGRUTransformer, self).__init__()
        # GRU to process the input sequence
        self.gru = nn.GRU(input_size, hidden_size, gru_layers, batch_first=True)
        
        # Transformer encoder with batch_first=True
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=num_heads, 
            dim_feedforward=transformer_dim, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        
        # Fully connected layer for binary classification
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: (batch_size, sequence_length, input_size)
        gru_out, _ = self.gru(x)  # Output shape: (batch_size, sequence_length, hidden_size)
        
        # Directly pass GRU output to the Transformer (no transpose needed)
        transformer_out = self.transformer_encoder(gru_out)  # (batch_size, sequence_length, hidden_size)
        
        # Use the output of the last time step as the summary representation
        last_token = transformer_out[:, -1, :]  # (batch_size, hidden_size)
        out = self.fc(last_token)  # (batch_size, 1)
        return self.sigmoid(out)

# %%
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_metric = -float('inf')
    best_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1}, Training Loss: {avg_loss:.4f}')
        
        # Early stopping: check if loss has decreased
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= 8:
            print("Early stopping triggered!")
            break

        # Evaluate on validation set
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch).squeeze()
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        # Use 0.5 threshold for binary predictions
        bin_preds = (all_preds > 0.5).astype(int)
        
        # Compute confusion matrix components
        cm = confusion_matrix(all_targets, bin_preds)
        if cm.shape == (2, 2):
            TP, FP, FN, TN = cm[1,1], cm[0,1], cm[1,0], cm[0,0]
        else:
            TP = FP = FN = TN = 0
            if np.sum(bin_preds == 1) == 0:
                TN = len(bin_preds)
            else:
                TP = len(bin_preds)
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        auc = roc_auc_score(all_targets, all_preds)
        
        avg_metric = (f1 + auc) / 2
        print(f'Validation Metrics -- F1: {f1:.4f}, AUC: {auc:.4f}, Avg (F1+AUC)/2: {avg_metric:.4f}')
        
        # Save checkpoint if current metric is the best so far
        if avg_metric > best_metric:
            best_metric = avg_metric
            torch.save(model.state_dict(), 'best_model.pt')
            print("Saving best model checkpoint.\n")
        else:
            print("\n")

# %%
# Hyperparameters for the hybrid model
input_size = X_train_seq.shape[2]  # Number of features
hidden_size = 64
gru_layers = 2
transformer_layers = 2
num_heads = 4
transformer_dim = 128
dropout = 0.1

model = FraudGRUTransformer(input_size, hidden_size, gru_layers, transformer_layers,
                            num_heads, transformer_dim, dropout)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 30
# Using test_loader as validation loader (or create a separate one)
train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device)

# %%
# Prediction and Evaluation on Test Set
model.eval()
all_test_preds = []
all_test_targets = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        preds = model(X_batch).squeeze()
        all_test_preds.extend(preds.cpu().numpy())
        all_test_targets.extend(y_batch.cpu().numpy())
        
all_test_preds = np.array(all_test_preds)
all_test_targets = np.array(all_test_targets)
bin_test_preds = (all_test_preds > 0.5).astype(int)

cm_test = confusion_matrix(all_test_targets, bin_test_preds)
if cm_test.shape == (2, 2):
    TP, FP, FN, TN = cm_test[1,1], cm_test[0,1], cm_test[1,0], cm_test[0,0]
else:
    TP = FP = FN = TN = 0
    if np.sum(bin_test_preds == 1) == 0:
        TN = len(bin_test_preds)
    else:
        TP = len(bin_test_preds)
        
precision_test = TP / (TP + FP) if (TP + FP) > 0 else 0
recall_test = TP / (TP + FN) if (TP + FN) > 0 else 0
f1_test = (2 * precision_test * recall_test) / (precision_test + recall_test) if (precision_test + recall_test) > 0 else 0
auc_test = roc_auc_score(all_test_targets, all_test_preds)

print("Test Evaluation:")
print(f'Precision: {precision_test:.4f}, Recall: {recall_test:.4f}, F1 Score: {f1_test:.4f}, ROC-AUC: {auc_test:.4f}')