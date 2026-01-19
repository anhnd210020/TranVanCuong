````md
# Credit Card Fraud Detection (TranVanCuong)

## Dataset
Download dataset here (Hugging Face):
https://huggingface.co/datasets/anhnd210020/creditcard-fraud-dataset/resolve/main/combined_data.csv

After downloading, put it into:
`TranVanCuong/Datasets/combined_data.csv`

## Run
- Training scripts are in: `Models/CCFD/Combined_Dataset/`

### Step 1: Build sequence cache
First, run the following script to generate `seq_cache.pt`:

```bash
python TranVanCuong/Models/CCFD/Combined_Dataset/GRU/build_cache_sequences_pt.py
````

Output:

* `seq_cache.pt`

### Step 2: Train model (Uncertainty Weighting)

After `seq_cache.pt` is generated, run this script to train the model and get results:

```bash
python TranVanCuong/Models/CCFD/Combined_Dataset/GRU/Uncertainty_Weighting.py
```

