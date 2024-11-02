# Credit Card Fraud Detection using DBSCAN

This project applies the DBSCAN clustering algorithm on a dataset of credit card transactions to detect potential fraud. The dataset is sourced from [Kaggle's Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and includes a variety of transaction records labeled as either fraudulent or legitimate.

## Dataset Overview
- **Source**: [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size**: 284,807 transactions
- **Features**: 30 anonymized numerical features, including `Time`, `Amount`, and a binary `Class` label where:
  - `1` indicates a fraudulent transaction
  - `0` indicates a legitimate transaction

## Objective
The primary goal of this project is to identify clusters of fraudulent transactions using DBSCAN (Density-Based Spatial Clustering of Applications with Noise), a density-based clustering algorithm particularly suitable for datasets where fraud instances form dense regions.

## Approach
1. **Data Preprocessing**: 
   - Normalization of numerical values.
   - Dimensionality reduction using PCA (Principal Component Analysis) to improve performance.
   
2. **Applying DBSCAN**:
   - The DBSCAN algorithm was chosen for its ability to detect clusters of varying densities and to mark outliers, which is helpful in identifying fraud cases.

3. **Evaluation**:
   - Performance is assessed based on the clustering’s ability to identify fraudulent transactions.

## Results
- DBSCAN was able to isolate several clusters, with certain clusters showing a higher density of fraudulent transactions.
- The algorithm’s performance was compared to traditional methods, and DBSCAN proved effective in scenarios where fraud cases formed dense groups within the data.
- Limitations: Due to the high dimensionality and imbalance in the dataset, results required careful tuning of the DBSCAN parameters.

## How to Run
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt