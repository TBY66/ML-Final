import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Step 1: Load and preprocess data
df = pd.read_csv('SeniorHGS.csv')
print(df.isna().sum())
df = df.dropna(subset=['GRIP_AVG_R', 'GRIP_AVG_L'])
imputer = IterativeImputer()
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

features = ['AGE_AT_SCREEN', 'GENDER', 'DAYS_WEEK', 'CARDIO_MIN', 'STRENGTH_MIN', 'DOM']
target_r = 'GRIP_AVG_R'
target_l = 'GRIP_AVG_L'

X = df_imputed[features]
y_r = df_imputed[target_r]
y_l = df_imputed[target_l]

X_train, X_temp, y_train_r, y_temp_r, y_train_l, y_temp_l = train_test_split(X, y_r, y_l, test_size=0.2, random_state=42)
X_val, X_test, y_val_r, y_test_r, y_val_l, y_test_l = train_test_split(X_temp, y_temp_r, y_temp_l, test_size=0.5, random_state=42)

# Step 2: Scale the data
scaler = StandardScaler()
# scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Step 3: Define models and hyperparameters
models = {
    'Lasso': Lasso(),
    'Ridge': Ridge(),
    'ElasticNet': ElasticNet(),
}

param_grid = {
    'Lasso': {'alpha': [0.1, 1, 10]},
    'Ridge': {'alpha': [0.1, 1, 10]},
    'ElasticNet': {'alpha': [0.1, 1, 10], 'l1_ratio': [0.1, 0.5, 0.9]},
}

# Function to compute adjusted R2 and MSE
def adjusted_r2(y_true, y_pred, X):
    n = len(y_true)
    p = X.shape[1]
    r2 = r2_score(y_true, y_pred)
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test):
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    train_r2 = adjusted_r2(y_train, y_train_pred, X_train)
    val_r2 = adjusted_r2(y_val, y_val_pred, X_val)
    test_r2 = adjusted_r2(y_test, y_test_pred, X_test)
    
    train_mse = mean_squared_error(y_train, y_train_pred)
    val_mse = mean_squared_error(y_val, y_val_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    
    return train_r2, val_r2, test_r2, train_mse, val_mse, test_mse

# Step 4: Train and tune models using GridSearchCV
best_models = {}
results = {}

for model_name, model in models.items():
    grid_search = GridSearchCV(model, param_grid[model_name], cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train_scaled, y_train_r)
    
    best_model_r = grid_search.best_estimator_
    train_r2, val_r2, test_r2, train_mse, val_mse, test_mse = evaluate_model(best_model_r, X_train_scaled, y_train_r, X_val_scaled, y_val_r, X_test_scaled, y_test_r)
    
    best_models[model_name] = best_model_r
    results[model_name] = {
        'Train Adjusted R2': train_r2,
        'Val Adjusted R2': val_r2,
        'Test Adjusted R2': test_r2,
        'Train MSE': train_mse,
        'Val MSE': val_mse,
        'Test MSE': test_mse
    }

# Step 5: Remove outliers using DBSCAN
dbscan = DBSCAN(eps=0.6, min_samples=5)
y_dbscan = dbscan.fit_predict(X_train_scaled)

# Cluster counts
unique_labels, counts = np.unique(y_dbscan, return_counts=True)

# Print the cluster counts
for label, count in zip(unique_labels, counts):
    print(f"Cluster {label}: {count} samples")

X_train_dbscan = X_train_scaled[y_dbscan != -1]
y_train_r_dbscan = y_train_r.iloc[y_dbscan != -1]
y_train_l_dbscan = y_train_l.iloc[y_dbscan != -1]

# Step 6: Evaluate models using DBSCAN outlier removed data
for model_name, model in models.items():
    grid_search = GridSearchCV(model, param_grid[model_name], cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train_dbscan, y_train_r_dbscan)
    
    best_model_r = grid_search.best_estimator_
    train_r2, val_r2, test_r2, train_mse, val_mse, test_mse = evaluate_model(best_model_r, X_train_dbscan, y_train_r_dbscan, X_val_scaled, y_val_r, X_test_scaled, y_test_r)
    
    results[model_name]['Train Adjusted R2 DBSCAN'] = train_r2
    results[model_name]['Val Adjusted R2 DBSCAN'] = val_r2
    results[model_name]['Test Adjusted R2 DBSCAN'] = test_r2
    results[model_name]['Train MSE DBSCAN'] = train_mse
    results[model_name]['Val MSE DBSCAN'] = val_mse
    results[model_name]['Test MSE DBSCAN'] = test_mse

# Step 7: Present Results in DataFrame
results_df = pd.DataFrame(results).T
print(results_df)

# Step 8: Plot results with bars side by side
fig, ax = plt.subplots(2, 2, figsize=(10, 8))

# Set the width of the bars
bar_width = 0.35
index = np.arange(len(results_df))

# 調整顏色為低明度
original_color = '#bfe0ff'  # 深灰色
dbscan_color = '#933407'    # 更淺灰色

# Adjusted R2 comparison for GRIP_AVG_R
ax[0, 0].bar(index, results_df['Test Adjusted R2'], bar_width, label='Original Data')
ax[0, 0].bar(index + bar_width, results_df['Test Adjusted R2 DBSCAN'], bar_width, label='DBSCAN Data')
ax[0, 0].set_title('Adjusted R2 Comparison for GRIP_AVG_R')
ax[0, 0].set_xticks(index + bar_width / 2)
ax[0, 0].set_xticklabels(results_df.index)
ax[0, 0].legend(loc='lower right')

# Adjusted R2 comparison for GRIP_AVG_L
ax[0, 1].bar(index, results_df['Test Adjusted R2'], bar_width, label='Original Data')
ax[0, 1].bar(index + bar_width, results_df['Test Adjusted R2 DBSCAN'], bar_width, label='DBSCAN Data')
ax[0, 1].set_title('Adjusted R2 Comparison for GRIP_AVG_L')
ax[0, 1].set_xticks(index + bar_width / 2)
ax[0, 1].set_xticklabels(results_df.index)
ax[0, 1].legend(loc='lower right')

# MSE comparison for GRIP_AVG_R
ax[1, 0].bar(index, results_df['Test MSE'], bar_width, label='Original Data')
ax[1, 0].bar(index + bar_width, results_df['Test MSE DBSCAN'], bar_width, label='DBSCAN Data')
ax[1, 0].set_title('MSE Comparison for GRIP_AVG_R')
ax[1, 0].set_xticks(index + bar_width / 2)
ax[1, 0].set_xticklabels(results_df.index)
ax[1, 0].legend(loc='lower right')

# MSE comparison for GRIP_AVG_L
ax[1, 1].bar(index, results_df['Test MSE'], bar_width, label='Original Data')
ax[1, 1].bar(index + bar_width, results_df['Test MSE DBSCAN'], bar_width, label='DBSCAN Data')
ax[1, 1].set_title('MSE Comparison for GRIP_AVG_L')
ax[1, 1].set_xticks(index + bar_width / 2)
ax[1, 1].set_xticklabels(results_df.index)
ax[1, 1].legend(loc='lower right')

plt.tight_layout()
plt.show()

