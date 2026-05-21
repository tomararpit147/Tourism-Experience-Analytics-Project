"""
Tourism Experience Analytics
Data Cleaning, Preprocessing, EDA, Model Training
"""

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             mean_squared_error, r2_score, classification_report)
from sklearn.linear_model import LogisticRegression, Ridge
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("Loading data...")
transaction = pd.read_excel("Transaction.xlsx")
user        = pd.read_excel("User.xlsx")
city        = pd.read_excel("City.xlsx")
type_df     = pd.read_excel("Type.xlsx")
mode_df     = pd.read_excel("Mode.xlsx")
continent   = pd.read_excel("Continent.xlsx")
country     = pd.read_excel("Country.xlsx")
region      = pd.read_excel("Region.xlsx")
item        = pd.read_excel("Updated_Item.xlsx")

# ─────────────────────────────────────────────
# 2. DATA CLEANING
# ─────────────────────────────────────────────
print("Cleaning data...")

# Drop placeholder rows (id=0 means unknown)
city    = city[city['CityId'] != 0].copy()
mode_df = mode_df[mode_df['VisitModeId'] != 0].copy()
continent = continent[continent['ContinentId'] != 0].copy()
country = country[country['CountryId'] != 0].copy()
region  = region[region['RegionId'] != 0].copy()

# Fill missing CityId in User with 0 (unknown), cast to int
user['CityId'] = user['CityId'].fillna(0).astype(int)

# Remove transactions with VisitMode=0 (unknown)
transaction = transaction[transaction['VisitMode'] != 0].copy()

# Clip ratings to valid range 1-5
transaction['Rating'] = transaction['Rating'].clip(1, 5)

# ─────────────────────────────────────────────
# 3. MERGE / FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("Merging and feature engineering...")

# Merge transaction with user demographics
df = transaction.merge(user, on='UserId', how='left')

# Merge geography lookup tables
df = df.merge(continent.rename(columns={'Continent':'ContinentName'}), on='ContinentId', how='left')
df = df.merge(region[['RegionId','Region']], on='RegionId', how='left')
df = df.merge(country[['CountryId','Country']], on='CountryId', how='left')
df = df.merge(city[['CityId','CityName']], on='CityId', how='left')

# Merge attraction info
df = df.merge(item[['AttractionId','AttractionTypeId','Attraction','AttractionCityId']], on='AttractionId', how='left')
df = df.merge(type_df, on='AttractionTypeId', how='left')

# Merge VisitMode label
df = df.merge(mode_df.rename(columns={'VisitModeId':'VisitMode','VisitMode':'VisitModeName'}), on='VisitMode', how='left')

# Aggregate attraction-level avg rating
attr_avg = transaction.groupby('AttractionId')['Rating'].mean().rename('AttractionAvgRating')
df = df.merge(attr_avg, on='AttractionId', how='left')

# User-level features
user_features = transaction.groupby('UserId').agg(
    UserAvgRating=('Rating','mean'),
    UserVisitCount=('TransactionId','count')
).reset_index()
df = df.merge(user_features, on='UserId', how='left')

# Fill remaining NaN text columns
for col in ['ContinentName','Region','Country','CityName','Attraction','AttractionType','VisitModeName']:
    df[col] = df[col].fillna('Unknown')

df['AttractionTypeId'] = df['AttractionTypeId'].fillna(-1).astype(int)
df['AttractionAvgRating'] = df['AttractionAvgRating'].fillna(df['Rating'].mean())
df['UserAvgRating'] = df['UserAvgRating'].fillna(df['Rating'].mean())
df['UserVisitCount'] = df['UserVisitCount'].fillna(1)

print(f"Merged dataset shape: {df.shape}")

# ─────────────────────────────────────────────
# 4. ENCODING
# ─────────────────────────────────────────────
FEATURES_REG   = ['ContinentId','RegionId','CountryId','VisitYear','VisitMonth',
                   'AttractionTypeId','AttractionAvgRating','UserAvgRating','UserVisitCount']
FEATURES_CLASS = ['ContinentId','RegionId','CountryId','VisitYear','VisitMonth',
                   'AttractionTypeId','AttractionAvgRating','UserAvgRating','UserVisitCount','Rating']
TARGET_REG   = 'Rating'
TARGET_CLASS = 'VisitMode'

reg_df  = df[FEATURES_REG  + [TARGET_REG]].dropna()
cls_df  = df[FEATURES_CLASS + [TARGET_CLASS]].dropna()

X_reg  = reg_df[FEATURES_REG]
y_reg  = reg_df[TARGET_REG]
X_cls  = cls_df[FEATURES_CLASS]
y_cls  = cls_df[TARGET_CLASS]

# ─────────────────────────────────────────────
# 5. MODEL TRAINING
# ─────────────────────────────────────────────
print("Training regression model...")
X_tr, X_te, y_tr, y_te = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

reg_models = {
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Ridge':        Ridge(alpha=1.0),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42)  # placeholder
}
# Use RF for regression
rf_reg = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1)
rf_reg.fit(X_tr, y_tr)
y_pred_reg = rf_reg.predict(X_te)
r2  = r2_score(y_te, y_pred_reg)
mse = mean_squared_error(y_te, y_pred_reg)
rmse = np.sqrt(mse)
print(f"  Regression RF  → R²={r2:.3f}, RMSE={rmse:.3f}")

ridge = Ridge(alpha=1.0)
ridge.fit(X_tr, y_tr)
y_pred_ridge = ridge.predict(X_te)
r2_r  = r2_score(y_te, y_pred_ridge)
rmse_r = np.sqrt(mean_squared_error(y_te, y_pred_ridge))
print(f"  Regression Ridge → R²={r2_r:.3f}, RMSE={rmse_r:.3f}")

print("Training classification model...")
X_ctr, X_cte, y_ctr, y_cte = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls)

rf_cls = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
rf_cls.fit(X_ctr, y_ctr)
y_pred_cls = rf_cls.predict(X_cte)
acc = accuracy_score(y_cte, y_pred_cls)
f1  = f1_score(y_cte, y_pred_cls, average='weighted')
print(f"  Classification RF  → Acc={acc:.3f}, F1={f1:.3f}")

gb_cls = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
gb_cls.fit(X_ctr, y_ctr)
y_pred_gb = gb_cls.predict(X_cte)
acc_gb = accuracy_score(y_cte, y_pred_gb)
f1_gb  = f1_score(y_cte, y_pred_gb, average='weighted')
print(f"  Classification GB  → Acc={acc_gb:.3f}, F1={f1_gb:.3f}")

# Choose best classifier
best_cls = rf_cls if acc >= acc_gb else gb_cls
best_cls_name = "RandomForest" if acc >= acc_gb else "GradientBoosting"
print(f"  Best classifier: {best_cls_name}")

# ─────────────────────────────────────────────
# 6. RECOMMENDATION SYSTEM
# ─────────────────────────────────────────────
print("Building recommendation system...")

# Build user-item rating matrix (collaborative filtering)
pivot = transaction.pivot_table(index='UserId', columns='AttractionId', values='Rating', aggfunc='mean')
pivot_filled = pivot.fillna(0)

# KNN-based collaborative filtering
user_matrix = csr_matrix(pivot_filled.values)
knn_model = NearestNeighbors(n_neighbors=11, metric='cosine', algorithm='brute', n_jobs=-1)
knn_model.fit(user_matrix)

# ─────────────────────────────────────────────
# 7. SAVE ARTIFACTS
# ─────────────────────────────────────────────
print("Saving models and data...")

artifacts = {
    'rf_reg':          rf_reg,
    'ridge_reg':       ridge,
    'best_cls':        best_cls,
    'best_cls_name':   best_cls_name,
    'knn_model':       knn_model,
    'pivot_filled':    pivot_filled,
    'features_reg':    FEATURES_REG,
    'features_cls':    FEATURES_CLASS,
    'df':              df,
    'mode_map':        dict(zip(mode_df['VisitModeId'], mode_df['VisitMode'])),
    'type_map':        dict(zip(type_df['AttractionTypeId'], type_df['AttractionType'])),
    'item':            item,
    'type_df':         type_df,
    'mode_df':         mode_df,
    'continent':       continent,
    'region':          region,
    'country':         country,
    # Evaluation metrics
    'reg_metrics': {
        'RandomForest': {'R2': round(r2,4), 'RMSE': round(rmse,4), 'MSE': round(mse,4)},
        'Ridge':        {'R2': round(r2_r,4), 'RMSE': round(rmse_r,4), 'MSE': round(mean_squared_error(y_te,y_pred_ridge),4)},
    },
    'cls_metrics': {
        'RandomForest': {
            'Accuracy': round(acc,4),
            'F1': round(f1,4),
            'Precision': round(precision_score(y_cte, y_pred_cls, average='weighted'),4),
            'Recall': round(recall_score(y_cte, y_pred_cls, average='weighted'),4),
        },
        'GradientBoosting': {
            'Accuracy': round(acc_gb,4),
            'F1': round(f1_gb,4),
            'Precision': round(precision_score(y_cte, y_pred_gb, average='weighted'),4),
            'Recall': round(recall_score(y_cte, y_pred_gb, average='weighted'),4),
        },
    },
}

with open('model_artifacts.pkl','wb') as f:
    pickle.dump(artifacts, f)

print("Done! Artifacts saved to model_artifacts.pkl")