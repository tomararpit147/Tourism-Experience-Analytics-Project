# ✈️ Tourism Experience Analytics
### Classification · Prediction · Recommendation System

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Business Use Cases](#business-use-cases)
3. [Dataset Description](#dataset-description)
4. [Project Structure](#project-structure)
5. [Installation & Setup](#installation--setup)
6. [Data Cleaning & Preprocessing](#data-cleaning--preprocessing)
7. [Exploratory Data Analysis](#exploratory-data-analysis)
8. [Machine Learning Models](#machine-learning-models)
9. [Recommendation System](#recommendation-system)
10. [Model Performance](#model-performance)
11. [Streamlit Application](#streamlit-application)
12. [Key Insights](#key-insights)
13. [Tech Stack](#tech-stack)

---

## 📌 Project Overview

**Domain:** Tourism  
**Goal:** Leverage historical travel data to build a full-stack analytics system that can predict user satisfaction, classify visitor behavior, and deliver personalized attraction recommendations.

Tourism agencies and travel platforms generate vast amounts of user interaction data. This project turns that data into three actionable intelligence systems:

| System | Task | Output |
|--------|------|--------|
| **Regression** | Predict the rating a user will give an attraction | Score from 1.0 – 5.0 |
| **Classification** | Predict the user's visit mode | Business / Couples / Family / Friends / Solo |
| **Recommendation** | Suggest attractions the user will enjoy | Ranked list of attractions |

All three systems are served through an interactive **Streamlit web application** with rich visualizations.

---

## 💼 Business Use Cases

**Personalized Recommendations**
Travel platforms can suggest attractions aligned with a user's past behavior and demographic profile, increasing engagement and booking conversion.

**Customer Segmentation**
Classifying visitors by travel mode (Business, Family, Couples, etc.) enables targeted marketing campaigns and tailored content delivery.

**Tourism Analytics**
Identifying popular attractions, peak travel months, and high-performing regions helps tourism operators optimize their offerings and resource planning.

**Customer Retention**
Predicting likely satisfaction scores before a visit allows agencies to proactively improve service, set correct expectations, and reduce negative reviews.

---

## 📂 Dataset Description

The project uses 10 relational tables covering users, attractions, transactions, and geography.

### Transaction Data (`Transaction.xlsx`)
The primary fact table — 52,930 records of user visits.

| Column | Description |
|--------|-------------|
| `TransactionId` | Unique identifier for each visit record |
| `UserId` | Reference to the visiting user |
| `VisitYear` | Year of the visit (2013–2022) |
| `VisitMonth` | Month of the visit (1–12) |
| `VisitMode` | Encoded visit type (1=Business, 2=Couples, 3=Family, 4=Friends, 5=Solo) |
| `AttractionId` | Reference to the attraction visited |
| `Rating` | User's satisfaction rating (1–5) |

### User Data (`User.xlsx`)
Demographic profile for 33,530 unique users.

| Column | Description |
|--------|-------------|
| `UserId` | Unique user identifier |
| `ContinentId` | User's continent |
| `RegionId` | User's region |
| `CountryId` | User's country |
| `CityId` | User's city (4 missing values) |

### Attraction Data (`Updated_Item.xlsx`)
1,698 tourist attractions with location and category.

| Column | Description |
|--------|-------------|
| `AttractionId` | Unique attraction identifier |
| `AttractionCityId` | City where the attraction is located |
| `AttractionTypeId` | Category of the attraction |
| `Attraction` | Name of the attraction |
| `AttractionAddress` | Physical address |

### Reference / Lookup Tables

| File | Key Columns | Description |
|------|-------------|-------------|
| `Continent.xlsx` | `ContinentId`, `Continent` | 5 continents: Africa, America, Asia, Australia & Oceania, Europe |
| `Region.xlsx` | `RegionId`, `Region`, `ContinentId` | 21 sub-continental regions |
| `Country.xlsx` | `CountryId`, `Country`, `RegionId` | 164 countries |
| `City.xlsx` | `CityId`, `CityName`, `CountryId` | 9,142 cities |
| `Type.xlsx` | `AttractionTypeId`, `AttractionType` | 17 attraction categories |
| `Mode.xlsx` | `VisitModeId`, `VisitMode` | 5 visit modes |

### Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Transactions | 52,930 |
| Unique Users | 33,530 |
| Unique Attractions | 1,698 |
| Date Range | 2013 – 2022 |
| Average Rating | 4.16 / 5.0 |
| Rating Distribution | 1★ (2.4%) · 2★ (3.8%) · 3★ (14.6%) · 4★ (33.9%) · 5★ (45.2%) |
| Visit Mode Split | Couples (40.8%) · Family (28.7%) · Friends (20.7%) · Solo (8.5%) · Business (1.2%) |
| Top User Continents | Asia · Australia & Oceania · Europe · America · Africa |

---

## 🗂 Project Structure

```
tourism_project/
│
├── app.py                    # Streamlit application (6 pages)
├── preprocess_train.py       # Data pipeline + model training script
├── model_artifacts.pkl       # Serialized models, encoders & processed data
├── README.md                 # This file
│
├── Transaction.xlsx          # Raw data — transactions
├── User.xlsx                 # Raw data — user demographics
├── Updated_Item.xlsx         # Raw data — attractions (1,698 items)
├── Item.xlsx                 # Raw data — attractions (30 items, subset)
├── City.xlsx                 # Lookup — cities
├── Country.xlsx              # Lookup — countries
├── Region.xlsx               # Lookup — regions
├── Continent.xlsx            # Lookup — continents
├── Type.xlsx                 # Lookup — attraction types
└── Mode.xlsx                 # Lookup — visit modes
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.8 or higher

### 1. Install Dependencies
```bash
pip install streamlit scikit-learn pandas numpy openpyxl plotly scipy
```

### 2. Train Models (first time only)
```bash
python preprocess_train.py
```
This will:
- Load and clean all Excel datasets
- Merge tables into a consolidated feature set
- Train regression and classification models
- Build the recommendation engine
- Save everything to `model_artifacts.pkl`

Expected output:
```
Loading data...
Cleaning data...
Merging and feature engineering...
Merged dataset shape: (52930, 23)
Training regression model...
  Regression RF   → R²=0.743, RMSE=0.492
  Regression Ridge → R²=0.738, RMSE=0.497
Training classification model...
  Classification RF  → Acc=0.499, F1=0.441
  Classification GB  → Acc=0.496, F1=0.446
Building recommendation system...
Done! Artifacts saved to model_artifacts.pkl
```

### 3. Launch the App
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`

---

## 🧹 Data Cleaning & Preprocessing

### Cleaning Steps

| Issue | Fix Applied |
|-------|-------------|
| Placeholder rows (id = 0) in lookup tables | Filtered out before merging |
| 4 missing `CityId` values in User table | Filled with 0 (unknown), cast to int |
| Transactions with `VisitMode = 0` (unknown) | Removed from analysis |
| Out-of-range ratings | Clipped to valid range [1, 5] |
| Missing attraction type after merge | Filled with `'Unknown'` |
| Missing `AttractionTypeId` after merge | Filled with `-1` (sentinel value) |

### Feature Engineering

The 10 source tables were merged into a single analytical dataset with 23 columns:

**User-level aggregates:**
- `UserAvgRating` — historical average rating per user
- `UserVisitCount` — total number of visits per user

**Attraction-level aggregates:**
- `AttractionAvgRating` — historical average rating per attraction

**Geography lookups joined in:**
- `ContinentName`, `Region`, `Country`, `CityName`

**Category labels decoded:**
- `VisitModeName` (e.g. "Family"), `AttractionType` (e.g. "Beaches")

### Feature Set for Models

**Regression features (9):**
`ContinentId · RegionId · CountryId · VisitYear · VisitMonth · AttractionTypeId · AttractionAvgRating · UserAvgRating · UserVisitCount`

**Classification features (10):**
Same as regression + `Rating`

---

## 📊 Exploratory Data Analysis

The EDA page of the Streamlit app provides 12+ interactive visualizations covering:

### Ratings Analysis
- Overall rating distribution — strongly skewed toward 4★ and 5★
- Average rating by visit mode — Solo travelers rate highest on average
- Average rating trend over years — slight dip during 2020 (COVID-19 effect)
- Average rating by month — peak satisfaction in summer months

### Visit Mode Analysis
- Visit mode distribution — Couples dominate at 40.8%, Business least common at 1.2%
- Visit mode trends by year — steady growth in Solo travel since 2017
- Seasonality chart — Family travel peaks in July–August school holidays

### Geographic Insights
- User distribution by continent — Asia and Australia & Oceania lead
- Average rating by continent — Europe has highest average satisfaction
- Top 15 user countries — diverse global user base

### Attraction Analysis
- Top attraction types by visit count — Beaches, Points of Interest, Historic Sites lead
- Average rating by attraction type — Spas and Nature areas rated highest
- Visits vs Rating scatter — identifying high-volume, high-rated "star" attractions

---

## 🤖 Machine Learning Models

### Task 1 — Regression: Predicting Attraction Ratings

**Objective:** Predict the rating (1–5) a user will give an attraction before they visit.

**Models Trained:**
- Random Forest Regressor (`n_estimators=150, max_depth=10`)
- Ridge Regression (`alpha=1.0`)

**Train/Test Split:** 80% / 20%, random state = 42

**Best Model:** Random Forest Regressor

| Model | R² | RMSE | MSE |
|-------|----|------|-----|
| **Random Forest** | **0.7426** | **0.4924** | **0.2424** |
| Ridge Regression | 0.7375 | 0.4972 | 0.2472 |

The Random Forest model explains **74.3% of variance** in user ratings — strong performance given the subjective nature of satisfaction scores.

### Task 2 — Classification: Predicting Visit Mode

**Objective:** Predict whether a user visits as Business, Couples, Family, Friends, or Solo.

**Models Trained:**
- Random Forest Classifier (`n_estimators=150, max_depth=12`)
- Gradient Boosting Classifier (`n_estimators=100, max_depth=5`)

**Train/Test Split:** 80% / 20%, stratified by class, random state = 42

**Best Model:** Random Forest Classifier

| Model | Accuracy | F1 (weighted) | Precision | Recall |
|-------|----------|---------------|-----------|--------|
| **Random Forest** | **0.4993** | **0.4414** | **0.5309** | **0.4993** |
| Gradient Boosting | 0.4957 | 0.4456 | 0.4990 | 0.4957 |

> **Note on classification accuracy:** This is a 5-class imbalanced problem (random baseline ≈ 20%). The ~50% accuracy represents a **2.5× improvement over random chance**. The class imbalance (Couples = 40.8%, Business = 1.2%) and the subtle behavioral signals between categories (e.g., Friends vs Couples) are the primary challenges. Future improvements: SMOTE oversampling, additional behavioral features, or a user-interaction sequence model.

---

## 🎯 Recommendation System

### Collaborative Filtering (User-Based KNN)

**Algorithm:** K-Nearest Neighbors on a user-item rating matrix
**Similarity Metric:** Cosine similarity
**Implementation:** `scipy.sparse.csr_matrix` + `sklearn.neighbors.NearestNeighbors`

**How it works:**
1. Build a user × attraction rating matrix (33,530 users × 1,698 attractions)
2. For a given user, find the 10 most similar users by cosine similarity of their rating vectors
3. Aggregate ratings from similar users for attractions the target user hasn't visited
4. Return the top-N unvisited attractions ranked by aggregated score

**Coverage:** All 33,530 users with at least one transaction

### Content-Based Filtering

**How it works:**
1. User selects an attraction type they enjoy (e.g. "Beaches")
2. System retrieves all attractions of that type from the item catalogue
3. Ranks them by historical average rating and visit count
4. Returns the top-N highest-rated attractions of that type

**Coverage:** All 17 attraction types, 1,698 attractions

---

## 📈 Model Performance

### Regression Summary

```
Random Forest Regressor
  ├─ R²    : 0.7426  (explains 74.3% of rating variance)
  ├─ RMSE  : 0.4924  (predictions within ~0.5 stars on average)
  └─ MSE   : 0.2424

Ridge Regression
  ├─ R²    : 0.7375
  ├─ RMSE  : 0.4972
  └─ MSE   : 0.2472
```

### Classification Summary

```
Random Forest Classifier (Best)
  ├─ Accuracy  : 49.93%  (vs 20% random baseline for 5 classes)
  ├─ F1 Score  : 44.14%
  ├─ Precision : 53.09%
  └─ Recall    : 49.93%

Gradient Boosting Classifier
  ├─ Accuracy  : 49.57%
  ├─ F1 Score  : 44.56%
  ├─ Precision : 49.90%
  └─ Recall    : 49.57%
```

### Recommendation System

```
Collaborative Filtering (KNN)
  ├─ Users covered    : 33,530
  ├─ Attractions      : 1,698
  ├─ Neighbors (k)    : 10
  └─ Similarity       : Cosine

Content-Based Filtering
  ├─ Attraction types : 17
  └─ Ranking criteria : Historical avg rating + visit frequency
```

---

## 🖥️ Streamlit Application

The app is organized into **6 interactive pages**:

### 🏠 Overview
- Summary metrics (total transactions, users, attractions, avg rating)
- Project objectives and use case descriptions
- Tabbed dataset explorer for Transactions, Attractions, and Users

### 📊 EDA & Visualizations
- 4 tabs: Ratings · Visit Modes · Geography · Attractions
- 12+ interactive Plotly charts with hover tooltips
- Seasonality analysis, geographic heatmaps, scatter plots

### ⭐ Rating Prediction
- Dropdown inputs: Continent → Region → Country (cascading)
- Sliders for attraction avg rating, user avg rating, visit count
- Output: predicted rating with a color-coded gauge chart (1–5 scale)

### 🗺️ Visit Mode Prediction
- Same geographic inputs + attraction type and rating context
- Output: predicted visit mode with full probability bar chart for all 5 classes
- Icons per predicted mode (💼 💑 👨‍👩‍👧‍👦 👫 🧍)

### 🎯 Recommendations
- **Collaborative tab:** Select any of the 33,530 user IDs → receive top-N unvisited attraction recommendations with scores and addresses
- **Content-based tab:** Select an attraction type → ranked list by historical rating

### 📈 Model Performance
- Metrics table for all trained models (regression and classification)
- Radar chart comparing classifiers across 4 metrics
- Recommendation system specification summary

---

## 💡 Key Insights

**Visitor behavior:**
- Couples are the most frequent travelers (40.8%), while Business travelers are rare (1.2%)
- Solo travel has grown steadily since 2017, suggesting a shifting demographic
- Family travel shows clear July–August seasonality aligned with school holidays

**Satisfaction drivers:**
- Attractions with higher historical average ratings reliably attract higher new ratings (top regression feature)
- Solo travelers tend to rate attractions higher on average than group travelers
- European users report the highest average satisfaction across all visit modes

**Attraction popularity:**
- Beaches, Points of Interest & Landmarks, and Historic Sites receive the most visits
- Spas and Nature & Wildlife Areas receive the highest average satisfaction ratings
- A small set of top attractions receive a disproportionate share of visits (long-tail distribution)

**Geography:**
- Asia and Australia & Oceania account for the most visits combined (~57% of transactions)
- Africa has relatively few users but above-average satisfaction ratings

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.8+ |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn (RandomForest, GradientBoosting, Ridge, KNN) |
| **Recommendation** | Scipy (sparse matrix), Scikit-learn NearestNeighbors |
| **Visualization** | Plotly Express, Plotly Graph Objects |
| **Web Application** | Streamlit |
| **Data Format** | Excel (.xlsx) via OpenPyXL |
| **Model Persistence** | Python Pickle |

---

## 👤 Author

**Project:** Tourism Experience Analytics  
**Domain:** Tourism  
**Skills Demonstrated:** Data Cleaning · EDA · Feature Engineering · Regression · Classification · Collaborative Filtering · Streamlit Deployment
