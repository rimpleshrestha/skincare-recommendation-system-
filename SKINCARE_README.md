# AI-Driven Skincare Recommendation Dataset
## Kathmandu Valley Females (18-26)

---

## 🌸 Dataset Overview

A comprehensive synthetic dataset of **1,000,000 records** designed for training AI-driven skincare recommendation systems, specifically targeted at **females aged 18-26 living in Kathmandu Valley, Nepal**.

| Metric | Value |
|--------|-------|
| **Total Records** | 1,000,000 |
| **Target Demographic** | Females 18-26 |
| **Geographic Scope** | 13 municipalities in Kathmandu Valley |
| **Total Features** | 48 columns |
| **Time Period** | Jan 2023 - Nov 2025 |
| **File Size** | ~471 MB |

---

## 📁 Files Included

1. **skincare_recommendations_1M.csv** (Main dataset)
   - 1,000,000 user records
   - 48 features per user
   - Demographics, lifestyle, skin profile, recommendations

2. **products_catalog.csv** (Product database)
   - 41 skincare products
   - Pricing, categories, suitability

3. **ingredients_catalog.csv** (Ingredient reference)
   - 15 key skincare ingredients
   - Benefits and categories

4. **locations_catalog.csv** (Area database)
   - 13 Kathmandu Valley municipalities
   - Environmental characteristics

5. **SKINCARE_README.md** (This documentation)

6. **SKINCARE_QUICK_REFERENCE.md** (Quick reference guide)

---

## 🎯 Dataset Features

### Demographics (7 columns)

| Feature | Description | Values |
|---------|-------------|--------|
| `user_id` | Unique identifier | SKC0000001 - SKC1000000 |
| `age` | Age in years | 18-26 |
| `location` | Municipality in Kathmandu Valley | 13 locations |
| `occupation` | Current occupation | 13 categories |
| `education_level` | Highest education | 5 levels |
| `income_bracket` | Monthly income category | 6 brackets |
| `monthly_income_npr` | Actual income in NPR | 0-150,000 |

### Skin Profile (6 columns)

| Feature | Description | Values |
|---------|-------------|--------|
| `skin_type` | Primary skin type | Oily, Dry, Combination, Normal, Sensitive |
| `skin_tone` | Fitzpatrick scale | Type III, IV, V, VI |
| `primary_concern` | Main skin concern | 10 concerns |
| `secondary_concerns` | Additional concerns | Multiple (;-separated) |
| `allergies` | Known ingredient allergies | 8 categories |

### Lifestyle Factors (11 columns)

| Feature | Description | Values |
|---------|-------------|--------|
| `sleep_hours` | Average sleep per night | 4-5 hrs to 8+ hrs |
| `daily_water_intake` | Daily water consumption | <1L to 4L+ |
| `stress_level` | Perceived stress | Low, Moderate, High, Very High |
| `exercise_frequency` | Physical activity | Never to Daily |
| `diet_type` | Dietary preference | Vegetarian, Non-Veg, etc. |
| `junk_food_frequency` | Processed food intake | Never to Daily |
| `dairy_intake` | Dairy consumption level | None to High |
| `sugar_intake` | Sugar consumption level | Low to Very High |
| `sun_exposure` | Daily sun exposure | Minimal to Very High |
| `sunscreen_usage` | SPF application habit | Never to Daily |
| `pollution_exposure` | Environmental exposure | Low to Very High |

### Current Skincare Habits (8 columns)

| Feature | Description | Values |
|---------|-------------|--------|
| `current_routine` | Routine complexity | None to Extensive |
| `product_preference` | Brand preference | K-Beauty, Ayurvedic, etc. |
| `skincare_budget` | Monthly budget (NPR) | Under 500 to 5000+ |
| `makeup_frequency` | Makeup usage | Never to Daily |
| `shopping_channel` | Where they buy | Online, Local Shop, etc. |
| `info_source` | Information source | Social Media, YouTube, etc. |
| `commute_type` | Daily transportation | Walking, Bus, Motorcycle, etc. |

### Hormonal Factors (3 columns)

| Feature | Description | Values |
|---------|-------------|--------|
| `menstrual_skin_changes` | Period-related skin issues | None, Mild, Moderate, Severe |
| `hormonal_condition` | Medical conditions | None, PCOS, Thyroid, Other |
| `contraceptive_use` | Contraceptive method | None, Oral Pills, IUD, etc. |

### Environmental Context (4 columns)

| Feature | Description | Values |
|---------|-------------|--------|
| `area_pollution` | Location pollution level | Low, Moderate, High |
| `area_altitude` | Elevation in meters | 1350-1600m |
| `area_water_hardness` | Water quality | Soft, Moderate, Hard |
| `current_season` | Season at registration | Pre-Monsoon, Monsoon, Post-Monsoon, Winter |

### Recommendations (8 columns - Target Variables)

| Feature | Description | Values |
|---------|-------------|--------|
| `rec_cleanser` | Recommended cleanser type | 8 options |
| `rec_serum` | Recommended serum | 11 options |
| `rec_moisturizer` | Recommended moisturizer | 7 options |
| `rec_sunscreen` | Recommended sunscreen | 7 options |
| `rec_treatment` | Recommended treatment | 10 options |
| `rec_ingredients` | Beneficial ingredients | Multiple (;-separated) |
| `avoid_ingredients` | Ingredients to avoid | Multiple (;-separated) |
| `rec_routine_type` | Recommended routine complexity | 5 levels |

### Predictions (4 columns)

| Feature | Description | Range |
|---------|-------------|-------|
| `skin_health_score` | Overall skin health | 0-100 |
| `predicted_satisfaction` | Expected satisfaction | 1.0-5.0 |
| `engagement_probability` | Likelihood to follow routine | 0.00-1.00 |
| `est_monthly_cost` | Estimated routine cost (NPR) | 200-10,000 |

---

## 📊 Key Statistics

### Skin Type Distribution

| Skin Type | Count | Percentage |
|-----------|-------|------------|
| Combination | 320,000 | 32.0% |
| Dry | 280,000 | 28.0% |
| Oily | 220,000 | 22.0% |
| Normal | 130,000 | 13.0% |
| Sensitive | 50,000 | 5.0% |

### Top Skin Concerns

| Concern | Count | Percentage |
|---------|-------|------------|
| Acne/Breakouts | 250,000 | 25.0% |
| Dark Spots/Hyperpigmentation | 220,000 | 22.0% |
| Dullness | 180,000 | 18.0% |
| Uneven Skin Tone | 120,000 | 12.0% |
| Oiliness/Large Pores | 80,000 | 8.0% |

### Location Distribution

| Location | Count | Percentage |
|----------|-------|------------|
| Kathmandu Metropolitan | 350,000 | 35.0% |
| Lalitpur Metropolitan | 200,000 | 20.0% |
| Bhaktapur Municipality | 100,000 | 10.0% |
| Kirtipur Municipality | 50,000 | 5.0% |
| Madhyapur Thimi | 50,000 | 5.0% |
| Other municipalities | 250,000 | 25.0% |

### Age Distribution

| Age | Approximate % |
|-----|---------------|
| 18-20 | ~33% |
| 21-23 | ~33% |
| 24-26 | ~33% |

### Income Distribution

| Income Bracket | Percentage |
|----------------|------------|
| No Income (Students) | 35% |
| Low (5-15K NPR) | 15% |
| Lower-Middle (15-30K) | 22% |
| Middle (30-50K) | 18% |
| Upper-Middle (50-80K) | 7% |
| High (80K+) | 3% |

---

## 🎯 Use Cases

### 1. Product Recommendation Engine
Train ML models to recommend specific products based on user profile:
```python
features = ['skin_type', 'primary_concern', 'skin_health_score', 'budget', ...]
target = ['rec_cleanser', 'rec_serum', 'rec_moisturizer', ...]
```

### 2. Skin Health Prediction
Predict skin health score from lifestyle factors:
```python
features = ['sleep_hours', 'water_intake', 'stress', 'exercise', 'diet', ...]
target = 'skin_health_score'
```

### 3. Customer Segmentation
Cluster users for targeted marketing:
```python
# K-means on demographic + behavior features
from sklearn.cluster import KMeans
```

### 4. Budget-Aware Recommendations
Build cost-optimized routine suggestions:
```python
features = ['skincare_budget', 'skin_type', 'concerns', ...]
target = 'rec_routine_type'
```

### 5. Engagement Prediction
Predict likelihood of following recommendations:
```python
features = ['current_routine', 'info_source', 'age', ...]
target = 'engagement_probability'
```

### 6. Geographic Analysis
Study location-specific skin concerns:
```python
# Analyze correlation between area_pollution and skin concerns
df.groupby(['location', 'primary_concern']).size()
```

---

## 🔬 Data Generation Logic

### Skin Health Score Calculation

The `skin_health_score` (0-100) is calculated based on:

| Factor | Positive Impact | Negative Impact |
|--------|-----------------|-----------------|
| Sleep | 7-8 hrs: +5 | 4-5 hrs: -10 |
| Water | 3-4L: +8 | <1L: -10 |
| Stress | Low: +8 | Very High: -15 |
| Exercise | Daily: +10 | Never: -8 |
| Junk Food | Never: +10 | Daily: -12 |
| Sunscreen | Daily: +10 | Never: -10 |
| Pollution | Low: +5 | Very High: -15 |

### Recommendation Logic

Products are recommended based on:

**Cleansers** → Matched to skin type
- Oily → Gel, Salicylic, Charcoal
- Dry → Cream, Milk, Oil
- Sensitive → Gentle, Micellar

**Serums** → Matched to primary concern
- Acne → Niacinamide, Salicylic
- Dark spots → Vitamin C, Alpha Arbutin
- Dryness → Hyaluronic Acid

**Moisturizers** → Matched to skin type
- Oily → Gel, Oil-free
- Dry → Rich cream, Ceramide
- Sensitive → Barrier repair

---

## 💡 ML Model Suggestions

### Classification Tasks
- **Product recommendation**: Random Forest, XGBoost, Neural Networks
- **Routine type**: Logistic Regression, SVM
- **Skin type prediction**: Decision Trees, Gradient Boosting

### Regression Tasks
- **Satisfaction prediction**: Linear Regression, Ridge, Neural Networks
- **Skin health score**: Gradient Boosting, Random Forest
- **Budget estimation**: XGBoost, LightGBM

### Clustering Tasks
- **Customer segmentation**: K-Means, DBSCAN, Hierarchical
- **Product grouping**: Spectral Clustering

### Deep Learning
- **Multi-output recommendation**: Multi-task Neural Networks
- **Sequential recommendations**: LSTM, Transformers

---

## 📋 Sample Code

### Loading the Data
```python
import pandas as pd

# Load main dataset
df = pd.read_csv('skincare_recommendations_1M.csv')

# Load catalogs
products = pd.read_csv('products_catalog.csv')
ingredients = pd.read_csv('ingredients_catalog.csv')
locations = pd.read_csv('locations_catalog.csv')

print(f"Records: {len(df):,}")
print(f"Features: {len(df.columns)}")
```

### Basic Analysis
```python
# Skin type distribution
print(df['skin_type'].value_counts(normalize=True))

# Average skin health by location
print(df.groupby('location')['skin_health_score'].mean().sort_values())

# Correlation between lifestyle and skin health
lifestyle_cols = ['sleep_hours', 'daily_water_intake', 'stress_level', 'exercise_frequency']
# Encode and correlate with skin_health_score
```

### Building a Recommendation Model
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Prepare features
feature_cols = ['skin_type', 'skin_tone', 'primary_concern', 'age', 
                'skincare_budget', 'skin_health_score']

# Encode categorical variables
le = LabelEncoder()
for col in feature_cols:
    if df[col].dtype == 'object':
        df[col + '_encoded'] = le.fit_transform(df[col])

# Train model for cleanser recommendation
X = df[[c + '_encoded' if c in ['skin_type', 'skin_tone', 'primary_concern', 'skincare_budget'] else c for c in feature_cols]]
y = le.fit_transform(df['rec_cleanser'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
print(f"Accuracy: {model.score(X_test, y_test):.2%}")
```

---

## ⚠️ Important Notes

1. **Synthetic Data**: This is artificially generated data for training purposes. Real-world validation is required.

2. **Nepal-Specific**: Distributions are based on Kathmandu Valley context (pollution levels, water quality, climate).

3. **Age Range**: Specifically designed for 18-26 age group. Different age groups would have different distributions.

4. **Gender**: All records are for females. Male skincare would require different concern distributions.

5. **Cultural Context**: Product preferences reflect local availability (Ayurvedic, K-Beauty popularity in Nepal).

6. **Seasonal Variation**: Nepal's distinct seasons (monsoon, winter, etc.) affect skin differently.

---

## 🎓 Educational Applications

This dataset is ideal for:
- Machine Learning coursework
- Data Science projects
- Skincare app development
- Market research analysis
- Customer behavior studies
- Recommendation system tutorials

---

## 📈 Potential Improvements

For production use, consider:
- Real user data collection
- A/B testing recommendations
- Dermatologist validation
- Product efficacy tracking
- Longitudinal skin health monitoring

---

**Perfect for building Nepal's first AI-driven skincare recommendation system! 🇳🇵✨**
