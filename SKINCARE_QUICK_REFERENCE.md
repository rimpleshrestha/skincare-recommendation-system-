# Skincare Dataset - Quick Reference Guide

## 🌸 Dataset at a Glance

| Metric | Value |
|--------|-------|
| **Total Records** | 1,000,000 |
| **Target** | Females 18-26, Kathmandu Valley |
| **Features** | 48 columns |
| **File Size** | ~471 MB |

---

## 📊 Quick Stats

### Skin Types
```
Combination: 32% | ████████████████
Dry:         28% | ██████████████
Oily:        22% | ███████████
Normal:      13% | ██████▌
Sensitive:    5% | ██▌
```

### Top 5 Concerns
```
Acne/Breakouts:      25% | █████████████
Dark Spots:          22% | ███████████
Dullness:            18% | █████████
Uneven Tone:         12% | ██████
Oiliness/Pores:       8% | ████
```

### Budget Distribution (Monthly NPR)
```
Under 500:    22% | ███████████
500-1000:     28% | ██████████████
1000-2000:    25% | ████████████▌
2000-3500:    15% | ███████▌
3500-5000:     6% | ███
5000+:         4% | ██
```

---

## 🎯 Target Variables (What to Predict)

| Variable | Type | Options |
|----------|------|---------|
| `rec_cleanser` | Classification | 8 types |
| `rec_serum` | Classification | 11 types |
| `rec_moisturizer` | Classification | 7 types |
| `rec_sunscreen` | Classification | 7 types |
| `rec_treatment` | Classification | 10 types |
| `rec_routine_type` | Classification | 5 levels |
| `skin_health_score` | Regression | 0-100 |
| `predicted_satisfaction` | Regression | 1-5 |
| `engagement_probability` | Regression | 0-1 |

---

## 📁 Files

```
📦 skincare_dataset/
├── 📄 skincare_recommendations_1M.csv  (471 MB - Main data)
├── 📄 products_catalog.csv             (2 KB - 41 products)
├── 📄 ingredients_catalog.csv          (1 KB - 15 ingredients)
├── 📄 locations_catalog.csv            (1 KB - 13 areas)
├── 📄 SKINCARE_README.md               (Full documentation)
└── 📄 SKINCARE_QUICK_REFERENCE.md      (This file)
```

---

## ⚡ Quick Start

### Load Data
```python
import pandas as pd
df = pd.read_csv('skincare_recommendations_1M.csv')
print(df.shape)  # (1000000, 48)
```

### Sample Analysis
```python
# Skin health by location
df.groupby('location')['skin_health_score'].mean()

# Concerns by age
df.groupby('age')['primary_concern'].value_counts()

# Budget vs routine complexity
pd.crosstab(df['skincare_budget'], df['current_routine'])
```

### Train Model
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Encode + Split + Train
le = LabelEncoder()
X = df[['age', 'skin_health_score']]  # Add more features
y = le.fit_transform(df['rec_cleanser'])

# ... train_test_split, fit, predict
```

---

## 🗂️ Feature Categories

### Input Features (Use for prediction)

**Demographics**: `age`, `location`, `occupation`, `income_bracket`

**Skin Profile**: `skin_type`, `skin_tone`, `primary_concern`, `allergies`

**Lifestyle**: `sleep_hours`, `water_intake`, `stress_level`, `exercise_frequency`, `diet_type`, `junk_food_frequency`

**Environment**: `sun_exposure`, `pollution_exposure`, `area_pollution`, `area_altitude`

**Current Habits**: `current_routine`, `skincare_budget`, `product_preference`

**Hormonal**: `menstrual_skin_changes`, `hormonal_condition`

### Output Features (Target variables)

**Recommendations**: `rec_cleanser`, `rec_serum`, `rec_moisturizer`, `rec_sunscreen`, `rec_treatment`

**Scores**: `skin_health_score`, `predicted_satisfaction`, `engagement_probability`

---

## 🎨 Key Mappings

### Cleanser → Skin Type
| Skin Type | Recommended |
|-----------|-------------|
| Oily | Gel, Salicylic, Charcoal |
| Dry | Cream, Milk, Oil |
| Sensitive | Gentle Foam, Micellar |
| Combination | Foam, Gel |
| Normal | Gentle, Cream |

### Serum → Concern
| Concern | Recommended |
|---------|-------------|
| Acne | Niacinamide, Salicylic |
| Dark Spots | Vitamin C, Alpha Arbutin |
| Dullness | Vitamin C, AHA |
| Dryness | Hyaluronic Acid |
| Sensitivity | Centella, Aloe |

### Routine Type → Budget
| Budget (NPR) | Routine |
|--------------|---------|
| Under 500 | Minimal (2-3 products) |
| 500-1000 | Basic (3-4 products) |
| 1000-2000 | Standard (4-5 products) |
| 2000-3500 | Standard |
| 3500-5000 | Comprehensive |
| 5000+ | Advanced (6+ products) |

---

## 📍 Location Reference

| Municipality | Pollution | Altitude | % Users |
|--------------|-----------|----------|---------|
| Kathmandu Metro | High | 1350m | 35% |
| Lalitpur Metro | Moderate | 1350m | 20% |
| Bhaktapur | Moderate | 1400m | 10% |
| Kirtipur | Moderate | 1450m | 5% |
| Tokha | Low | 1500m | 4% |
| Chandragiri | Low | 1600m | 4% |
| Others | Varies | 1350-1550m | 22% |

---

## 🧮 Skin Health Score Formula

```
Base Score: 50

Modifiers:
  Sleep:      7-8 hrs (+5)  →  4-5 hrs (-10)
  Water:      3-4L (+8)     →  <1L (-10)
  Stress:     Low (+8)      →  Very High (-15)
  Exercise:   Daily (+10)   →  Never (-8)
  Junk Food:  Never (+10)   →  Daily (-12)
  Sunscreen:  Daily (+10)   →  Never (-10)
  Pollution:  Low (+5)      →  Very High (-15)

Range: 0-100 (clipped)
```

---

## 📈 Model Ideas

| Task | Algorithm | Target |
|------|-----------|--------|
| Product Rec | Random Forest | rec_cleanser/serum/etc |
| Health Score | XGBoost | skin_health_score |
| Segmentation | K-Means | Cluster users |
| Engagement | Logistic Reg | engagement_probability |
| Multi-output | Neural Net | All recommendations |

---

## ⚠️ Quick Notes

- **Synthetic data** - For training/testing only
- **Nepal-specific** - Kathmandu Valley context
- **Female 18-26** - Age/gender specific
- **Encode categoricals** - Use LabelEncoder or OneHot
- **Handle multi-values** - `;` separated (secondary_concerns, ingredients)

---

## 🚀 Recommended Workflow

1. **Load** → Read CSV, check shape
2. **Explore** → Value counts, distributions
3. **Clean** → Handle multi-value fields
4. **Encode** → Transform categoricals
5. **Split** → Train/test/validation
6. **Train** → Start with Random Forest
7. **Evaluate** → Accuracy, F1, confusion matrix
8. **Iterate** → Feature engineering, hypertuning

---

**Happy modeling! 🧴✨**
