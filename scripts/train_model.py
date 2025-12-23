import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

# Load cleaned dataset
data = pd.read_csv("cleaned_skincare_dataset.csv")

# Select input and output columns
input_columns = ['age','skin_type','skin_tone','primary_concern','occupation','education_level','income_bracket',
                 'sleep_hours','daily_water_intake','stress_level','exercise_frequency','diet_type','junk_food_frequency',
                 'dairy_intake','sugar_intake','sun_exposure','sunscreen_usage','pollution_exposure','commute_type',
                 'current_routine','skincare_budget','allergies','makeup_frequency']
output_columns = ['rec_moisturizer','rec_serum','rec_sunscreen','rec_treatment',
                  'rec_ingredients','avoid_ingredients','rec_routine_type']

# Take a smaller random sample for fast training
data_sample = data.sample(n=20000, random_state=42)  # 20k rows

X = data_sample[input_columns]
y = data_sample[output_columns]

# Encode categorical columns
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

for col in y.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    y[col] = le.fit_transform(y[col])

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Multi-output Random Forest with fewer trees and single-threaded to save memory
forest = RandomForestClassifier(n_estimators=30, random_state=42, verbose=1, n_jobs=1)
multi_target_forest = MultiOutputClassifier(forest)

print("Training started on sample dataset...")
multi_target_forest.fit(X_train, y_train)
print("Training finished!")

# Test a sample prediction
sample_input = X_test.iloc[0:1]
predicted = multi_target_forest.predict(sample_input)
print("Sample prediction:", predicted)
