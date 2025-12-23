TARGET_COLUMNS = [
    "rec_moisturizer",
    "rec_serum",
    "rec_sunscreen",
    "rec_treatment"
]

TEXT_COLUMNS = [
    "rec_ingredients",
    "avoid_ingredients",
    "rec_routine_type"
]

FEATURE_COLUMNS = [
    "age",
    "skin_type",
    "skin_tone",
    "primary_concern",
    "secondary_concerns",
    "occupation",
    "sleep_hours",
    "daily_water_intake",
    "stress_level",
    "exercise_frequency",
    "diet_type",
    "junk_food_frequency",
    "dairy_intake",
    "sugar_intake",
    "sun_exposure",
    "sunscreen_usage",
    "allergies"
]

print("FEATURES:", FEATURE_COLUMNS)
print("TARGETS:", TARGET_COLUMNS)
