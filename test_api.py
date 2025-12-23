import requests

url = "http://127.0.0.1:5000/predict"

data = {
    "age": 25,
    "skin_type": "Oily",
    "skin_tone": "Type III",
    "primary_concern": "Acne/Breakouts",
    "secondary_concerns": "Blackheads",
    "occupation": "Student",
    "sleep_hours": 7,
    "daily_water_intake": 2,
    "stress_level": 5,
    "exercise_frequency": 3,
    "diet_type": "Vegetarian",
    "junk_food_frequency": 1,
    "dairy_intake": 2,
    "sugar_intake": 1,
    "sun_exposure": "Moderate",
    "sunscreen_usage": "Daily",
    "allergies": "Unknown"
}

response = requests.post(url, json=data)
print(response.json())
