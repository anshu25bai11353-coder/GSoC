"""
Generate synthetic crop yield dataset with realistic soil and weather parameters.
"""
import pandas as pd
import numpy as np
import random
import os

os.makedirs('data/raw', exist_ok=True)
np.random.seed(42)
random.seed(42)

# Optimal parameters for each crop
crops = {
    'Rice': {'N': 80, 'P': 40, 'K': 40, 'pH': 6.5, 'temp': 28, 'humidity': 80, 'rainfall': 200, 'yield_factor': 3.5},
    'Wheat': {'N': 120, 'P': 60, 'K': 60, 'pH': 7.0, 'temp': 20, 'humidity': 60, 'rainfall': 150, 'yield_factor': 3.0},
    'Maize': {'N': 150, 'P': 70, 'K': 50, 'pH': 6.8, 'temp': 25, 'humidity': 65, 'rainfall': 180, 'yield_factor': 4.0},
    'Sugarcane': {'N': 200, 'P': 100, 'K': 100, 'pH': 7.2, 'temp': 30, 'humidity': 75, 'rainfall': 250, 'yield_factor': 8.0},
    'Cotton': {'N': 100, 'P': 50, 'K': 70, 'pH': 7.0, 'temp': 28, 'humidity': 55, 'rainfall': 120, 'yield_factor': 2.5},
    'Groundnut': {'N': 30, 'P': 50, 'K': 40, 'pH': 6.2, 'temp': 27, 'humidity': 60, 'rainfall': 100, 'yield_factor': 1.8},
    'Soybean': {'N': 80, 'P': 60, 'K': 50, 'pH': 6.5, 'temp': 26, 'humidity': 70, 'rainfall': 130, 'yield_factor': 2.2},
    'Potato': {'N': 150, 'P': 80, 'K': 100, 'pH': 6.0, 'temp': 18, 'humidity': 75, 'rainfall': 150, 'yield_factor': 2.5},
    'Onion': {'N': 100, 'P': 50, 'K': 50, 'pH': 6.8, 'temp': 22, 'humidity': 65, 'rainfall': 100, 'yield_factor': 2.0},
    'Tomato': {'N': 120, 'P': 60, 'K': 80, 'pH': 6.5, 'temp': 24, 'humidity': 70, 'rainfall': 120, 'yield_factor': 3.0}
}

n_samples = 5000
data = []

for _ in range(n_samples):
    crop = random.choice(list(crops.keys()))
    optimal = crops[crop]
    
    # Add noise to parameters
    N = max(0, optimal['N'] + np.random.normal(0, 30))
    P = max(0, optimal['P'] + np.random.normal(0, 20))
    K = max(0, optimal['K'] + np.random.normal(0, 20))
    pH = optimal['pH'] + np.random.normal(0, 0.5)
    temperature = optimal['temp'] + np.random.normal(0, 5)
    humidity = optimal['humidity'] + np.random.normal(0, 10)
    rainfall = optimal['rainfall'] + np.random.normal(0, 50)
    
    # Calculate yield based on deviation from optimal
    distance = (
        abs(N - optimal['N']) / optimal['N'] * 0.3 +
        abs(P - optimal['P']) / optimal['P'] * 0.2 +
        abs(K - optimal['K']) / optimal['K'] * 0.2 +
        abs(pH - optimal['pH']) / optimal['pH'] * 0.1 +
        abs(temperature - optimal['temp']) / optimal['temp'] * 0.1 +
        abs(humidity - optimal['humidity']) / optimal['humidity'] * 0.1
    )
    
    # Yield decreases as distance increases
    yield_tons = optimal['yield_factor'] * (1 - distance * 0.15) + np.random.normal(0, 0.2)
    yield_tons = max(0.1, yield_tons)
    
    data.append({
        'Crop': crop,
        'N': round(N, 2),
        'P': round(P, 2),
        'K': round(K, 2),
        'pH': round(pH, 2),
        'Temperature': round(temperature, 2),
        'Humidity': round(humidity, 2),
        'Rainfall': round(rainfall, 2),
        'Yield_tons_per_hectare': round(yield_tons, 2)
    })

df = pd.DataFrame(data)
df.to_csv('data/raw/crop_yield_dataset.csv', index=False)
print(f"✅ Dataset generated with {len(df)} samples at data/raw/crop_yield_dataset.csv")
print(df.head())
