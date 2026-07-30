"""
Data loading, preprocessing, model training, and utility functions for FarmIQ.
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')


def load_data(filepath=None):
    """Load the crop yield dataset."""
    if filepath is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(base_dir, 'data', 'raw', 'crop_yield_dataset.csv')
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}. Please run create_dataset.py first.")
    
    df = pd.read_csv(filepath)
    print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def preprocess_data(df):
    """Encode categorical crop names and prepare features/target."""
    df_processed = df.copy()
    le = LabelEncoder()
    df_processed['Crop_Encoded'] = le.fit_transform(df_processed['Crop'])
    
    feature_cols = ['N', 'P', 'K', 'pH', 'Temperature', 'Humidity', 'Rainfall', 'Crop_Encoded']
    X = df_processed[feature_cols]
    y = df_processed['Yield_tons_per_hectare']
    return X, y, le


def scale_features(X_train, X_test):
    """Scale features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def get_train_test_split(df, test_size=0.2, random_state=42):
    """Split data into train/test sets with scaling."""
    X, y, le = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, le


class CropYieldModel:
    """Train and evaluate multiple models for crop yield prediction."""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = -np.inf
    
    def train_all(self, X_train, y_train):
        """Train Random Forest, XGBoost, Gradient Boosting, and Ensemble."""
        print("=" * 50)
        print("Training Models...")
        print("=" * 50)
        
        self.models['Random Forest'] = RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42
        )
        self.models['XGBoost'] = XGBRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
        )
        self.models['Gradient Boosting'] = GradientBoostingRegressor(
            n_estimators=100, max_depth=5, random_state=42
        )
        
        estimators = [
            ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
            ('xgb', XGBRegressor(n_estimators=100, random_state=42)),
            ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42))
        ]
        self.models['Ensemble'] = VotingRegressor(estimators)
        
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            train_pred = model.predict(X_train)
            train_r2 = r2_score(y_train, train_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            print(f"✅ {name:20s} | R²: {train_r2:.4f} | RMSE: {train_rmse:.4f}")
            
            if train_r2 > self.best_score:
                self.best_score = train_r2
                self.best_model = model
                self.best_model_name = name
        
        print("=" * 50)
        print(f"🏆 Best Model: {self.best_model_name} with R²: {self.best_score:.4f}")
        return self.models
    
    def evaluate(self, X_test, y_test):
        """Evaluate all models on test data."""
        results = []
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            results.append({
                'Model': name,
                'R² Score': round(r2, 4),
                'RMSE': round(rmse, 4),
                'MAE': round(mae, 4)
            })
        return pd.DataFrame(results).sort_values('R² Score', ascending=False)
    
    def save_best_model(self, filepath='models/crop_yield_model.pkl'):
        """Save the best model to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.best_model, filepath)
        print(f"✅ Best model saved to {filepath}")
    
    def load_model(self, filepath='models/crop_yield_model.pkl'):
        """Load a saved model from disk."""
        self.best_model = joblib.load(filepath)
        print(f"✅ Model loaded from {filepath}")
        return self.best_model


class ProfitCalculator:
    """Calculate profit, costs, and ROI for different crops."""
    
    crop_prices = {
        'Rice': 22000, 'Wheat': 24000, 'Maize': 20000, 'Sugarcane': 3500,
        'Cotton': 60000, 'Groundnut': 55000, 'Soybean': 45000,
        'Potato': 15000, 'Onion': 18000, 'Tomato': 12000
    }
    
    input_costs = {
        'Rice': {'seeds': 4000, 'fertilizer': 5000, 'pesticides': 3000, 'labor': 8000, 'irrigation': 2000},
        'Wheat': {'seeds': 3500, 'fertilizer': 4500, 'pesticides': 2500, 'labor': 7000, 'irrigation': 1500},
        'Maize': {'seeds': 3000, 'fertilizer': 4000, 'pesticides': 2000, 'labor': 6000, 'irrigation': 1500},
        'Sugarcane': {'seeds': 8000, 'fertilizer': 10000, 'pesticides': 5000, 'labor': 15000, 'irrigation': 3000},
        'Cotton': {'seeds': 5000, 'fertilizer': 6000, 'pesticides': 4000, 'labor': 10000, 'irrigation': 2000},
        'Groundnut': {'seeds': 4000, 'fertilizer': 3500, 'pesticides': 2000, 'labor': 5000, 'irrigation': 1000},
        'Soybean': {'seeds': 3500, 'fertilizer': 4000, 'pesticides': 2500, 'labor': 6000, 'irrigation': 1200},
        'Potato': {'seeds': 6000, 'fertilizer': 5000, 'pesticides': 3000, 'labor': 8000, 'irrigation': 1500},
        'Onion': {'seeds': 5000, 'fertilizer': 4500, 'pesticides': 3000, 'labor': 7000, 'irrigation': 1200},
        'Tomato': {'seeds': 4000, 'fertilizer': 5000, 'pesticides': 3500, 'labor': 8000, 'irrigation': 1500}
    }
    
    def __init__(self, land_area=1):
        self.land_area = land_area
    
    def calculate_total_cost(self, crop):
        """Calculate total input cost for a crop."""
        costs = self.input_costs.get(crop, {'seeds': 4000, 'fertilizer': 4000, 'pesticides': 2000, 'labor': 5000, 'irrigation': 1000})
        return sum(costs.values()) * self.land_area
    
    def calculate_profit(self, crop, yield_tons):
        """Calculate revenue, cost, profit, and ROI."""
        price = self.crop_prices.get(crop, 20000)
        total_yield = yield_tons * self.land_area
        revenue = total_yield * price
        cost = self.calculate_total_cost(crop)
        profit = revenue - cost
        return {
            'crop': crop,
            'yield_tons_per_hectare': yield_tons,
            'total_yield_tons': total_yield,
            'market_price': price,
            'revenue': revenue,
            'total_cost': cost,
            'profit': profit,
            'roi_percentage': (profit / cost) * 100 if cost > 0 else 0
        }
    
    @staticmethod
    def get_fertilizer_recommendation(N, P, K, pH):
        """Generate fertilizer recommendations based on soil parameters."""
        recs = []
        if N < 80:
            recs.append("🌱 Add Nitrogen: Apply Urea or DAP")
        elif N > 200:
            recs.append("⚠️ Nitrogen high: Reduce application")
        else:
            recs.append("✅ Nitrogen level optimal")
        
        if P < 40:
            recs.append("🌱 Add Phosphorus: Apply SSP or DAP")
        elif P > 100:
            recs.append("⚠️ Phosphorus high: Reduce application")
        else:
            recs.append("✅ Phosphorus level optimal")
        
        if K < 40:
            recs.append("🌱 Add Potassium: Apply MOP")
        elif K > 150:
            recs.append("⚠️ Potassium high: Reduce application")
        else:
            recs.append("✅ Potassium level optimal")
        
        if pH < 5.5:
            recs.append("⚖️ Soil too acidic: Apply lime")
        elif pH > 7.5:
            recs.append("⚖️ Soil too alkaline: Apply gypsum")
        else:
            recs.append("✅ Soil pH optimal")
        
        return recs
