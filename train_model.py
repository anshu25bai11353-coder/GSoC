"""
Train and save all ML models for crop yield prediction.
"""
import os
import joblib
from src.data_loader import load_data, get_train_test_split, CropYieldModel

os.makedirs('models', exist_ok=True)

print("📊 Loading data...")
df = load_data()

print("🔧 Preparing data...")
X_train, X_test, y_train, y_test, scaler, le = get_train_test_split(df)

print("🧠 Training models...")
model = CropYieldModel()
model.train_all(X_train, y_train)

print("\n📈 Evaluation Results:")
results = model.evaluate(X_test, y_test)
print(results.to_string(index=False))

model.save_best_model('models/crop_yield_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(le, 'models/label_encoder.pkl')

print("\n✅ Training complete! Files saved in 'models/' folder")
