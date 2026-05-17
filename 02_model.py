# ============================================================
#  02_model.py  —  Nassau Candy: Machine Learning Model
#  Trains a Random Forest to predict shipping lead time.
#  HOW TO RUN: right-click this file → "Run Python File in Terminal"
#  Run AFTER 01_explore.py
# ============================================================

import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ── 0. Create folders if they don't exist ────────────────────
os.makedirs('models', exist_ok=True)
os.makedirs('data',   exist_ok=True)

# ── 1. Load & clean data ──────────────────────────────────────
print("Loading data...")
df = pd.read_csv('data/Nassau_Candy_Distributor.csv')

df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
df['Ship Date']  = pd.to_datetime(df['Ship Date'],  dayfirst=True)
df['Lead Time']  = (df['Ship Date'] - df['Order Date']).dt.days

# Add factory mapping
PRODUCT_FACTORY = {
    'Wonka Bar - Nutty Crunch Surprise':    "Lot's O' Nuts",
    'Wonka Bar - Fudge Mallows':            "Lot's O' Nuts",
    'Wonka Bar -Scrumdiddlyumptious':       "Lot's O' Nuts",
    'Wonka Bar - Milk Chocolate':           "Wicked Choccy's",
    'Wonka Bar - Triple Dazzle Caramel':    "Wicked Choccy's",
    'Laffy Taffy':                          'Sugar Shack',
    'SweeTARTS':                            'Sugar Shack',
    'Nerds':                                'Sugar Shack',
    'Fun Dip':                              'Sugar Shack',
    'Fizzy Lifting Drinks':                 'Sugar Shack',
    'Everlasting Gobstopper':               'Secret Factory',
    'Hair Toffee':                          'The Other Factory',
    'Lickable Wallpaper':                   'Secret Factory',
    'Wonka Gum':                            'Secret Factory',
    'Kazookles':                            'The Other Factory',
}
df['Factory'] = df['Product Name'].map(PRODUCT_FACTORY)

# Remove rows with missing values
df = df.dropna()

# ── 2. Feature engineering ────────────────────────────────────
# Add profit margin as a feature
df['Margin_Pct'] = (df['Gross Profit'] / df['Sales']) * 100

# Encode categorical columns as numbers (ML needs numbers, not text)
label_encoders = {}
for col in ['Region', 'Ship Mode', 'Factory', 'Division']:
    le = LabelEncoder()
    df[col + '_enc'] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le          # save so we can decode later

# ── 3. Define features (X) and target (y) ─────────────────────
FEATURES = [
    'Region_enc',
    'Ship Mode_enc',
    'Factory_enc',
    'Division_enc',
    'Sales',
    'Units',
    'Gross Profit',
    'Cost',
    'Margin_Pct',
]

X = df[FEATURES]
y = df['Lead Time']

print(f"Dataset: {len(df):,} rows | {len(FEATURES)} features")
print(f"Target (Lead Time) — min: {y.min()} days, max: {y.max()} days, mean: {y.mean():.1f} days\n")

# ── 4. Train / test split ─────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training rows: {len(X_train):,}  |  Testing rows: {len(X_test):,}\n")

# ── 5. Train 3 models and compare ────────────────────────────
models = {
    'Linear Regression (baseline)': LinearRegression(),
    'Random Forest':                 RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1),
    'Gradient Boosting':             GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, random_state=42),
}

results = {}
print("Training models...")
print("-" * 55)

for name, m in models.items():
    m.fit(X_train, y_train)
    preds = m.predict(X_test)

    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)

    results[name] = {'model': m, 'MAE': mae, 'RMSE': rmse, 'R2': r2}

    print(f"  {name}")
    print(f"    MAE  = {mae:.2f} days   (lower is better)")
    print(f"    RMSE = {rmse:.2f} days   (lower is better)")
    print(f"    R²   = {r2:.4f}         (closer to 1.0 is better)")
    print()

# ── 6. Pick best model (highest R²) ──────────────────────────
best_name = max(results, key=lambda k: results[k]['R2'])
best_model = results[best_name]['model']
print(f"✅ Best model: {best_name}  (R² = {results[best_name]['R2']:.4f})\n")

# ── 7. Feature importance chart ───────────────────────────────
if hasattr(best_model, 'feature_importances_'):
    importances = pd.Series(best_model.feature_importances_, index=FEATURES).sort_values()
    plt.figure(figsize=(9, 5))
    importances.plot(kind='barh', color='steelblue')
    plt.title(f'Feature Importance — {best_name}')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('data/feature_importance.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("📊 Feature importance chart saved to data/feature_importance.png")

# ── 8. Save everything ────────────────────────────────────────
joblib.dump(best_model,     'models/lead_time_model.pkl')
joblib.dump(label_encoders, 'models/label_encoders.pkl')
joblib.dump(FEATURES,       'models/feature_names.pkl')

# Save a clean version of the processed data for the dashboard
df.to_csv('data/processed_data.csv', index=False)

print()
print("=" * 55)
print("  FILES SAVED:")
print("  models/lead_time_model.pkl   ← trained model")
print("  models/label_encoders.pkl    ← text→number converters")
print("  models/feature_names.pkl     ← feature list")
print("  data/processed_data.csv      ← cleaned dataset")
print("=" * 55)
print()
print("✅ Model training complete! Now run:  streamlit run app.py")
