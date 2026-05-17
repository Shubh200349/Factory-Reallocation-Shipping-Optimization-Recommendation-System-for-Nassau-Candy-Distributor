# ============================================================
#  01_explore.py  —  Nassau Candy: Data Exploration
#  Run this first to understand your dataset.
#  HOW TO RUN: right-click this file → "Run Python File in Terminal"
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load data ─────────────────────────────────────────────
df = pd.read_csv('data/Nassau_Candy_Distributor.csv')

# Fix dates  (format is DD-MM-YYYY in this dataset)
df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
df['Ship Date']  = pd.to_datetime(df['Ship Date'],  dayfirst=True)

# Create lead time in days
df['Lead Time'] = (df['Ship Date'] - df['Order Date']).dt.days

# ── 2. Add factory column using the product-factory mapping ──
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

# ── 3. Basic overview ────────────────────────────────────────
print("=" * 55)
print("  NASSAU CANDY — DATASET OVERVIEW")
print("=" * 55)
print(f"  Total rows    : {len(df):,}")
print(f"  Total columns : {df.shape[1]}")
print(f"  Date range    : {df['Order Date'].min().date()}  →  {df['Order Date'].max().date()}")
print(f"  Missing values: {df.isnull().sum().sum()}")
print()

print("── Regions ──────────────────────────────────────────────")
print(df['Region'].value_counts().to_string())
print()

print("── Ship Modes ───────────────────────────────────────────")
print(df['Ship Mode'].value_counts().to_string())
print()

print("── Divisions ────────────────────────────────────────────")
print(df['Division'].value_counts().to_string())
print()

print("── Lead Time (days) ─────────────────────────────────────")
print(df['Lead Time'].describe().round(1).to_string())
print()

print("── Sales & Profit Summary ───────────────────────────────")
print(df[['Sales', 'Gross Profit', 'Cost', 'Units']].describe().round(2).to_string())
print()

# ── 4. Factory-level summary ─────────────────────────────────
print("── Factory Performance ──────────────────────────────────")
factory_summary = df.groupby('Factory').agg(
    Orders        = ('Row ID',       'count'),
    Avg_Lead_Days = ('Lead Time',    'mean'),
    Total_Sales   = ('Sales',        'sum'),
    Total_Profit  = ('Gross Profit', 'sum'),
    Avg_Margin    = ('Gross Profit', lambda x: (x / df.loc[x.index, 'Sales']).mean() * 100)
).round(2)
print(factory_summary.to_string())
print()

# ── 5. Region × Ship Mode cross-tab ──────────────────────────
print("── Orders by Region × Ship Mode ─────────────────────────")
print(pd.crosstab(df['Region'], df['Ship Mode']).to_string())
print()

# ── 6. Charts ────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Nassau Candy — Exploratory Analysis', fontsize=16, fontweight='bold')

# Chart 1: Sales by Factory
factory_sales = df.groupby('Factory')['Sales'].sum().sort_values(ascending=True)
axes[0, 0].barh(factory_sales.index, factory_sales.values, color='steelblue')
axes[0, 0].set_title('Total Sales by Factory')
axes[0, 0].set_xlabel('Total Sales ($)')

# Chart 2: Gross Profit by Factory
factory_profit = df.groupby('Factory')['Gross Profit'].sum().sort_values(ascending=True)
axes[0, 1].barh(factory_profit.index, factory_profit.values, color='seagreen')
axes[0, 1].set_title('Total Gross Profit by Factory')
axes[0, 1].set_xlabel('Gross Profit ($)')

# Chart 3: Lead Time by Region
lead_region = df.groupby('Region')['Lead Time'].mean().sort_values()
axes[0, 2].bar(lead_region.index, lead_region.values, color='coral')
axes[0, 2].set_title('Avg Lead Time by Region (days)')
axes[0, 2].set_ylabel('Days')

# Chart 4: Lead Time by Ship Mode
lead_ship = df.groupby('Ship Mode')['Lead Time'].mean().sort_values()
axes[1, 0].bar(lead_ship.index, lead_ship.values, color='mediumpurple')
axes[1, 0].set_title('Avg Lead Time by Ship Mode (days)')
axes[1, 0].set_ylabel('Days')
axes[1, 0].tick_params(axis='x', rotation=15)

# Chart 5: Sales distribution
axes[1, 1].hist(df['Sales'], bins=40, color='steelblue', edgecolor='white')
axes[1, 1].set_title('Sales Distribution')
axes[1, 1].set_xlabel('Sales ($)')

# Chart 6: Profit margin by Division
df['Margin'] = df['Gross Profit'] / df['Sales'] * 100
margin_div = df.groupby('Division')['Margin'].mean().sort_values()
axes[1, 2].bar(margin_div.index, margin_div.values, color='darkorange')
axes[1, 2].set_title('Avg Profit Margin by Division (%)')
axes[1, 2].set_ylabel('Margin %')

plt.tight_layout()
plt.savefig('data/exploration_charts.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Charts saved to data/exploration_charts.png")
print("✅ Exploration complete!")
