# 🍬 Nassau Candy — Factory Optimization System
## Step-by-step setup guide (read this first!)

---

## YOUR FOLDER STRUCTURE (set this up exactly like this)

```
nassau_candy/
│
├── data/
│   └── Nassau_Candy_Distributor.csv   ← PUT YOUR CSV FILE HERE
│
├── models/                            ← leave empty, created automatically
│
├── 01_explore.py                      ← paste file here
├── 02_model.py                        ← paste file here
├── app.py                             ← paste file here
├── requirements.txt                   ← paste file here
└── README.md                          ← this file
```

---

## STEP 1 — Install packages (do this ONCE)

Open the terminal in VS Code (top menu → Terminal → New Terminal) and run:

```
pip install -r requirements.txt
```

Wait for it to finish (2–5 minutes). You will see a lot of text — that is normal.

---

## STEP 2 — Explore your data

Right-click `01_explore.py` → "Run Python File in Terminal"

What it does:
- Prints a summary of the dataset (rows, columns, stats)
- Shows factory performance (sales, profit, lead times)
- Saves charts to data/exploration_charts.png

---

## STEP 3 — Train the machine learning model

Right-click `02_model.py` → "Run Python File in Terminal"

What it does:
- Cleans and prepares the data
- Trains 3 models (Linear Regression, Random Forest, Gradient Boosting)
- Picks the best one automatically
- Saves the model to models/lead_time_model.pkl
- Saves processed data to data/processed_data.csv

You will see accuracy scores printed. Look for R² — anything above 0.5 is acceptable.

---

## STEP 4 — Launch the dashboard

In the terminal, run:

```
streamlit run app.py
```

A browser window will open automatically at http://localhost:8501

The dashboard has 4 tabs:
1. Overview      — charts, KPIs, factory map
2. Simulator     — predict lead time for any product + factory combo
3. What-If       — compare all products current vs best factory
4. Recommendations — ranked list of reassignments with risk panel

---

## TROUBLESHOOTING

Problem: "ModuleNotFoundError"
Fix: Run `pip install -r requirements.txt` again

Problem: "FileNotFoundError: data/Nassau_Candy_Distributor.csv"
Fix: Make sure your CSV is inside the `data/` subfolder, not in the root

Problem: Dashboard says "Run 02_model.py first"
Fix: Run 02_model.py and wait for it to finish, then refresh the dashboard

Problem: "streamlit is not recognized"
Fix: Run `pip install streamlit` then try again

---

## PROJECT SUMMARY

- Dataset:   10,194 orders across 15 candy products and 5 factories
- Goal:      Predict shipping lead time and find optimal factory assignments
- ML Model:  Random Forest Regressor (best performer)
- Dashboard: Streamlit web app with 4 interactive modules
- KPIs:      Lead Time Reduction, Profit Impact, Recommendation Coverage
