# F1 Project – Human Power vs Horse Power

This project analyzes Formula 1 race results to understand **which features most strongly influence the points a driver earns in a race**.  

The focus is not on predicting final positions, but on explaining **which driver, team, and race-context characteristics drive points per race**.

---

## Business Question

> **Which features dominate the points earned in a Formula 1 race?**

Using historical race data, the project examines whether **driver-related factors** (skill, form, age) or **constructor / car-related factors** (team performance, car strength) play a more important role, and how this changes across seasons and individual races.

---

## Repository Structure

- `f1_data/`  
  - All CSV files used in the project.  
  - `Formula 1_Github version.ipynb` – main analysis notebook (data preparation, modeling, SHAP analysis)
- `.gitignore`  
  - Ensures local / system files and secrets (e.g., Kaggle API credentials) are not committed.

> **Note:** Kaggle API keys are stored locally in `~/.kaggle/kaggle.json` and are never included in this repository.

---

## Data

- **Source:** Kaggle – *Formula 1 World Championship (1950–2024)*  https://www.kaggle.com/datasets/muhammadehsan02/formula-1-world-championship-history-1950-2024/data
  Dataset: `muhammadehsan02/formula-1-world-championship-history-1950-2024`
- **Scope used in this project:** last **15 seasons**  (2010 - 2024 )
- Data is split into several tables (results, drivers, teams, rankings, schedule, etc.) and combined into a modeling dataset.

All CSV files used in this project are stored under:

- `f1_data/`

---

## Target & Features

**Target variable**

- `points` – points earned by a driver in a specific race.

**Input feature groups**

1. **Driver-related**
   - `driver_ranking_points` – driver’s season points before the race  
   - `driver_ranking_wins` – driver’s season wins  
   - `driver_age` – age of the driver at race date  
   - `driver_nationality`

2. **Constructor / team-related**
   - `constructor_ranking_points` – team’s season points  
   - `constructor_ranking_wins` – team’s season wins  
   - `team_name`  
   - `team_nationality`

3. **Race context**
   - `race_year`  
   - `race_month`

**Intentionally dropped features (to avoid target leakage)**

These variables were removed because they directly encode race outcome or technical failure, and would make the model unrealistically optimistic:

- `positionOrder` – final race position (almost a direct proxy for points)  
- `driver_ranking_position`  
- `constructor_ranking_position`  
- `statusId` – technical DNF / accident / disqualification codes  
- `qualifying_position&grid` – in some model versions removed when highly correlated with outcome

---

## Modeling Approach

Two tree-based models were used:

### 1. Random Forest (Baseline)

- Used as an initial benchmark model.  
- Captures non-linear relationships between features and race points.

**Performance:**

- **Train RMSE:** 1.80  
- **Test RMSE:** 4.83  
- **Train R²:** 0.94  
- **Test R²:** 0.54  

These results show that the Random Forest fits the training data very well but does not generalize as strongly to unseen races (clear train–test gap), indicating overfitting.

---

### 2. XGBoost (Main Model)

- Gradient-boosted trees model used as the primary model.  
- Hyperparameters tuned with cross-validation (RandomizedSearchCV).

**Best parameters (simplified):**

- `n_estimators = 100`  
- `max_depth = 5`  
- `learning_rate ≈ 0.042`  
- `subsample = 0.6`  
- `colsample_bytree = 0.7`  
- `reg_alpha = 0.5`, `reg_lambda = 2`

**Performance:**

- **CV RMSE (3-fold):** 4.59  
- **Train RMSE:** 4.01  
- **Test RMSE:** 4.36  
- **Train R²:** 0.68  
- **Test R²:** 0.63  

Compared to the Random Forest, XGBoost shows **better generalisation**: the train–test gap is smaller and the test R² is higher, making it the preferred model for explaining which features drive race points.

## Explainability – SHAP Analysis

To understand **why** the model makes its predictions, SHAP (SHapley Additive exPlanations) values were calculated on the XGBoost model:

- **Global (season-level) view:**  
  Across the full dataset, the model relies heavily on **constructor-related features**, especially `constructor_ranking_wins`.  
  This reflects the global trend in F1: overall car and team performance have a very strong impact on race outcomes.

- **Local (single-race) view:**  
  For specific races, especially unusual or difficult conditions, SHAP shows that **driver-related features** (`driver_ranking_points`) can dominate the prediction for that particular race.  
  In other words, while the car tends to dominate over a full season, individual races can be strongly influenced by driver form and characteristics.

Overall, the explainability analysis provides a balanced narrative: **“horse power” (constructor performance) dominates globally, but “human power” (driver performance) often decides individual races.**

---
## Fairness analysis: 

A team-level fairness check was performed using a bias gap metric (mean_pred − mean_true) to see whether XGBoost systematically over- or under-predicts points for certain constructors. The model generally preserves the overall ranking of top teams, but shows a slight under-prediction tendency for leading teams such as Mercedes, Red Bull and Ferrari, and a mild over-prediction for lower-performing teams like Haas and Alfa Romeo.

