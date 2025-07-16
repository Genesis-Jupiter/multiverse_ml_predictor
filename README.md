# 🔮 Multiverse ML Predictor

Inspired by Doctor Strange's 14,000,605 simulations in *Avengers: Infinity War*, this ML project attempts to simulate battle outcomes and predict rare victories using machine learning.

## 🎯 Goal
Predict whether a given battle simulation leads to **Victory** or **Defeat** using features like:
- Team Strength
- Enemy Power
- Use of Time Stone
- Strategic Complexity
- Sacrifice Possibility
- and more...

## 🧠 ML Approach

- 📊 Dataset: 5000-row custom simulation dataset (imbalanced — only 20% victories)
- 🔍 Models: Random Forest, Logistic Regression
- ⚖️ Imbalance Handling: Class weights + threshold tuning
- 📈 Metrics: Precision, Recall, F1-score, Confusion Matrix

## 🖼️ Visualizations

| Random Forest Confusion Matrix | Feature Importance |
|-------------------------------|--------------------|
| ![CM](plots/confusion_matrix_rf.png) | ![FI](plots/feature_importance.png) |

## 📁 Project Structure

notebook/ → Jupyter/Colab notebook
dataset/ → Simulation CSV
plots/ → Saved evaluation graphs

## 🚀 How to Use

1. Clone this repo
2. Open `notebook/multiverse_predictor.ipynb`
3. Run cells step-by-step:
   - Preprocess data
   - Train models
   - Tune thresholds
   - View visualizations

## 🛠️ Tools Used

- Python 3.10+
- Scikit-Learn
- Matplotlib
- Pandas, NumPy
- Google Colab

## 🧪 Dataset Info

File: `simulated_multiverse_dataset.csv`  
5000 rows × 18 features + target  
With:
- Outliers
- Missing values
- Inconsistent strings (`'???'`, `'unknown')

---

“Doctor Strange saw only one... but we found more. Machine Learning unlocks the multiverse.”
