
# 🌀 Multiverse Victory Predictor

Inspired by **Doctor Strange's 14,000,605 simulations** in *Avengers: Infinity War*, this machine learning project predicts whether a timeline results in **Victory** or **Defeat** based on various strategic parameters.

---

## 🚀 Live Demo

🔗 Run locally with Streamlit:
```bash
streamlit run streamlit_app.py
```

---

## 📁 Repository Structure

```
multiverse-victory-predictor/
├── multiverse_data.csv              # Dataset (5000 simulated timelines)
├── model_training.ipynb             # Jupyter Notebook for model training
├── multiverse_logistic.pkl          # Trained Logistic Regression model
├── multiverse_random_forest.pkl     # Trained Random Forest model
├── streamlit_app.py                 # Streamlit UI (Dark Theme)
├── requirements.txt                 # Python dependencies
├── assets/
│   └── pie_chart.png                # Victory vs Defeat Pie Chart
└── README.md                        # This file
```

---

## 🧠 Features Used

The model uses 18+ simulation parameters such as:

- `team_strength`
- `enemy_strength`
- `team_coordination`
- `intel_accuracy`
- `diversion_success_rate`
- `enemy_stone_count`
- `universe_variability`
- Categorical: `has_time_stone`, `terrain_advantage`, `enemy_mind_state`, etc.

---

## 🧪 ML Models Used

| Model               | Purpose                     |
|--------------------|-----------------------------|
| Logistic Regression| Baseline for binary outcome |
| Random Forest       | Robust performance model    |

✅ Both models are trained using `Pipeline` (preprocessing + model), so they're ready for use without separate preprocessing.

---

## 📸 UI Preview

![Pie Chart](assets/pie_chart.png)

---

## 📦 Installation

1. Clone this repo:
```bash
git clone https://github.com/yourusername/multiverse-victory-predictor.git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Launch app:
```bash
streamlit run streamlit_app.py
```

---

## 📜 References

- Marvel’s *Avengers: Infinity War* – Doctor Strange sees 14,000,605 timelines
- Machine Learning: scikit-learn, pandas, matplotlib, streamlit

---

## 👨‍💻 Author

**Bhaskar Gupta**  
📬 [LinkedIn Post Coming Soon!]

---

## 📥 Downloads

- 📘 [`model_training.ipynb`](sandbox:/mnt/data/model_training.ipynb)
- 🌌 [`streamlit_app.py`](sandbox:/mnt/data/streamlit_app.py)

Enjoy exploring the multiverse!
