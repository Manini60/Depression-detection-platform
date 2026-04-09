# MindScan — Student Depression Detection

A professional Flask web app using a Bayesian Network to predict student depression risk.

## Setup (macOS + VS Code + venv)

### 1. Create virtual environment
```bash
cd depression_app
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note:** pgmpy may take a few minutes to install. If you see errors with pgmpy,
> try: `pip install pgmpy --no-deps` then install dependencies individually.

### 3. Ensure the dataset is in the project root
```
depression_app/
├── student_depression_dataset.csv   ← must be here
├── app.py
├── model.py
├── preprocess.py
├── requirements.txt
├── static/
│   └── style.css
└── templates/
    ├── index.html
    ├── predict.html
    ├── model.html
    ├── metrics.html
    └── about.html
```

### 4. Run the app
```bash
python app.py
```

Open: http://127.0.0.1:5000

---

## Pages

| Route      | Description                              |
|------------|------------------------------------------|
| `/`        | Dashboard with dataset charts            |
| `/predict` | Depression risk assessment form          |
| `/model`   | Bayesian Network structure               |
| `/metrics` | Confusion matrix & model evaluation      |
| `/about`   | Project info & how it works              |

---

## Features Added (vs original)
- ✅ Interactive Chart.js charts on dashboard (donut, bar, horizontal bar)
- ✅ Personalised recommendations based on input
- ✅ Probability bar chart instead of just text
- ✅ Professional dark UI with Space Mono + DM Sans
- ✅ Animated navbar with live indicator
- ✅ Responsive layout for all screen sizes
- ✅ Metrics legend explaining TP/TN/FP/FN
- ✅ About page with timeline & tech stack badges
- ✅ Disclaimer card
- ✅ JSON API endpoint at `/api/quick-predict`
- ✅ Risk badge (HIGH / MEDIUM / LOW) with colour coding
