# MindScan — Student Depression Detection

A professional Flask web application that predicts student depression risk using a Bayesian Network model.

---

## 📌 Project Overview

This project is designed to analyze student-related factors and predict the likelihood of depression.
It combines Machine Learning with a web interface to provide an easy-to-use prediction system.

---

## ⚙️ Setup (macOS + VS Code + venv)

### 1. Create virtual environment

cd depression_app
python3 -m venv venv
source venv/bin/activate

### 2. Install dependencies

pip install -r requirements.txt

> Note: pgmpy may take time to install. If errors occur, install it separately.

---

## 📂 Project Structure

depression_app/
│── student_depression_dataset.csv
│── app.py
│── model.py
│── preprocess.py
│── requirements.txt
│
├── static/
│   └── style.css
│
├── templates/
└── index.html

---

## 🧠 Model Explanation

This project uses a **Bayesian Network** to predict depression.

A Bayesian Network is a probabilistic model that represents relationships between different variables and calculates the probability of an outcome.

The model considers various factors such as:

* Academic Pressure
* Work Pressure
* CGPA
* Sleep Duration
* Study Satisfaction
* Financial Stress
* Family History of Mental Illness
* Suicidal Thoughts

Based on these inputs, it calculates the probability of whether a student is likely to experience depression.

---

## 📊 Dataset

The dataset used in this project is a student depression dataset in CSV format.

It contains features such as:

* Gender, Age, City, Profession
* Academic Pressure and Work Pressure
* CGPA and Study Satisfaction
* Sleep Duration and Dietary Habits
* Work/Study Hours and Financial Stress
* Family History of Mental Illness
* Suicidal Thoughts

The target variable is:

* Depression (0 = No, 1 = Yes)

This dataset is used to train the Bayesian Network model for predicting depression risk.

---

## 📈 Results

The system predicts whether a student is likely to experience depression based on input data.

The Bayesian Network model calculates the probability of depression using multiple influencing factors and provides a final prediction:

* Depression
* No Depression

---


## 🚀 How to Run the Project

pip install -r requirements.txt
python app.py

Then open your browser and go to:
http://127.0.0.1:5000/

---

## 🔮 Future Scope

* Improve accuracy using advanced Machine Learning or Deep Learning models
* Add real-time data collection
* Develop a mobile application
* Integrate chatbot-based mental health support

---

## 🛠️ Technologies Used

* Python
* Flask
* Machine Learning
* Bayesian Networks (pgmpy)
* HTML, CSS

---

## 📌 Conclusion

This project demonstrates how Machine Learning can be used to identify mental health risks and provide early predictions for depression among students.
