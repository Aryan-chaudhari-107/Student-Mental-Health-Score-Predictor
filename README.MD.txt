# Student Mental Health Score Predictor

A machine learning project that predicts a student's **Mental Health Score (0–10)** from their social media habits, study time, sleep, physical activity and stress level. The trained model is served through a **FastAPI** backend and used by a simple **HTML/CSS/JavaScript** frontend.

> This is a learning project built on survey data. It is **not** a clinical or diagnostic tool.

## Features

- Regression model (Random Forest) trained on 5,000 student records
- Full scikit-learn pipeline (preprocessing and model saved together), so the API takes raw input
- `POST /predict` endpoint with Pydantic input validation
- Responsive frontend with a score gauge and clear error messages

## Tech Stack

| Area | Tools |
|---|---|
| Language | Python |
| Machine learning | scikit-learn, pandas, NumPy, joblib |
| Backend | FastAPI, Pydantic, Uvicorn |
| Frontend | HTML, CSS, JavaScript |
| Notebook | Jupyter / Google Colab, Matplotlib, Seaborn |

## Dataset

**Student Social Media and Mental Health Impact**: 5,000 records and 13 columns (12 input features plus the target `Mental_Health_Score`).

| Type | Columns |
|---|---|
| Demographics | Age, Gender, Country, Academic_Level |
| Social media | Most_Used_Platform, Purpose_Of_Use, Avg_Daily_Usage_Hours, Daily_Unlocks |
| Lifestyle | Study_Hours, Physical_Activity_Hours, Sleep_Hours_Per_Night, Stress_Level |
| Target | Mental_Health_Score |

Source: _add the dataset link here_

## Approach

**Data cleaning**
- Removed 2 duplicate rows.
- Clipped negative `Physical_Activity_Hours` values (minimum was -0.4) to 0.

**Feature engineering**
- `Country` has 111 unique values, which is too many to one-hot encode. It was grouped into 10 categories: the 9 most frequent countries plus `Other`.

**Preprocessing (`ColumnTransformer`)**
- `Study_Hours`: log transform (`log1p`) followed by standard scaling
- Other numeric columns: standard scaling
- `Stress_Level`: ordinal encoding (Low < Medium < High < Very High)
- Gender, Academic level, Platform, Purpose, Grouped country: one-hot encoding

**Modeling**
- 70/30 train-test split (`random_state=42`)
- Compared Linear Regression with Random Forest, and also tried `RandomizedSearchCV` tuning on the Random Forest.
- The saved model is the default Random Forest pipeline, which performed best on the test set.

## Results

| Model | Test R² | MAE | RMSE |
|---|---|---|---|
| Linear Regression | 0.740 | 0.536 | 0.676 |
| **Random Forest (default), used in the API** | **0.878** | **0.347** | **0.464** |
| Random Forest (tuned) | 0.865 | 0.369 | 0.487 |

The default Random Forest scores R² 0.98 on training data against 0.88 on test data, so it overfits somewhat. The tuned version did not improve test performance.

## Project Structure

```
mental-health-score-predictor/
├── main.py                   # FastAPI app (loads the model, exposes /predict)
├── Model/
│   └── Mental_Health_Model.pkl
├── Data/
│   └── Student Social Media And Mental Health Impact.csv
├── ML_Project.ipynb          # EDA, preprocessing, training, evaluation
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── requirements.txt
└── README.md
```

## Getting Started

### 1. Clone and install

```bash
git clone <your-repo-url>
cd mental-health-score-predictor
pip install -r requirements.txt
```

`requirements.txt`:

```
fastapi
uvicorn
pandas
joblib
scikit-learn==1.8.0
```

> The model was saved with scikit-learn 1.8.0. Use the same version to avoid loading warnings.

### 2. Start the API

Run this from the project folder:

```bash
uvicorn main:app --reload
```

- API: http://127.0.0.1:8000
- Interactive docs: http://127.0.0.1:8000/docs

### 3. Open the frontend

Open `frontend/index.html` in your browser. In `frontend/script.js`, make sure the API address matches your server:

```js
const API_BASE = "http://127.0.0.1:8000";
```

## API Reference

### `POST /predict`

**Request**

```json
{
  "age": 21,
  "gender": "Male",
  "country": "Canada",
  "academic_level": "Undergraduate",
  "most_used_platform": "Instagram",
  "purpose_of_use": "Entertainment",
  "avg_daily_usage_hours": 4.6,
  "daily_unlocks": 166,
  "study_hours": 4.0,
  "physical_activity_hours": 1.8,
  "sleep_hours_per_night": 6.7,
  "stress_level": "Medium"
}
```

**Response**

```json
{
  "predicted_mental_health_score": 6.74
}
```

**Allowed values**

| Field | Allowed values |
|---|---|
| age | Integer, 10–100 (the model was trained on ages 18–24) |
| gender | Male, Female |
| country | Any text; countries outside the top 9 are treated as `Other` |
| academic_level | High School, Undergraduate, Graduate |
| most_used_platform | Facebook, Instagram, Snapchat, Twitter, YouTube, TikTok, LinkedIn, LINE, KakaoTalk, VKontakte, WhatsApp, WeChat |
| purpose_of_use | Networking, Education, Entertainment, News |
| avg_daily_usage_hours, study_hours, physical_activity_hours, sleep_hours_per_night | Number, 0–24 |
| daily_unlocks | Integer, 0 or more |
| stress_level | Low, Medium, High, Very High |

Invalid input returns a `422` response with the field that failed validation.

## Limitations

- The training data covers ages 18–24 only, so predictions outside that range are unreliable.
- The data comes from one survey dataset and shows correlation, not causation.
- The model shows a noticeable gap between training and test performance.
- Predictions are estimates for learning purposes and should not be used for any health decision.

## Future Improvements

- Deploy the API (for example on Render) and host the frontend online
- Try other models such as Gradient Boosting and compare with cross-validation
- Reduce overfitting through stronger regularisation or more data
- Add automated tests for the API

## Author

**Aryan Chaudhari**
[GitHub](https://github.com/Aryan-chaudhari-107) · [LinkedIn](https://linkedin.com/in/aryan-chaudhari-aka-bunny)

## Acknowledgements

- Dataset: _add the source and author here_
- Frontend design: _credit the template or source here if you used one_