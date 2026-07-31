# SmartCare Agent Console Developer

Hi, I’m [Binara](https://github.com/binarays). I am responsible for training and implementing the **Logistic Regression** and **K-Nearest Neighbors (KNN)** models for this project. I also developed the **Flask server** to connect the **Machine Learning models** with the application and worked on the client side development to provide a user friendly interface.

### Task list
- [x] Training Logistic Regression model
- [x] Training KNN model
- [X] Track both model using `MlFlow`
- [x] Save Model Artifacts in [**Dagshub**](https://dagshub.com/heshan.thilakawardena/smartcare-noshow-ai/experiments)
- [ ] Data versioning using `dvc`
- [x] `SHAP` implement for Explanations
- [x] Frountend using `Boostrap` + `Flask`
- [ ] `exe` Ready

### For Developers
- Download Code
- Install Esenntial libraries      
  ```
  pip install requirements.txt
  ```
- Start Training
  ```
  python main.py
  ```

# Logistic Regression Model

### Confusion Matrix
![Model Confustion mattrix](reports/Logistic_Regression/confusion_matrix.png)

### ROC
![Model Confustion mattrix](reports/Logistic_Regression/roc_curve.png)

# KNN Model

### Confusion Matrix
![Model Confustion mattrix](reports/KNN/confusion_matrix.png)

### ROC
![Model Confustion mattrix](reports/KNN/roc_curve.png)

# Folder Structure

```
│
├── data/
│ ├── processed/
│ └── raw/
│
├── models/
│
│
├── reports/
│ │
│ ├── KNN/
│ │
│ └── Logistic_Regression/
│
├── src/
│ │
│ ├── preprocessing/
│ │ ├── CleanData.py
│ │ ├── LoadData.py
│ │ ├── preprocess.py
│ │ └── SaveObjects.py
│ │
│ ├── training/
│ │ ├── evaluate.py
│ │ └── train.py
│ │
│ └── utils/
│ └── mlflow_tracker.py
│ └── path.py
│
├── main.py
└── requirements.txt
```


## Contributor

- [Binara Wijewickrama](https://github.com/binarays)