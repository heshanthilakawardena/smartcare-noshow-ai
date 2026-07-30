# SmartCare Hospital AI

Our AI-powered prediction system uses Machine Learning models such as Logistic Regression and K-Nearest Neighbors (KNN) to identify patients who are more likely to miss their appointments before the scheduled date. It helps healthcare providers understand potential no-shows, take early actions such as reminders or follow-ups, and improve appointment management.

#### Task
    [x] Training Logistic Regression model
    [x] Training KNN model
    [X] Track both model using `MlFlow`
    [x] Save Model Artifacts in [**Dagshub**](https://dagshub.com/heshan.thilakawardena/smartcare-noshow-ai/experiments)
    [ ] Data versioning using `dvc`
    [x] Frountend using `steamlit` + `Flask` 

# Logistic Regression Model

### Confusion Matrix
![Model Confustion mattrix](reports/Logistic_Regression/confusion_matrix.png)

### ROC
![Model Confustion mattrix](reports/Logistic_Regression/roc_curve.png)

# Logistic Regression Model

### Confusion Matrix
![Model Confustion mattrix](reports/KNN/confusion_matrix.png)

### ROC
![Model Confustion mattrix](reports/KNN/roc_curve.png)