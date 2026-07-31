<p align="center">
  <img src="assets/SmartCare.jpg" alt="thmbanner" width="100%">
</p>

# SmartCare Hospital AI Agent Console

Our AI powered prediction system uses Machine Learning models to identify patients who are more likely to miss their appointments before the scheduled date. It helps healthcare providers understand potential no-shows, take early actions such as reminders or follow-ups, and improve appointment management.

[ ](assets/SmartCare Ai Console.png)

### Usage
- Download Code
  ```
  git clone -b main --single-branch https://github.com/heshanthilakawardena/smartcare-noshow-ai.git
  ```
- Install Esenntial libraries      
  ```
  pip install requirements.txt
  ```
- Run code in terminal
  ```
  python app.py
  ```
### Suported Models
- Logistic Regression
- KNN
- XGBosst
- Random Forest
- Desicion Tree

### Application Work Flow
```
                 User
                  |
                  |
              Boostrap UI
                  |
          Select Model Dropdown
                  |
                  |
            POST JSON Request
                  |
                  ↓
            Flask API
                  |
        --------------------
        |                  |
 Logistic Regression      KNN
        |                  |
 Smartcare_Logistic     Smartcare_KNN
        |                  |
        --------------------
                  |
                  ↓
      Smartcare_Preprocessor.joblib
                  |
                  ↓
              Prediction
                  |
                  ↓
  SHAP + Smartcare_Preprocessor.joblib
                  |
                  ↓
          Result back to UI
```

### Contibutors
[Hehsan Thilakawadhana](https://github.com/heshanthilakawardena)
[Zahra Ismail](https://github.com/Zahra-Ismail)
[Binara Wijewickrama](https://github.com/binarays)