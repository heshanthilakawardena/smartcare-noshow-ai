import joblib
import shap
import matplotlib.pyplot as plt
from src.preprocess import load_and_preprocess_data

def generate_shap_analysis():
    # Load test data and trained pipeline
    _, X_test, _, y_test, _, _ = load_and_preprocess_data('data/smartcare_ai_dataset_1000.csv')
    pipeline = joblib.load('models/xgboost_model.joblib')
    
    preprocessor = pipeline.named_steps['preprocessor']
    model = pipeline.named_steps['classifier']
    
    # Transform test set features
    X_test_transformed = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()
    
    # Compute SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test_transformed)
    
    # Generate and save SHAP Summary Plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test_transformed, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig('models/shap_summary.png', dpi=300)
    plt.close()
    print("Successfully generated and saved SHAP summary plot to models/shap_summary.png")

if __name__ == '__main__':
    generate_shap_analysis()