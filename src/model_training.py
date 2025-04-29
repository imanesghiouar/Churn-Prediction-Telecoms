# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report, roc_curve, auc
)
from xgboost import XGBClassifier
import pickle
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load and combine both datasets"""
    print("Loading datasets...")
    try:
        df_bigml_20 = pd.read_csv('churn-bigml-20.csv')
        df_bigml_80 = pd.read_csv('churn-bigml-80.csv')
        df = pd.concat([df_bigml_20, df_bigml_80], axis=0)
        print(f"Data loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(df):
    """Preprocess the data for model training"""
    print("Preprocessing data...")
    
    # Convert target variable
    le = LabelEncoder()
    df['Churn'] = le.fit_transform(df['Churn'])
    
    # Handle categorical variables
    # Convert yes/no features to binary
    for col in ['International plan', 'Voice mail plan']:
        df[col] = df[col].map({'Yes': 1, 'No': 0})
    
    # One-hot encode state
    df = pd.get_dummies(df, columns=['State'], drop_first=True)
    
    # Feature engineering
    df['Minutes_per_call'] = df['Total day minutes'] / df['Total day calls']
    df['Charge_per_minute'] = df['Total day charge'] / df['Total day minutes']
    df['Voicemail_to_call_ratio'] = df['Number of voice mail messages'] / df['Total day calls']
    
    # Fill NaN values created by division by zero
    df.fillna(0, inplace=True)
    
    # Remove redundant features
    cols_to_drop = ['Total day charge', 'Total eve charge', 'Total night charge', 'Total intl charge']
    df = df.drop(columns=cols_to_drop)
    
    # Split the data
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Standardize numerical features
    numerical_cols = [col for col in X.select_dtypes(include=['int64', 'float64']).columns]
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Preprocessing complete. Training set: {X_train.shape}, Test set: {X_test.shape}")
    return X_train, X_test, y_train, y_test, X, y

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Train and evaluate a model"""
    print(f"\nTraining {model_name}...")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"{model_name} Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # If model can predict probabilities, calculate ROC AUC
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
        print(f"ROC AUC: {roc_auc:.4f}")
    
    # Create confusion matrix visualization
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f"{model_name.replace(' ', '_').lower()}_confusion_matrix.png")
    plt.close()
    
    # Create classification report
    report = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(report)
    
    # Return model and metrics in a dictionary
    results = {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }
    
    if hasattr(model, "predict_proba"):
        results['roc_auc'] = roc_auc
    
    return results

def tune_hyperparameters(best_model_name, X_train, y_train):
    """Tune hyperparameters for the best performing model"""
    print(f"\nPerforming hyperparameter tuning for {best_model_name}...")
    
    if best_model_name == 'Random Forest':
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    elif best_model_name == 'Gradient Boosting':
        model = GradientBoostingClassifier(random_state=42)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0]
        }
    elif best_model_name == 'XGBoost':
        model = XGBClassifier(random_state=42)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
    else:  # Logistic Regression
        model = LogisticRegression(random_state=42)
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2', 'elasticnet', None],
            'solver': ['newton-cg', 'lbfgs', 'liblinear'],
            'max_iter': [100, 500, 1000]
        }
    
    # Perform grid search
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)
    
    return grid_search.best_estimator_

def analyze_feature_importance(model, X, model_name):
    """Analyze and visualize feature importance"""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title(f'Feature Importances - {model_name}')
        plt.bar(range(X.shape[1]), importances[indices], align='center')
        plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
        plt.tight_layout()
        plt.savefig(f"{model_name.replace(' ', '_').lower()}_feature_importance.png")
        plt.close()
        
        # Print top 10 features
        print("\nTop 10 most important features:")
        for i in range(min(10, len(indices))):
            print(f"{i+1}. {X.columns[indices[i]]}: {importances[indices[i]]:.4f}")

def save_model(model, filename):
    """Save the model to disk"""
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved as '{filename}'")

def main():
    """Main function to run the entire process"""
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Preprocess data
    X_train, X_test, y_train, y_test, X, y = preprocess_data(df)
    
    # Train and evaluate multiple models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42)
    }
    
    results = {}
    for model_name, model in models.items():
        results[model_name] = evaluate_model(model, X_train, X_test, y_train, y_test, model_name)
    
    # Find the best model based on F1 score
    best_model_name = max(results, key=lambda k: results[k]['f1'])
    print(f"\nBest model based on F1 score: {best_model_name} (F1: {results[best_model_name]['f1']:.4f})")
    
    # Tune hyperparameters for the best model
    best_model = tune_hyperparameters(best_model_name, X_train, y_train)
    
    # Evaluate the tuned model
    tuned_results = evaluate_model(best_model, X_train, X_test, y_train, y_test, f"Tuned {best_model_name}")
    
    # Analyze feature importance
    analyze_feature_importance(best_model, X, f"Tuned {best_model_name}")
    
    # Save the model
    save_model(best_model, 'best_churn_model.pkl')
    
    # Print final comparison
    print("\nModel Performance Comparison:")
    for model_name, result in results.items():
        print(f"{model_name}: F1 = {result['f1']:.4f}, Accuracy = {result['accuracy']:.4f}")
    
    print(f"\nTuned {best_model_name}: F1 = {tuned_results['f1']:.4f}, Accuracy = {tuned_results['accuracy']:.4f}")
    
    print("\nTraining complete!")

if __name__ == "__main__":
    main()