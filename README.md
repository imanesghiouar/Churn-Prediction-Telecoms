# **Churn Prediction – Télécoms**

## **Project Overview**

This project is a complete machine learning pipeline built with **PySpark**, **Pandas**, and **Keras** for predicting customer churn in the telecommunications sector. It includes data preprocessing, exploratory data analysis, feature engineering, model training (Logistic Regression, Random Forest, and Neural Networks), and final evaluation.

The goal is to predict whether a customer will churn based on their historical data, such as service usage, billing, and customer demographics. The technologies used are **PySpark** for scalable data handling, **Pandas** for data exploration and preprocessing, classical machine learning models (**Logistic Regression** and **Random Forest**), and deep learning models using **Keras**.

---

## **Project Structure**

The project is organized into several folders to ensure clean modularity:

- **data/**: Contains the raw dataset (`telecom_churn.csv`).
- **notebooks/**: Includes exploratory notebooks for data analysis, cleaning, model training, and evaluation.
- **src/**: Includes source Python scripts for preprocessing, feature engineering, model training, evaluation, and Keras model building.
- **models/**: Saves the trained machine learning and deep learning models.
- **utils/**: Contains utility scripts for logging and configuration.
- **scripts/**: Holds the main runnable script (`run_pipeline.py`) to execute the entire machine learning workflow.

---

## **Data Pipeline**

The data pipeline starts with loading the raw dataset using **PySpark**. Data cleaning steps include handling missing values, encoding categorical variables, and removing unnecessary features. Feature engineering steps involve preparing the input features for machine learning models, including optional scaling and transformation.

The data is split into training and testing sets, with 80% for training and 20% for testing. Then, three models are trained: a **Logistic Regression** model and a **Random Forest Classifier** using PySpark's MLlib library, and a **Neural Network** built with **Keras** (Sequential API).

---

## **Deep Learning Model (Keras)**

A simple **Multilayer Perceptron (MLP)** architecture is used for the churn prediction task. The architecture consists of:
- An input layer,
- One or two hidden layers with **ReLU** activations,
- An output layer with **sigmoid** activation for binary classification.

The model uses **Binary Crossentropy** as the loss function and the **Adam optimizer** for training. **EarlyStopping** is applied during training to prevent overfitting by monitoring the validation loss. The trained **Keras model** is saved as a `.h5` file for future use.

---

## **Model Evaluation**

Each model is evaluated based on several standard metrics:
- **Accuracy**,
- **Precision**,
- **Recall**,
- **F1-score**,
- **Confusion matrix**,
- **ROC-AUC score**.

These evaluation metrics help in selecting the best performing model. Typically, the **Keras neural network** achieved the highest **ROC-AUC** score, closely followed by the **Random Forest** model. All models are saved after training:
- Machine learning models are saved in `.pkl` format,
- The Keras model is saved in `.h5` format.

Installation and Running
To install and run the project, clone the repository, create a virtual environment, install the requirements listed in requirements.txt, and then run the full pipeline with python scripts/run_pipeline.py. You can also use the Jupyter notebooks for more detailed exploration or testing.

Requirements
The key Python packages used in this project are PySpark, Pandas, Scikit-learn, TensorFlow, Keras, Matplotlib, Seaborn, and Joblib. All dependencies are listed in the requirements.txt file.

Notes
This project highlights how PySpark and Pandas can be combined for flexible big data and small data handling, and how Keras can easily integrate into such a pipeline for building deep learning models. The pipeline is modular and scalable, allowing easy future improvements such as hyperparameter tuning, adding feature selection techniques, or experimenting with deeper neural network architectures.

Contributing
Contributions are welcome! You can fork the repository, create a new branch, and submit a pull request. Improvements related to feature engineering, model optimization, or deep learning enhancements are appreciated.

Author
Built by Imane Sghiouar,Engineering Student, passionate about Data Science, Machine Learning, Deep Learning, and Big Data technologies.