import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os
import joblib

from utils.logger import get_logger

#still i have to adapt the names of the datasets to the ones that i have 

logger = get_logger()

class DataPreprocessor:
    """Classe pour le prétraitement des données de churn"""
    
    def __init__(self, config):
        """
        Initialise le préprocesseur
        
        Args:
            config: Instance de la classe Config
        """
        self.config = config
        self.numeric_transformer = None
        self.categorical_transformer = None
        self.preprocessor = None
        self.label_encoder = None
    
    def load_data(self, file_path=None):
        """
        Charge les données depuis un fichier CSV
        
        Args:
            file_path (str, optional): Chemin vers le fichier de données
        
        Returns:
            pd.DataFrame: Dataframe contenant les données
        """
        if file_path is None:
            file_path = os.path.join(self.config.get('paths.data_dir'), 'telecom_churn.csv')
        
        logger.info(f"Chargement des données depuis {file_path}")
        
        try:
            data = pd.read_csv(file_path)
            logger.info(f"Données chargées avec succès: {data.shape[0]} lignes et {data.shape[1]} colonnes")
            return data
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données: {str(e)}")
            raise
    
    def identify_columns(self, data):
        """
        Identifie les colonnes numériques et catégorielles
        
        Args:
            data (pd.DataFrame): Données à analyser
        
        Returns:
            tuple: (liste de colonnes numériques, liste de colonnes catégorielles)
        """
        numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Exclusion de la colonne cible et des identifiants si présents
        exclude_cols = ['customerID']
        target_col = 'Churn'
        
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        if target_col in categorical_cols:
            categorical_cols.remove(target_col)
        
        for col in exclude_cols:
            if col in numeric_cols:
                numeric_cols.remove(col)
            if col in categorical_cols:
                categorical_cols.remove(col)
        
        logger.info(f"Colonnes numériques identifiées: {numeric_cols}")
        logger.info(f"Colonnes catégorielles identifiées: {categorical_cols}")
        
        return numeric_cols, categorical_cols
    
    def create_preprocessor(self, numeric_cols, categorical_cols):
        """
        Crée un pipeline de prétraitement
        
        Args:
            numeric_cols (list): Liste des colonnes numériques
            categorical_cols (list): Liste des colonnes catégorielles
        """
        scaling_method = self.config.get('preprocessing.scaling', 'standard')
        
        if scaling_method == 'standard':
            self.numeric_transformer = Pipeline(steps=[
                ('scaler', StandardScaler())
            ])
        elif scaling_method == 'minmax':
            self.numeric_transformer = Pipeline(steps=[
                ('scaler', MinMaxScaler())
            ])
        else:
            self.numeric_transformer = Pipeline(steps=[
                ('passthrough', 'passthrough')
            ])
        
        self.categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', self.numeric_transformer, numeric_cols),
                ('cat', self.categorical_transformer, categorical_cols)
            ],
            remainder='drop'
        )
        
        logger.info("Préprocesseur créé avec succès")
    
    def encode_target(self, y):
        """
        Encode la variable cible
        
        Args:
            y (pd.Series): Variable cible
        
        Returns:
            np.ndarray: Variable cible encodée
        """
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        logger.info(f"Classes de la variable cible: {self.label_encoder.classes_}")
        return y_encoded
    
    def prepare_data(self, data, target_col='Churn'):
        """
        Prépare les données pour la modélisation
        
        Args:
            data (pd.DataFrame): Données brutes
            target_col (str): Nom de la colonne cible
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        logger.info("Préparation des données...")
        
        # Séparation des caractéristiques et de la cible
        X = data.drop(columns=[target_col])
        y = data[target_col]
        
        # Identification des colonnes
        numeric_cols, categorical_cols = self.identify_columns(X)
        
        # Création du préprocesseur
        self.create_preprocessor(numeric_cols, categorical_cols)
        
        # Encodage de la variable cible
        y_encoded = self.encode_target(y)
        
        # Division en ensembles d'entraînement et de test
        test_size = self.config.get('preprocessing.test_size', 0.2)
        random_state = self.config.get('preprocessing.random_state', 42)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
        )
        
        logger.info(f"Division train/test: {X_train.shape[0]} exemples d'entraînement, {X_test.shape[0]} exemples de test")
        
        return X_train, X_test, y_train, y_test
    
    def transform_data(self, X_train, X_test):
        """
        Applique le prétraitement aux données
        
        Args:
            X_train (pd.DataFrame): Données d'entraînement
            X_test (pd.DataFrame): Données de test
        
        Returns:
            tuple: (X_train_transformed, X_test_transformed)
        """
        logger.info("Application du préprocesseur aux données...")
        
        X_train_transformed = self.preprocessor.fit_transform(X_train)
        X_test_transformed = self.preprocessor.transform(X_test)
        
        logger.info(f"Données transformées: {X_train_transformed.shape[1]} caractéristiques")
        
        return X_train_transformed, X_test_transformed
    
    def save_preprocessor(self, output_path=None):
        """
        Sauvegarde le préprocesseur
        
        Args:
            output_path (str, optional): Chemin pour sauvegarder le préprocesseur
        """
        if output_path is None:
            output_path = os.path.join(self.config.get('paths.models_dir'), 'preprocessor.pkl')
        
        # Création du répertoire si nécessaire
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        joblib.dump({
            'preprocessor': self.preprocessor,
            'label_encoder': self.label_encoder
        }, output_path)
        
        logger.info(f"Préprocesseur sauvegardé à {output_path}")
    
    def load_preprocessor(self, input_path=None):
        """
        Charge un préprocesseur sauvegardé
        
        Args:
            input_path (str, optional): Chemin du préprocesseur sauvegardé
        """
        if input_path is None:
            input_path = os.path.join(self.config.get('paths.models_dir'), 'preprocessor.pkl')
        
        logger.info(f"Chargement du préprocesseur depuis {input_path}")
        
        preprocessor_data = joblib.load(input_path)
        self.preprocessor = preprocessor_data['preprocessor']
        self.label_encoder = preprocessor_data['label_encoder']
        
        logger.info("Préprocesseur chargé avec succès")