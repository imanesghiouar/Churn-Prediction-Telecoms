import os
import sys
import time
import argparse
import pandas as pd

# Ajouter le répertoire parent au chemin pour pouvoir importer les modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.config import Config
from utils.logger import get_logger
from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer
from src.model_evaluation import ModelEvaluator

def parse_arguments():
    """
    Parse les arguments de ligne de commande
    
    Returns:
        argparse.Namespace: Arguments parsés
    """
    parser = argparse.ArgumentParser(description='Pipeline de prédiction de churn')
    
    parser.add_argument(
        '--config', 
        type=str, 
        default=None,
        help='Chemin vers le fichier de configuration YAML'
    )
    
    parser.add_argument(
        '--data', 
        type=str, 
        default=None,
        help='Chemin vers le fichier de données CSV'
    )
    
    parser.add_argument(
        '--output', 
        type=str, 
        default=None,
        help='Répertoire de sortie pour les modèles et résultats'
    )
    
    parser.add_argument(
        '--models', 
        type=str, 
        nargs='+',
        default=['logistic_regression', 'random_forest', 'neural_network'],
        choices=['logistic_regression', 'random_forest', 'neural_network'],
        help='Liste des modèles à entraîner'
    )
    
    parser.add_argument(
        '--skip-preprocessing', 
        action='store_true',
        help='Sauter l\'étape de prétraitement des données'
    )
    
    parser.add_argument(
        '--skip-feature-engineering', 
        action='store_true',
        help='Sauter l\'étape d\'ingénierie des fonctionnalités'
    )
    
    return parser.parse_args()

def run_pipeline():
    """
    Exécute le pipeline complet de prédiction de churn
    """
    # Parsing des arguments
    args = parse_arguments()
    
    # Création de la configuration
    config = Config(args.config)
    
    # Configuration du logger
    logger = get_logger(config.get('paths.logs_dir'))
    
    # Affichage des informations de début
    logger.info("====== Début du Pipeline de Prédiction de Churn ======")
    start_time = time.time()
    
    # Étape 1: Prétraitement des données
    logger.info("Étape 1: Prétraitement des données")
    preprocessor = DataPreprocessor(config)
    
    # Chargement des données
    data_path = args.data or os.path.join(config.get('paths.data_dir'), 'telecom_churn.csv')
    data = preprocessor.load_data(data_path)
    
    # Préparation des données
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(data)
    
    if not args.skip_preprocessing:
        # Transformation des données
        X_train_transformed, X_test_transformed = preprocessor.transform_data(X_train, X_test)
        
        # Sauvegarde du préprocesseur
        preprocessor.save_preprocessor()
    else:
        logger.info("Étape de prétraitement ignorée")
        X_train_transformed, X_test_transformed = X_train, X_test
    
    # Étape 2: Ingénierie des fonctionnalités
    logger.info("Étape 2: Ingénierie des fonctionnalités")
    
    if not args.skip_feature_engineering:
        feature_engineer = FeatureEngineer(config)
        
        # Application de l'ingénierie des fonctionnalités
        X_train_features = feature_engineer.fit_transform(X_train_transformed)
        X_test_features = feature_engineer.transform(X_test_transformed)
        
        # Sauvegarde du pipeline de fonctionnalités
        feature_engineer.save_pipeline()
    else:
        logger.info("Étape d'ingénierie des fonctionnalités ignorée")
        X_train_features, X_test_features = X_train_transformed, X_test_transformed
    
    # Étape 3: Entraînement des modèles
    logger.info("Étape 3: Entraînement des modèles")
    model_trainer = ModelTrainer(config)
    
    # Dictionnaire pour stocker les modèles entraînés
    trained_models = {}
    
    # Entraînement des modèles sélectionnés
    if 'logistic_regression' in args.models:
        logger.info("Entraînement du modèle de régression logistique")
        lr_model = model_trainer.train_logistic_regression(X_train_features, y_train)
        trained_models['logistic_regression'] = lr_model
        model_trainer.save_model(lr_model, 'logistic_regression')
    
    if 'random_forest' in args.models:
        logger.info("Entraînement du modèle de forêt aléatoire")
        rf_model = model_trainer.train_random_forest(X_train_features, y_train)
        trained_models['random_forest'] = rf_model
        model_trainer.save_model(rf_model, 'random_forest')
    
    if 'neural_network' in args.models:
        logger.info("Entraînement du modèle de réseau de neurones")
        nn_model, _ = model_trainer.train_neural_network(X_train_features, y_train)
        trained_models['neural_network'] = nn_model
        model_trainer.save_model(nn_model, 'neural_network')
    
    # Étape 4: Évaluation des modèles
    logger.info("Étape 4: Évaluation des modèles")
    evaluator = ModelEvaluator(config)
    
    # Évaluation individuelle des modèles
    for name, model in trained_models.items():
        logger.info(f"Évaluation du modèle: {name}")
        evaluator.evaluate_model(model, X_test_features, y_test, model_name=name)
    
    # Comparaison des modèles s'il y en a plusieurs
    if len(trained_models) > 1:
        logger.info("Comparaison des modèles")
        comparison = evaluator.compare_models(trained_models, X_test_features, y_test)
        logger.info(f"Résultats de la comparaison:\n{comparison}")
    
    # Affichage du temps d'exécution
    end_time = time.time()
    execution_time = end_time - start_time
    logger.info(f"Temps d'exécution total: {execution_time:.2f} secondes")
    
    logger.info("====== Fin du Pipeline de Prédiction de Churn ======")

if __name__ == "__main__":
    run_pipeline()